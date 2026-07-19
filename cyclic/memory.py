import asyncio
import hashlib
import os
import pathlib
import sqlite3

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from filelock import FileLock
from loguru import logger

from .chroma_utils import chroma_retry, new_persistent_client, utc_now_iso


class Memory:
    """Semantic memory of past successful solutions, backed by ChromaDB.

    Mirrors SemanticCache's async offload, locking, retry, and privacy
    patterns (see cyclic/cache.py), but embeds locally via ChromaDB's default
    embedding function rather than a remote provider.
    """

    DEFAULT_MIN_SIMILARITY = 0.45

    def __init__(
        self,
        path: str | None = None,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        store_prompt: bool = False,
    ) -> None:
        persist = pathlib.Path(path or os.path.expanduser("~/.cyclic/memory"))
        persist.mkdir(parents=True, exist_ok=True)

        self.client = new_persistent_client(persist)
        # Cosine distance is pinned explicitly: Chroma defaults to L2, but all
        # similarity math below assumes cosine (see SemanticCache, cache.py).
        self.collection = self.client.get_or_create_collection(
            "solutions", metadata={"hnsw:space": "cosine"}
        )
        # Held so we can embed manually and control whether the raw prompt
        # document is persisted (chromadb's default EF does local ONNX
        # inference and downloads its model on first use).
        self._embedding_fn = DefaultEmbeddingFunction()

        self._proc_lock = FileLock(str(persist / ".cyclic_memory.lock"), timeout=10)
        self.min_similarity = min_similarity
        self.store_prompt = store_prompt

    async def remember(self, prompt: str, code: str, reasoning: str) -> None:
        """Store a successful solution for future recall."""
        try:
            embedding = (await asyncio.to_thread(self._embedding_fn, [prompt]))[0]

            doc_id = self._compute_doc_id(prompt, code)
            metadata = {
                "code": code,
                "reasoning": reasoning,
                "timestamp": utc_now_iso(),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }

            await self._chroma_upsert(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[prompt] if self.store_prompt else None,
            )
        except Exception as e:
            logger.warning(f"Failed to store solution in memory: {e}")

    async def recall(self, prompt: str, k: int = 3) -> list[dict]:
        """Recall up to k past solutions similar to prompt."""
        try:
            query_embedding = (await asyncio.to_thread(self._embedding_fn, [prompt]))[0]

            results = await self._chroma_query(query_embedding, n_results=k)

            ids = results.get("ids") or []
            if not ids or not ids[0]:
                return []

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            hits = []
            for document, metadata, distance in zip(documents, metadatas, distances, strict=True):
                similarity = max(0.0, min(1.0, 1.0 - distance))
                if similarity < self.min_similarity:
                    continue
                hits.append({"prompt": document, "code": metadata["code"]})
            return hits
        except Exception as e:
            logger.warning(f"Memory recall failed: {e}")
            return []

    async def recall_context(self, prompt: str, k: int = 3) -> str | None:
        """Return a rendered reference block of similar past solutions, or None."""
        hits = await self.recall(prompt, k=k)
        if not hits:
            return None

        blocks = []
        for hit in hits:
            if hit["prompt"]:
                blocks.append(f"Problem: {hit['prompt']}\nSolution:\n```python\n{hit['code']}\n```")
            else:
                blocks.append(
                    "A similar problem you solved before — solution:\n"
                    f"```python\n{hit['code']}\n```"
                )

        return (
            "Here are similar problems you solved successfully before. "
            "Use them as reference if helpful.\n\n" + "\n\n".join(blocks)
        )

    async def count(self) -> int:
        """Return the number of stored solutions."""
        return await asyncio.to_thread(self.collection.count)

    async def clear(self) -> None:
        """Clear all stored solutions."""

        def _clear_sync():
            with self._proc_lock:
                self.client.delete_collection("solutions")
                self.collection = self.client.get_or_create_collection(
                    "solutions", metadata={"hnsw:space": "cosine"}
                )

        await asyncio.to_thread(_clear_sync)

    def _compute_doc_id(self, prompt: str, code: str) -> str:
        """Compute deterministic document ID from prompt and code."""
        h = hashlib.sha256()
        for part in (prompt, code):
            encoded = part.encode("utf-8")
            h.update(len(encoded).to_bytes(8, "big"))
            h.update(encoded)
        return h.hexdigest()

    @chroma_retry
    async def _chroma_query(self, query_embedding, n_results):
        """Execute ChromaDB query in thread pool."""

        def _query_sync():
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

        return await asyncio.to_thread(_query_sync)

    @chroma_retry
    async def _chroma_upsert(self, ids, embeddings, metadatas, documents=None):
        """Execute ChromaDB upsert in thread pool."""

        def _upsert_sync():
            with self._proc_lock:
                try:
                    self.collection.upsert(
                        ids=ids,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        documents=documents,
                    )
                except sqlite3.OperationalError as e:
                    raise RuntimeError(f"Database contention: {e}") from e

        return await asyncio.to_thread(_upsert_sync)
