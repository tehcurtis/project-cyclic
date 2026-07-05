import asyncio
import hashlib
import os
import pathlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from filelock import FileLock
from litellm import aembedding
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .agent import AgentResponse
from .sandbox import ExecutionResult

# Shared retry policy for local ChromaDB operations.
_chroma_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((OSError, RuntimeError)),
)


@dataclass
class CacheHit:
    """Represents a cache hit with all relevant data."""
    prompt: Optional[str]
    code: str
    reasoning: str
    confidence: float
    execution_result: ExecutionResult
    similarity: float
    timestamp: str


class SemanticCache:
    """Semantic cache using ChromaDB and LiteLLM embeddings."""

    COLLECTION_PREFIX = "cyclic_semantic_cache"
    SCHEMA_VERSION = 4
    DEFAULT_THRESHOLD = 0.85

    # Chroma collection names must be 3-63 chars, starting/ending alphanumeric.
    _MAX_COLLECTION_NAME_LEN = 63

    @classmethod
    def _sanitize_model(cls, model: str) -> str:
        """Sanitize an embedding model id for embedding into a Chroma collection name."""
        sanitized = re.sub(r"[^a-z0-9._-]", "-", model.lower())
        prefix = f"{cls.COLLECTION_PREFIX}_v{cls.SCHEMA_VERSION}_"
        budget = cls._MAX_COLLECTION_NAME_LEN - len(prefix)
        sanitized = sanitized[:budget]
        # Name must end alphanumeric (the prefix guarantees the start).
        sanitized = sanitized.rstrip("._-")
        return sanitized or "default"

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        similarity_threshold: float = DEFAULT_THRESHOLD,
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        store_prompt: bool = False,
        store_outputs: bool = True,
    ):
        """
        Initialize the semantic cache.

        Args:
            cache_dir: Directory for ChromaDB persistence (defaults to ~/.cyclic/cache/)
            similarity_threshold: Minimum similarity score for cache hits (0.0-1.0)
            embedding_model: LiteLLM embedding model identifier
            api_key: API key for embedding service (defaults to env var)
            store_prompt: If True, persist the raw prompt in Chroma metadata (default: False).
                Regardless of this flag, the raw prompt is sent to the embedding provider on every
                cache read and write when an API key is configured.
            store_outputs: If True, store stdout/stderr in metadata (default: True)
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cyclic/cache/")

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.collection_name = (
            f"{self.COLLECTION_PREFIX}_v{self.SCHEMA_VERSION}_"
            f"{self._sanitize_model(embedding_model)}"
        )
        self.api_key = api_key or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.store_prompt = store_prompt
        self.store_outputs = store_outputs

        self.client = chromadb.PersistentClient(
            path=str(self.cache_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self._proc_lock = FileLock(
            str(self.cache_dir / ".cyclic_cache.lock"),
            timeout=10,
        )
        self._chroma_lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        self.collection: chromadb.Collection | None = None
        self._embedding_memo: tuple[str, list[float]] | None = None

    def _get_or_create_collection_sync(self):
        with self._proc_lock:
            return self._bootstrap_collection_unlocked()

    def _bootstrap_collection_unlocked(self):
        """Open or create the active collection; caller must hold ``_proc_lock``."""
        try:
            collection = self.client.get_collection(name=self.collection_name)
        except NotFoundError:
            stale = [
                c.name
                for c in self.client.list_collections()
                if c.name.startswith(self.COLLECTION_PREFIX) and c.name != self.collection_name
            ]
            if stale:
                logger.warning(
                    f"Found stale semantic cache collections {stale}. "
                    f"Active cache is {self.collection_name}; stale data is not migrated. "
                    "Run `cache clear` if you want to reclaim disk space."
                )

            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Semantic cache for Cyclic code generation",
                    "hnsw:space": "cosine",
                },
            )
            logger.debug(f"Created collection {self.collection_name} with cosine distance")

        return collection

    async def _ensure_collection(self) -> None:
        if self.collection is not None:
            return
        async with self._init_lock:
            if self.collection is None:
                self.collection = await asyncio.to_thread(self._get_or_create_collection_sync)

    def _compute_doc_id(self, prompt: str, code: str) -> str:
        """Compute deterministic document ID from prompt and code."""
        h = hashlib.sha256()
        for part in (prompt, code, self.embedding_model):
            encoded = part.encode("utf-8")
            h.update(len(encoded).to_bytes(8, "big"))
            h.update(encoded)
        return h.hexdigest()

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """Map Chroma cosine distance (1 - cos_sim, in [0, 2]) to cosine similarity clamped to [0, 1]."""
        return max(0.0, min(1.0, 1.0 - distance))

    @_chroma_retry
    async def _chroma_query(self, query_embeddings, n_results=1):
        """Execute ChromaDB query in thread pool."""

        def _query_sync():
            return self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
            )

        return await asyncio.to_thread(_query_sync)

    @_chroma_retry
    async def _chroma_upsert(self, ids, embeddings, metadatas):
        """Execute ChromaDB upsert in thread pool."""

        def _upsert_sync():
            with self._proc_lock:
                try:
                    self.collection.upsert(
                        ids=ids,
                        embeddings=embeddings,
                        metadatas=metadatas,
                    )
                except sqlite3.OperationalError as e:
                    raise RuntimeError(f"Database contention: {e}") from e

        return await asyncio.to_thread(_upsert_sync)

    @_chroma_retry
    async def _chroma_count(self):
        """Execute ChromaDB count in thread pool."""

        def _count_sync():
            return self.collection.count()

        return await asyncio.to_thread(_count_sync)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (
                APIConnectionError,
                APIError,
                RateLimitError,
                ServiceUnavailableError,
                Timeout,
                TimeoutError,
            )
        ),
    )
    async def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using LiteLLM."""
        try:
            response = await aembedding(
                model=self.embedding_model,
                input=[text],
                api_key=self.api_key,
            )
            if not response or not response.data:
                raise ValueError("Empty embedding response")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def search(self, prompt: str) -> Optional[CacheHit]:
        """
        Search for similar prompts in the cache.

        Args:
            prompt: User prompt to search for

        Returns:
            CacheHit if similar prompt found (similarity >= threshold), None otherwise
        """
        if not self.api_key:
            logger.debug("No API key for embeddings, skipping cache search")
            return None

        try:
            await self._ensure_collection()
        except Exception as e:
            logger.warning(f"Cache unavailable: {e}, falling back to normal generation")
            return None

        async with self._chroma_lock:
            try:
                query_embedding = await self._get_embedding(prompt)
                self._embedding_memo = (prompt, query_embedding)

                results = await self._chroma_query([query_embedding], n_results=1)

                if not results["ids"] or not results["ids"][0]:
                    return None

                distance = results["distances"][0][0]
                similarity = self._distance_to_similarity(float(distance))

                if similarity < self.similarity_threshold:
                    return None

                metadata = results["metadatas"][0][0]

                prompt_value: Optional[str] = None
                if self.store_prompt:
                    prompt_value = metadata.get("prompt")

                stdout_value = metadata.get("stdout", "") if self.store_outputs else ""
                stderr_value = metadata.get("stderr", "") if self.store_outputs else ""

                return CacheHit(
                    prompt=prompt_value,
                    code=metadata["code"],
                    reasoning=metadata["reasoning"],
                    confidence=float(metadata["confidence"]),
                    execution_result=ExecutionResult(
                        stdout=stdout_value,
                        stderr=stderr_value,
                        exit_code=int(metadata["exit_code"]),
                    ),
                    similarity=similarity,
                    timestamp=metadata["timestamp"],
                )

            except Exception as e:
                logger.warning(f"Cache search failed: {e}, falling back to normal generation")
                return None

    async def store(
        self,
        prompt: str,
        response: AgentResponse,
        result: ExecutionResult,
    ) -> None:
        """
        Store a successful execution in the cache.

        Args:
            prompt: User prompt
            response: Agent response with code and metadata
            result: Successful execution result (should have exit_code == 0)
        """
        if result.exit_code != 0:
            logger.debug("Skipping cache storage for failed execution")
            return

        if not self.api_key:
            logger.debug("No API key for embeddings, skipping cache storage")
            return

        try:
            await self._ensure_collection()
        except Exception as e:
            logger.warning(f"Cache unavailable: {e}, skipping cache storage")
            return

        async with self._chroma_lock:
            try:
                if self._embedding_memo is not None and self._embedding_memo[0] == prompt:
                    embedding = self._embedding_memo[1]
                else:
                    embedding = await self._get_embedding(prompt)
                    self._embedding_memo = (prompt, embedding)

                doc_id = self._compute_doc_id(prompt, response.code)

                metadata = {
                    "code": response.code,
                    "reasoning": response.reasoning,
                    "confidence": float(response.confidence),
                    "exit_code": int(result.exit_code),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "embedding_model": self.embedding_model,
                }

                if self.store_prompt:
                    metadata["prompt"] = prompt
                else:
                    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    metadata["prompt_sha256"] = prompt_hash

                if self.store_outputs:
                    metadata["stdout"] = result.stdout
                    metadata["stderr"] = result.stderr

                await self._chroma_upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    metadatas=[metadata],
                )

                logger.debug(f"Cached execution (ID: {doc_id[:16]}...)")

            except Exception as e:
                logger.warning(f"Cache storage failed: {e}")

    async def clear(self) -> None:
        """Clear all entries from the cache."""
        await self._ensure_collection()

        async with self._chroma_lock:

            def _clear_sync():
                with self._proc_lock:
                    for collection in self.client.list_collections():
                        if collection.name.startswith(self.COLLECTION_PREFIX):
                            try:
                                self.client.delete_collection(name=collection.name)
                            except NotFoundError:
                                pass
                    self.collection = self._bootstrap_collection_unlocked()

            try:
                await asyncio.to_thread(_clear_sync)
                logger.info("Cache cleared")
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
                raise

    async def get_stats(self) -> dict:
        """Get persistent cache statistics."""
        await self._ensure_collection()

        async with self._chroma_lock:
            count = await self._chroma_count()

            cache_size_bytes = 0
            try:
                for root, dirs, files in os.walk(self.cache_dir):
                    for file in files:
                        file_path = pathlib.Path(root) / file
                        try:
                            cache_size_bytes += file_path.stat().st_size
                        except OSError:
                            pass
            except Exception as e:
                logger.debug(f"Could not calculate cache size: {e}")

            return {
                "total_entries": count,
                "cache_dir": str(self.cache_dir),
                "cache_size_bytes": cache_size_bytes,
            }
