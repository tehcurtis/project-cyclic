import asyncio
import hashlib
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import chromadb
from chromadb.config import Settings
from litellm import aembedding
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .agent import AgentResponse
from .sandbox import ExecutionResult


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

    COLLECTION_NAME_V1 = "cyclic_semantic_cache_v1"
    COLLECTION_NAME_V2 = "cyclic_semantic_cache_v2"
    DEFAULT_THRESHOLD = 0.85

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        similarity_threshold: float = DEFAULT_THRESHOLD,
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        store_prompt: bool = False,
        store_outputs: bool = False,
    ):
        """
        Initialize the semantic cache.

        Args:
            cache_dir: Directory for ChromaDB persistence (defaults to ~/.cyclic/cache/)
            similarity_threshold: Minimum similarity score for cache hits (0.0-1.0)
            embedding_model: LiteLLM embedding model identifier
            api_key: API key for embedding service (defaults to env var)
            store_prompt: If True, store raw prompt in metadata (default: False)
            store_outputs: If True, store stdout/stderr in metadata (default: False)
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cyclic/cache/")

        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.api_key = api_key or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.store_prompt = store_prompt
        self.store_outputs = store_outputs

        self.client = chromadb.PersistentClient(
            path=str(self.cache_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        self._chroma_lock = asyncio.Lock()
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Get or create collection with cosine distance pinned."""
        try:
            # Try to get v2 collection (with cosine pinned)
            try:
                collection = self.client.get_collection(name=self.COLLECTION_NAME_V2)
                # Verify it has cosine distance pinned (if metadata is available)
                try:
                    metadata = getattr(collection, "metadata", None) or {}
                    if metadata.get("distance_space") != "cosine":
                        logger.warning(
                            f"Collection {self.COLLECTION_NAME_V2} exists but distance space is not pinned to cosine. "
                            "Recreating with cosine distance."
                        )
                        self.client.delete_collection(name=self.COLLECTION_NAME_V2)
                        collection = None
                except AttributeError:
                    logger.debug(f"Collection {self.COLLECTION_NAME_V2} metadata not accessible, assuming cosine")
            except Exception:
                collection = None

            if collection is None:
                try:
                    self.client.get_collection(name=self.COLLECTION_NAME_V1)
                    logger.warning(
                        f"Found old collection {self.COLLECTION_NAME_V1} without pinned cosine distance. "
                        f"Migrating to {self.COLLECTION_NAME_V2} with cosine distance."
                    )
                except Exception:
                    pass

                collection = self.client.create_collection(
                    name=self.COLLECTION_NAME_V2,
                    metadata={
                        "description": "Semantic cache for Cyclic code generation",
                        "distance_space": "cosine",
                    }
                )
                logger.debug(f"Created collection {self.COLLECTION_NAME_V2} with cosine distance")

            return collection

        except Exception as e:
            logger.error(f"Failed to get or create collection: {e}")
            raise

    def _compute_doc_id(self, prompt: str, code: str) -> str:
        """Compute deterministic document ID from prompt and code."""
        content = f"{prompt}\0{code}\0{self.embedding_model}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((OSError, IOError, RuntimeError)),
    )
    async def _chroma_query(self, query_embeddings, n_results=1):
        """Execute ChromaDB query in thread pool."""
        def _query_sync():
            return self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
            )
        return await asyncio.to_thread(_query_sync)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((OSError, IOError, RuntimeError)),
    )
    async def _chroma_upsert(self, ids, embeddings, metadatas):
        """Execute ChromaDB upsert in thread pool."""
        def _upsert_sync():
            try:
                if hasattr(self.collection, "upsert"):
                    self.collection.upsert(
                        ids=ids,
                        embeddings=embeddings,
                        metadatas=metadatas,
                    )
                else:
                    existing = self.collection.get(ids=ids)
                    if existing["ids"]:
                        self.collection.update(
                            ids=ids,
                            embeddings=embeddings,
                            metadatas=metadatas,
                        )
                    else:
                        self.collection.add(
                            ids=ids,
                            embeddings=embeddings,
                            metadatas=metadatas,
                        )
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["lock", "locked", "timeout", "resource temporarily unavailable", "database"]):
                    raise RuntimeError(f"Database contention: {e}") from e
                raise

        return await asyncio.to_thread(_upsert_sync)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((OSError, IOError, RuntimeError)),
    )
    async def _chroma_count(self):
        """Execute ChromaDB count in thread pool."""
        def _count_sync():
            return self.collection.count()
        return await asyncio.to_thread(_count_sync)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
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

        async with self._chroma_lock:
            try:
                query_embedding = await self._get_embedding(prompt)

                results = await self._chroma_query([query_embedding], n_results=1)

                if not results["ids"] or not results["ids"][0]:
                    return None

                distance = results["distances"][0][0]
                similarity = max(0.0, min(1.0, 1.0 - distance))

                if similarity < self.similarity_threshold:
                    return None

                metadata = results["metadatas"][0][0]

                prompt_value = metadata.get("prompt") if self.store_prompt else None
                if prompt_value is None:
                    prompt_value = metadata.get("prompt_sha256")

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

        async with self._chroma_lock:
            try:
                embedding = await self._get_embedding(prompt)

                doc_id = self._compute_doc_id(prompt, response.code)

                metadata = {
                    "code": response.code,
                    "reasoning": response.reasoning,
                    "confidence": str(response.confidence),
                    "exit_code": str(result.exit_code),
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
        async with self._chroma_lock:
            try:
                def _clear_sync():
                    self.client.delete_collection(name=self.COLLECTION_NAME_V2)
                    self.collection = self._get_or_create_collection()

                await asyncio.to_thread(_clear_sync)
                logger.info("Cache cleared")
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
                raise

    async def get_stats(self) -> dict:
        """Get persistent cache statistics."""
        async with self._chroma_lock:
            try:
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
            except Exception as e:
                logger.error(f"Failed to get cache stats: {e}")
                return {
                    "total_entries": 0,
                    "cache_dir": str(self.cache_dir),
                    "cache_size_bytes": 0,
                }
