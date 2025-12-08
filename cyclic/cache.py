import json
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import chromadb
from chromadb.config import Settings
from litellm import aembedding
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from .agent import AgentResponse
from .sandbox import ExecutionResult


@dataclass
class CacheHit:
    """Represents a cache hit with all relevant data."""
    prompt: str
    code: str
    reasoning: str
    confidence: float
    execution_result: ExecutionResult
    similarity: float
    timestamp: str


class SemanticCache:
    """Semantic cache using ChromaDB and LiteLLM embeddings."""

    COLLECTION_NAME = "cyclic_semantic_cache_v1"
    DEFAULT_THRESHOLD = 0.85

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        similarity_threshold: float = DEFAULT_THRESHOLD,
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the semantic cache.

        Args:
            cache_dir: Directory for ChromaDB persistence (defaults to ~/.cyclic/cache/)
            similarity_threshold: Minimum similarity score for cache hits (0.0-1.0)
            embedding_model: LiteLLM embedding model identifier
            api_key: API key for embedding service (defaults to env var)
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cyclic/cache/")

        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.api_key = api_key or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")

        self.client = chromadb.PersistentClient(
            path=str(self.cache_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Semantic cache for Cyclic code generation"}
        )

        self._hit_count = 0
        self._miss_count = 0

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

        try:
            query_embedding = await self._get_embedding(prompt)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
            )

            if not results["ids"] or not results["ids"][0]:
                self._miss_count += 1
                return None

            distance = results["distances"][0][0]
            similarity = 1.0 - distance

            if similarity < self.similarity_threshold:
                self._miss_count += 1
                return None

            metadata = results["metadatas"][0][0]
            self._hit_count += 1

            return CacheHit(
                prompt=metadata["prompt"],
                code=metadata["code"],
                reasoning=metadata["reasoning"],
                confidence=float(metadata["confidence"]),
                execution_result=ExecutionResult(
                    stdout=metadata["stdout"],
                    stderr=metadata["stderr"],
                    exit_code=int(metadata["exit_code"]),
                ),
                similarity=similarity,
                timestamp=metadata["timestamp"],
            )

        except Exception as e:
            logger.warning(f"Cache search failed: {e}, falling back to normal generation")
            self._miss_count += 1
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
            embedding = await self._get_embedding(prompt)

            metadata = {
                "prompt": prompt,
                "code": response.code,
                "reasoning": response.reasoning,
                "confidence": str(response.confidence),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": str(result.exit_code),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            doc_id = f"{hash(prompt)}_{hash(response.code)}"

            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[metadata],
            )

            logger.debug(f"Cached execution for prompt: {prompt[:50]}...")

        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    async def clear(self) -> None:
        """Clear all entries from the cache."""
        try:
            self.client.delete_collection(name=self.COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "Semantic cache for Cyclic code generation"}
            )
            self._hit_count = 0
            self._miss_count = 0
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise

    def get_stats(self) -> dict:
        """Get cache statistics."""
        count = self.collection.count()
        total_queries = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_queries if total_queries > 0 else 0.0

        return {
            "total_entries": count,
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": hit_rate,
            "cache_dir": str(self.cache_dir),
        }







