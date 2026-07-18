import datetime
import os
import pathlib
import uuid

import chromadb
from chromadb.config import Settings
from loguru import logger


class Memory:
    """Synchronous semantic memory of past successful (and failed) solutions."""

    def __init__(self, path: str | None = None) -> None:
        persist = path or os.path.expanduser("~/.cyclic/memory")
        pathlib.Path(persist).mkdir(parents=True, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(
                path=persist,
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection("solutions")
        except Exception as e:
            logger.warning(f"Failed to initialize memory: {e}, continuing without memory")
            self.collection = None

    def remember(
        self,
        prompt: str,
        code: str,
        reasoning: str,
        success: bool,
        error: str = "",
    ) -> None:
        """Store a solution attempt (successful or failed) for future recall."""
        if self.collection is None:
            return

        try:
            self.collection.add(
                documents=[prompt],
                ids=[str(uuid.uuid4())],
                metadatas=[
                    {
                        "code": code,
                        "reasoning": reasoning,
                        "success": success,
                        "error": error,
                        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                    }
                ],
            )
        except Exception as e:
            logger.warning(f"Failed to store solution in memory: {e}")

    def recall(self, prompt: str, k: int = 3) -> list[dict]:
        """Recall up to k past successful solutions similar to prompt."""
        if self.collection is None:
            return []

        try:
            count = self.collection.count()
            if count == 0:
                return []

            results = self.collection.query(
                query_texts=[prompt],
                n_results=min(k, count),
                where={"success": True},
            )

            if not results["ids"] or not results["ids"][0]:
                return []

            hits = []
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i]
                hits.append(
                    {
                        "prompt": results["documents"][0][i],
                        "code": metadata["code"],
                        "reasoning": metadata["reasoning"],
                    }
                )
            return hits[:k]
        except Exception as e:
            logger.warning(f"Memory recall failed: {e}")
            return []

    def count(self) -> int:
        """Return the number of stored solutions."""
        if self.collection is None:
            return 0
        try:
            return self.collection.count()
        except Exception as e:
            logger.warning(f"Failed to get memory count: {e}")
            return 0

    def clear(self) -> None:
        """Clear all stored solutions."""
        try:
            self.client.delete_collection("solutions")
            self.collection = self.client.get_or_create_collection("solutions")
        except Exception as e:
            logger.warning(f"Failed to clear memory: {e}")
