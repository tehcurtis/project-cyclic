"""Shared ChromaDB infrastructure used by both SemanticCache and Memory."""

import pathlib
from datetime import UTC, datetime

import chromadb
from chromadb.config import Settings
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Shared retry policy for local ChromaDB operations.
chroma_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((OSError, RuntimeError)),
)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def new_persistent_client(path: str | pathlib.Path) -> chromadb.PersistentClient:
    """Create a ChromaDB PersistentClient with telemetry disabled."""
    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )
