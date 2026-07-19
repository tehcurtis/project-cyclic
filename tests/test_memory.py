"""
Tests for semantic memory functionality.

Note: chromadb's default embedding function downloads a small ONNX model on
first use, so the first test run in a fresh environment may be slow. This is
expected.
"""

import pytest

from cyclic.memory import Memory


@pytest.mark.asyncio
async def test_recall_returns_stored_solution_for_similar_prompt(tmp_path):
    memory = Memory(path=str(tmp_path / "memory"))

    await memory.remember(
        prompt="write a function that adds two numbers",
        code="def add(a, b):\n    return a + b",
        reasoning="Simple addition function",
    )

    hits = await memory.recall("create a function to add two numbers")

    assert len(hits) >= 1
    assert hits[0]["code"] == "def add(a, b):\n    return a + b"
    # store_prompt defaults to False: raw prompt is not persisted.
    assert hits[0]["prompt"] is None


@pytest.mark.asyncio
async def test_recall_returns_prompt_when_store_prompt_enabled(tmp_path):
    memory = Memory(path=str(tmp_path / "memory"), store_prompt=True)

    await memory.remember(
        prompt="write a function that adds two numbers",
        code="def add(a, b):\n    return a + b",
        reasoning="Simple addition function",
    )

    hits = await memory.recall("create a function to add two numbers")

    assert len(hits) >= 1
    assert hits[0]["code"] == "def add(a, b):\n    return a + b"
    assert hits[0]["prompt"] == "write a function that adds two numbers"


@pytest.mark.asyncio
async def test_recall_filters_unrelated_prompt_by_threshold(tmp_path):
    memory = Memory(path=str(tmp_path / "memory"))

    await memory.remember(
        prompt="write a function that adds two numbers",
        code="def add(a, b):\n    return a + b",
        reasoning="Simple addition function",
    )

    hits = await memory.recall("open a websocket connection and stream stock prices")

    assert hits == []


@pytest.mark.asyncio
async def test_recall_on_empty_store_returns_empty_list(tmp_path):
    memory = Memory(path=str(tmp_path / "memory"))

    hits = await memory.recall("write a function that adds two numbers")

    assert hits == []


@pytest.mark.asyncio
async def test_remember_same_prompt_and_code_dedupes(tmp_path):
    memory = Memory(path=str(tmp_path / "memory"))

    await memory.remember(
        prompt="write a function that adds two numbers",
        code="def add(a, b):\n    return a + b",
        reasoning="Simple addition function",
    )
    await memory.remember(
        prompt="write a function that adds two numbers",
        code="def add(a, b):\n    return a + b",
        reasoning="Simple addition function, updated reasoning",
    )

    assert await memory.count() == 1


@pytest.mark.asyncio
async def test_clear_removes_all_stored_solutions(tmp_path):
    memory = Memory(path=str(tmp_path / "memory"))

    await memory.remember(
        prompt="write a function that adds two numbers",
        code="def add(a, b):\n    return a + b",
        reasoning="Simple addition function",
    )
    assert await memory.count() == 1

    await memory.clear()

    assert await memory.count() == 0


def test_init_failure_propagates(tmp_path):
    """A broken store must raise, not masquerade as an empty one."""
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")

    with pytest.raises(OSError):
        Memory(path=str(blocked))
