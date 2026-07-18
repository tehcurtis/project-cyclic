"""
Tests for semantic memory functionality.

Note: chromadb's default embedding function downloads a small ONNX model on
first use, so the first test run in a fresh environment may be slow. This is
expected.
"""

from cyclic.memory import Memory


def test_recall_returns_stored_solution_for_similar_prompt(tmp_path):
    memory = Memory(path=str(tmp_path / "memory"))

    memory.remember(
        prompt="write a function that adds two numbers",
        code="def add(a, b):\n    return a + b",
        reasoning="Simple addition function",
        success=True,
    )

    hits = memory.recall("create a function to add two numbers")

    assert len(hits) >= 1
    assert hits[0]["code"] == "def add(a, b):\n    return a + b"


def test_recall_does_not_return_failed_solutions(tmp_path):
    memory = Memory(path=str(tmp_path / "memory"))

    memory.remember(
        prompt="write a function that adds two numbers",
        code="def add(a, b):\n    return a - b",
        reasoning="Buggy subtraction instead of addition",
        success=False,
        error="Assertion failed: add(2, 2) != 4",
    )

    hits = memory.recall("create a function to add two numbers")

    assert hits == []


def test_recall_on_empty_store_returns_empty_list(tmp_path):
    memory = Memory(path=str(tmp_path / "memory"))

    hits = memory.recall("write a function that adds two numbers")

    assert hits == []


def test_clear_removes_all_stored_solutions(tmp_path):
    memory = Memory(path=str(tmp_path / "memory"))

    memory.remember(
        prompt="write a function that adds two numbers",
        code="def add(a, b):\n    return a + b",
        reasoning="Simple addition function",
        success=True,
    )
    assert memory.count() == 1

    memory.clear()

    assert memory.count() == 0
