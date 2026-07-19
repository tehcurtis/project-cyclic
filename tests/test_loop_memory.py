"""
Tests for CyclicLoop's integration with Memory: the recall block must be used
as a transient prompt prefix (never persisted to history), and successful
runs must backfill memory.
"""

from unittest.mock import AsyncMock, patch

import pytest

from cyclic.agent import Agent, AgentResponse
from cyclic.main import CyclicLoop
from cyclic.sandbox import ExecutionResult

MARKER = "MEMORY_CONTEXT_BLOCK"


@pytest.mark.asyncio
async def test_recall_block_used_as_transient_prefix_and_not_persisted():
    memory = AsyncMock()
    memory.recall_context = AsyncMock(return_value=MARKER)
    memory.remember = AsyncMock()

    sandbox = AsyncMock()
    sandbox.run = AsyncMock(
        return_value=ExecutionResult(stdout="ok", stderr="", exit_code=0)
    )

    loop = CyclicLoop(cache=None, memory=memory, sandbox=sandbox)

    with patch.object(
        Agent,
        "generate",
        AsyncMock(
            return_value=AgentResponse(code="print('x')", reasoning="r", confidence=0.9)
        ),
    ) as mock_generate:
        original_prompt = "write a function that adds two numbers"
        await loop.run(original_prompt)

        # generate() was called with a prompt containing the recalled marker.
        call_args = mock_generate.call_args
        assert MARKER in call_args.args[0]

        # The marker must never leak into persisted conversation history.
        for message in loop.history:
            assert MARKER not in message["content"]

        # Memory was backfilled with the original (unmodified) prompt.
        memory.remember.assert_awaited_once_with(original_prompt, "print('x')", "r")
