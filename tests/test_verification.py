"""
Unit tests for the success-verification repair loop in CyclicLoop.

These tests use stub Agent/Sandbox implementations and a mocked cache so
they run without Docker or a live LLM.
"""

from unittest.mock import AsyncMock

import pytest

from cyclic.agent import AgentResponse
from cyclic.cache import CacheHit
from cyclic.main import CyclicLoop
from cyclic.sandbox import ExecutionResult


class StubAgent:
    """Agent stand-in that returns a scripted sequence of AgentResponses."""

    def __init__(self, responses: list[AgentResponse]):
        self.responses = list(responses)
        self.calls: list[tuple[str, list[dict] | None]] = []

    async def generate(self, prompt: str, history: list[dict] | None = None) -> AgentResponse:
        self.calls.append((prompt, history))
        if not self.responses:
            raise AssertionError("StubAgent exhausted its scripted responses")
        return self.responses.pop(0)


class StubSandbox:
    """Sandbox stand-in that returns a scripted sequence of ExecutionResults."""

    def __init__(self, results: list[ExecutionResult]):
        self.results = list(results)
        self.calls: list[dict] = []

    async def run(
        self,
        code: str,
        timeout: int = 10,
        test_code: str | None = None,
        user_checks: str | None = None,
    ) -> ExecutionResult:
        self.calls.append(
            {"code": code, "timeout": timeout, "test_code": test_code, "user_checks": user_checks}
        )
        if not self.results:
            raise AssertionError("StubSandbox exhausted its scripted results")
        return self.results.pop(0)


def make_cache(search_result=None) -> AsyncMock:
    """Build an AsyncMock standing in for SemanticCache."""
    cache = AsyncMock()
    cache.search = AsyncMock(return_value=search_result)
    cache.store = AsyncMock(return_value=None)
    return cache


@pytest.mark.asyncio
async def test_verified_success_first_attempt():
    """A verified success on the first attempt stores exactly once."""
    agent = StubAgent(
        [AgentResponse(code="print(1)", reasoning="r", confidence=0.9, test_code="assert True")]
    )
    sandbox = StubSandbox([ExecutionResult(stdout="1\n", stderr="", exit_code=0)])
    cache = make_cache(search_result=None)

    loop = CyclicLoop(sandbox=sandbox, cache=cache, verify=True, max_retries=3)
    loop.agent = agent

    result = await loop.run("print one")

    assert result.verified is True
    assert result.attempts == 1
    assert result.from_cache is False
    cache.store.assert_awaited_once()
    stored_response = cache.store.await_args.args[1]
    assert stored_response is result.agent_response


@pytest.mark.asyncio
async def test_self_test_failure_then_success():
    """Self-test failure (exit 3) followed by success retries once and stores once."""
    agent = StubAgent(
        [
            AgentResponse(code="print(1)", reasoning="r1", confidence=0.9, test_code="assert False"),
            AgentResponse(code="print(2)", reasoning="r2", confidence=0.9, test_code="assert True"),
        ]
    )
    sandbox = StubSandbox(
        [
            ExecutionResult(
                stdout="",
                stderr="Verification Failure (self-test): AssertionError: ",
                exit_code=3,
            ),
            ExecutionResult(stdout="2\n", stderr="", exit_code=0),
        ]
    )
    cache = make_cache(search_result=None)

    loop = CyclicLoop(sandbox=sandbox, cache=cache, verify=True, max_retries=3)
    loop.agent = agent

    result = await loop.run("print a number")

    assert result.attempts == 2
    assert result.verified is True
    history_blob = "\n".join(str(m["content"]) for m in loop.history)
    assert "FAILED its verification tests" in history_blob
    assert "assert False" in history_blob
    cache.store.assert_awaited_once()


@pytest.mark.asyncio
async def test_user_check_failure_feedback():
    """User-check failure (exit 4) surfaces 'user-required check' and doesn't store until pass."""
    agent = StubAgent(
        [
            AgentResponse(code="print(1)", reasoning="r1", confidence=0.9, test_code="assert True"),
            AgentResponse(code="print(2)", reasoning="r2", confidence=0.9, test_code="assert True"),
        ]
    )
    sandbox = StubSandbox(
        [
            ExecutionResult(
                stdout="",
                stderr="Verification Failure (user check): AssertionError: ",
                exit_code=4,
            ),
            ExecutionResult(stdout="2\n", stderr="", exit_code=0),
        ]
    )
    cache = make_cache(search_result=None)

    loop = CyclicLoop(
        sandbox=sandbox, cache=cache, verify=True, max_retries=3, user_checks=["assert True"]
    )
    loop.agent = agent

    result = await loop.run("print a number")

    assert result.attempts == 2
    history_blob = "\n".join(str(m["content"]) for m in loop.history)
    assert "user-required check" in history_blob
    cache.store.assert_awaited_once()


@pytest.mark.asyncio
async def test_expect_mismatch_retries_without_storing():
    """--expect mismatch on exit 0 triggers an output_mismatch retry; no store on mismatch."""
    agent = StubAgent(
        [
            AgentResponse(code="print('hi')", reasoning="r1", confidence=0.9, test_code="assert True"),
            AgentResponse(code="print('needle')", reasoning="r2", confidence=0.9, test_code="assert True"),
        ]
    )
    sandbox = StubSandbox(
        [
            ExecutionResult(stdout="hi\n", stderr="", exit_code=0),
            ExecutionResult(stdout="needle\n", stderr="", exit_code=0),
        ]
    )
    cache = make_cache(search_result=None)

    loop = CyclicLoop(
        sandbox=sandbox, cache=cache, verify=True, max_retries=3, expected_outputs=["needle"]
    )
    loop.agent = agent

    result = await loop.run("print needle")

    assert result.attempts == 2
    assert result.verified is True
    # Only the final, matching attempt is stored.
    cache.store.assert_awaited_once()
    stored_result = cache.store.await_args.args[2]
    assert "needle" in stored_result.stdout


@pytest.mark.asyncio
async def test_missing_test_code_consumes_attempt_without_sandbox_call():
    """Empty test_code consumes an attempt and never reaches the sandbox."""
    agent = StubAgent(
        [
            AgentResponse(code="print(1)", reasoning="r1", confidence=0.9, test_code=""),
            AgentResponse(code="print(2)", reasoning="r2", confidence=0.9, test_code="assert True"),
        ]
    )
    sandbox = StubSandbox([ExecutionResult(stdout="2\n", stderr="", exit_code=0)])
    cache = make_cache(search_result=None)

    loop = CyclicLoop(sandbox=sandbox, cache=cache, verify=True, max_retries=3)
    loop.agent = agent

    result = await loop.run("print a number")

    assert result.attempts == 2
    assert len(sandbox.calls) == 1  # sandbox skipped on the missing-tests attempt
    history_blob = "\n".join(str(m["content"]) for m in loop.history)
    assert "test_code" in history_blob


@pytest.mark.asyncio
async def test_missing_test_code_every_attempt_returns_clean_failure():
    """If the agent never produces test_code, run() returns cleanly with failure='missing_tests'."""
    agent = StubAgent(
        [
            AgentResponse(code="print(1)", reasoning="r1", confidence=0.9, test_code=""),
            AgentResponse(code="print(2)", reasoning="r2", confidence=0.9, test_code="  "),
        ]
    )
    sandbox = StubSandbox([])
    cache = make_cache(search_result=None)

    loop = CyclicLoop(sandbox=sandbox, cache=cache, verify=True, max_retries=2)
    loop.agent = agent

    result = await loop.run("do something")

    assert result.failure == "missing_tests"
    assert result.verified is False
    assert result.from_cache is False
    assert result.attempts == 2
    assert result.result.exit_code != 0
    assert len(sandbox.calls) == 0
    cache.store.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_verify_disables_verification_and_caching():
    """verify=False: sandbox gets test_code=None, success is verified=False, and cache never stores."""
    agent = StubAgent(
        [AgentResponse(code="print(1)", reasoning="r", confidence=0.9)]
    )
    sandbox = StubSandbox([ExecutionResult(stdout="1\n", stderr="", exit_code=0)])
    cache = make_cache(search_result=None)

    loop = CyclicLoop(sandbox=sandbox, cache=cache, verify=False, max_retries=3)
    loop.agent = agent

    result = await loop.run("print one")

    assert sandbox.calls[0]["test_code"] is None
    assert sandbox.calls[0]["user_checks"] is None
    assert result.verified is False
    cache.store.assert_not_awaited()


@pytest.mark.asyncio
async def test_exhaustion_all_self_test_failures():
    """Exhausting all attempts on self-test failures reports failure='self_test' and never stores."""
    agent = StubAgent(
        [
            AgentResponse(code="print(1)", reasoning="r1", confidence=0.9, test_code="assert False"),
            AgentResponse(code="print(2)", reasoning="r2", confidence=0.9, test_code="assert False"),
        ]
    )
    sandbox = StubSandbox(
        [
            ExecutionResult(stdout="", stderr="Verification Failure (self-test): x", exit_code=3),
            ExecutionResult(stdout="", stderr="Verification Failure (self-test): x", exit_code=3),
        ]
    )
    cache = make_cache(search_result=None)

    loop = CyclicLoop(sandbox=sandbox, cache=cache, verify=True, max_retries=2)
    loop.agent = agent

    result = await loop.run("do something")

    assert result.failure == "self_test"
    assert result.verified is False
    cache.store.assert_not_awaited()


@pytest.mark.asyncio
async def test_cache_hit_failing_expect_falls_through_to_generation():
    """A cache hit that fails --expect falls through to fresh generation."""
    cached_hit = CacheHit(
        prompt=None,
        code="print('no match')",
        reasoning="cached reasoning",
        confidence=0.9,
        execution_result=ExecutionResult(stdout="no match here\n", stderr="", exit_code=0),
        similarity=0.95,
        timestamp="2026-01-01T00:00:00+00:00",
        test_code="assert True",
    )
    agent = StubAgent(
        [AgentResponse(code="print('needle')", reasoning="r", confidence=0.9, test_code="assert True")]
    )
    sandbox = StubSandbox([ExecutionResult(stdout="needle\n", stderr="", exit_code=0)])
    cache = make_cache(search_result=cached_hit)

    loop = CyclicLoop(
        sandbox=sandbox, cache=cache, verify=True, max_retries=3, expected_outputs=["needle"]
    )
    loop.agent = agent

    result = await loop.run("print needle")

    assert result.from_cache is False
    assert len(agent.calls) == 1
    assert "needle" in result.result.stdout


@pytest.mark.asyncio
async def test_cache_hit_with_check_reverifies_without_self_tests():
    """A cache hit with --check re-runs only the check payload in the sandbox; agent is never invoked."""
    cached_hit = CacheHit(
        prompt=None,
        code="print('x')",
        reasoning="cached reasoning",
        confidence=0.9,
        execution_result=ExecutionResult(stdout="x\n", stderr="", exit_code=0),
        similarity=0.95,
        timestamp="2026-01-01T00:00:00+00:00",
        test_code="assert True",
    )
    agent = StubAgent([])
    sandbox = StubSandbox([ExecutionResult(stdout="x\n", stderr="", exit_code=0)])
    cache = make_cache(search_result=cached_hit)

    loop = CyclicLoop(
        sandbox=sandbox, cache=cache, verify=True, max_retries=3, user_checks=["assert True"]
    )
    loop.agent = agent

    result = await loop.run("print x")

    assert len(sandbox.calls) == 1
    assert sandbox.calls[0]["user_checks"] == "assert True"
    assert sandbox.calls[0]["test_code"] is None
    assert result.from_cache is True
    assert len(agent.calls) == 0


@pytest.mark.asyncio
async def test_unsafe_test_code_attributed_and_classified_as_execution():
    """A safety violation attributed to the test payload is classified as an 'execution' failure."""
    agent = StubAgent(
        [
            AgentResponse(
                code="print(1)", reasoning="r1", confidence=0.9, test_code="import os\nassert True"
            ),
            AgentResponse(code="print(2)", reasoning="r2", confidence=0.9, test_code="assert True"),
        ]
    )
    sandbox = StubSandbox(
        [
            ExecutionResult(
                stdout="",
                stderr="Safety Violation in test code:\nImporting 'os' is not allowed.",
                exit_code=1,
            ),
            ExecutionResult(stdout="2\n", stderr="", exit_code=0),
        ]
    )
    cache = make_cache(search_result=None)

    loop = CyclicLoop(sandbox=sandbox, cache=cache, verify=True, max_retries=3)
    loop.agent = agent

    result = await loop.run("do something")

    assert result.attempts == 2
    history_blob = "\n".join(str(m["content"]) for m in loop.history)
    assert "Safety Violation in test code" in history_blob
