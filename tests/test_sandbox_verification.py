"""
Docker integration tests for success-verification in DockerSandbox.

Mirrors the style of tests/test_jailbreak.py: real containers, real
runner.py, exercised through DockerSandbox.run().
"""

import pytest
from cyclic.sandbox import DockerSandbox


@pytest.mark.asyncio
async def test_failing_self_test_exits_3():
    """A failing self-test assert exits 3 with the failure message in stderr."""
    sandbox = DockerSandbox()

    code = "x = 2 + 2"
    test_code = "assert x == 5, 'math is broken'"

    result = await sandbox.run(code, timeout=5, test_code=test_code)

    assert result.exit_code == 3
    assert "Verification Failure (self-test)" in result.stderr
    assert "math is broken" in result.stderr


@pytest.mark.asyncio
async def test_passing_tests_and_checks_keep_stdout_clean():
    """Function-defining code + passing tests + passing user checks exit 0 with clean stdout."""
    sandbox = DockerSandbox()

    code = """
def add(a, b):
    return a + b

print(add(2, 3))
"""
    test_code = """
assert add(2, 3) == 5
assert add(-1, 1) == 0
assert add(0, 0) == 0
"""
    user_checks = "assert add(10, 5) == 15"

    result = await sandbox.run(code, timeout=5, test_code=test_code, user_checks=user_checks)

    assert result.exit_code == 0
    assert result.stdout.strip() == "5"
