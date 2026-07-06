"""
Tests for cyclic/runner.py's exit-code verification contract.

Runs runner.py directly via subprocess (no Docker) with the same env vars
DockerSandbox would set, to exercise the exit-code semantics in isolation.
"""

import os
import pathlib
import subprocess
import sys

RUNNER_PATH = pathlib.Path(__file__).resolve().parent.parent / "cyclic" / "runner.py"


def run_runner(
    code: str,
    test_code: str | None = None,
    check_code: str | None = None,
    timeout: int = 10,
) -> subprocess.CompletedProcess:
    """Invoke runner.py as a subprocess with the given payloads as env vars."""
    env = os.environ.copy()
    env["CYCLIC_CODE_PAYLOAD"] = code
    env.pop("CYCLIC_TEST_PAYLOAD", None)
    env.pop("CYCLIC_CHECK_PAYLOAD", None)
    if test_code is not None:
        env["CYCLIC_TEST_PAYLOAD"] = test_code
    if check_code is not None:
        env["CYCLIC_CHECK_PAYLOAD"] = check_code

    return subprocess.run(
        [sys.executable, str(RUNNER_PATH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_passing_code_and_tests_suppresses_test_output():
    """Passing code + tests exit 0; test_code's own prints never reach stdout."""
    proc = run_runner(
        code="print('hello')",
        test_code="print('should not appear')\nassert True",
    )

    assert proc.returncode == 0
    assert proc.stdout.strip() == "hello"
    assert "should not appear" not in proc.stdout


def test_failing_self_test_exits_3_with_diagnostics():
    """A failing self-test assert exits 3 with 'Verification Failure (self-test)' + traceback."""
    proc = run_runner(
        code="x = 1",
        test_code="assert x == 2, 'boom'",
    )

    assert proc.returncode == 3
    assert "Verification Failure (self-test)" in proc.stderr
    assert "Traceback" in proc.stderr


def test_failing_user_check_exits_4():
    """A failing user --check assert exits 4 with 'Verification Failure (user check)'."""
    proc = run_runner(
        code="x = 1",
        check_code="assert x == 2, 'user check boom'",
    )

    assert proc.returncode == 4
    assert "Verification Failure (user check)" in proc.stderr


def test_exit_zero_in_code_with_failing_tests_still_fails():
    """Regression: exit(0) in the solution code must not bypass self-test verification."""
    proc = run_runner(
        code="exit(0)",
        test_code="assert False, 'tests should still run'",
    )

    assert proc.returncode == 3
    assert "Verification Failure (self-test)" in proc.stderr


def test_exit_zero_inside_test_payload_is_a_failure():
    """Regression: exit(0) inside test_code itself must not fake success."""
    proc = run_runner(
        code="x = 1",
        test_code="exit(0)",
    )

    assert proc.returncode == 3
    assert "Verification Failure (self-test)" in proc.stderr


def test_nonzero_exit_in_code_propagates_unchanged():
    """A nonzero exit() call in the solution code exits with that code (unchanged behavior)."""
    proc = run_runner(code="exit(7)")

    assert proc.returncode == 7


def test_raising_exception_in_code_exits_1():
    """An uncaught exception in the solution code exits 1 (unchanged behavior)."""
    proc = run_runner(code="raise ValueError('boom')")

    assert proc.returncode == 1
