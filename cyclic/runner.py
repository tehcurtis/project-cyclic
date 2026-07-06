#!/usr/bin/env python3
"""
Secure runner script for executing untrusted Python code inside Docker containers.
Uses PEP 578 audit hooks to intercept dangerous system calls at runtime.
"""

import io
import os.path
import sys
import traceback
from typing import Any

# Absolute path to this script itself. traceback.format_exc() (used for
# verification-failure diagnostics) needs to read source lines from frames
# that belong to this trusted, read-only file, so it must stay readable even
# though it lives outside /tmp and outside the stdlib prefixes.
_SELF_PATH = os.path.normpath(os.path.abspath(__file__))


def cyclic_guard(event: str, args: tuple[Any, ...]) -> None:
    """
    Audit hook that intercepts dangerous system calls.
    Raises SecurityError for blocked operations.
    """
    # Block all networking
    if event.startswith("socket."):
        raise SecurityError(f"Security Violation: Network access blocked ({event})")

    # Block process creation
    if event in ("os.system", "os.spawn", "os.spawnve", "os.spawnv", "os.spawnvp", "os.spawnvpe"):
        raise SecurityError(f"Security Violation: Process execution blocked ({event})")

    if event == "subprocess.Popen":
        raise SecurityError(f"Security Violation: Subprocess creation blocked ({event})")

    # Block dynamic library loading
    if event == "ctypes.dlopen":
        raise SecurityError(f"Security Violation: Dynamic library loading blocked ({event})")

    # Restrict file operations
    if event == "open":
        if not args:
            return

        path = args[0]

        # Allow file descriptors (integers)
        if isinstance(path, int):
            return

        # Normalize path to prevent traversal attacks (e.g., /tmp/../etc/passwd)
        if isinstance(path, str):
            normalized = os.path.normpath(path)
            # Allow /tmp directory access (but not /tmp/../etc/passwd)
            if normalized.startswith("/tmp/") or normalized == "/tmp":
                return

            # Allow reading this script's own source (needed by
            # traceback.format_exc() to render source-line context for
            # frames inside this file, e.g. verification failures).
            if normalized == _SELF_PATH:
                return

            # Allow reading files from standard library locations
            # This is required for imports to work (reading .pyc files, etc.)
            # We check if the path starts with sys.base_prefix or sys.base_exec_prefix
            allowed_prefixes = [sys.base_prefix, sys.base_exec_prefix]

            # Also include site-packages if it's not covered (usually it is)
            # But be careful not to expose everything if prefix is /

            for prefix in allowed_prefixes:
                if normalized.startswith(prefix):
                    # Ensure we don't allow reading sensitive files that might happen to be there
                    # For now, standard library paths are generally safe to read
                    return

        # Block all other file access
        raise SecurityError(f"Security Violation: File access blocked ({path})")


class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass


def create_safe_globals() -> dict[str, Any]:
    """
    Create a restricted globals dictionary with only safe builtins.
    """
    safe_builtins = {
        # Basic types
        'int', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set',
        # Import functionality
        '__import__',
        # Basic functions
        'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
        'sorted', 'reversed', 'min', 'max', 'sum', 'abs', 'round',
        # Type conversions
        'chr', 'ord', 'hex', 'oct', 'bin',
        # File operations (restricted by runtime audit hook)
        'open',
        # System exit
        'exit', 'quit',
        # Exceptions
        'Exception', 'ValueError', 'TypeError', 'IndexError', 'KeyError',
        'AttributeError', 'RuntimeError', 'AssertionError',
        # Type checks (needed for assert-based tests)
        'isinstance',
    }

    # Import only safe builtins
    import builtins
    restricted_builtins = {
        name: getattr(builtins, name)
        for name in safe_builtins
        if hasattr(builtins, name)
    }

    return {
        '__builtins__': restricted_builtins,
        '__name__': '__main__',
    }


def _run_verification(payload: str, globals_: dict[str, Any], label: str, exit_code: int) -> None:
    """
    Execute a verification payload (self-tests or user checks) against the
    solution's namespace. On ANY failure (including SystemExit, even
    SystemExit(0)) print diagnostics to the real stderr and exit with
    ``exit_code``. On success, discard captured output and return.
    """
    real_stdout, real_stderr = sys.stdout, sys.stderr
    captured = io.StringIO()
    try:
        sys.stdout = captured
        sys.stderr = captured
        exec(payload, globals_)
    except BaseException as e:
        sys.stdout, sys.stderr = real_stdout, real_stderr
        print(f"Verification Failure ({label}): {type(e).__name__}: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        captured_output = captured.getvalue()
        if captured_output:
            print(f"Captured output ({label}):\n{captured_output}", file=sys.stderr)
        sys.exit(exit_code)
    else:
        sys.stdout, sys.stderr = real_stdout, real_stderr
        # Discard captured buffer on success - container output stays purely
        # the solution's stdout.


def main() -> None:
    """Main entry point for the secure runner."""
    # Register audit hook before any code execution
    sys.addaudithook(cyclic_guard)

    # Read code from environment variable (more reliable) or stdin (fallback)
    code = os.environ.get("CYCLIC_CODE_PAYLOAD", "")
    if not code:
        code = sys.stdin.read()

    if not code:
        print("Error: No code provided", file=sys.stderr)
        sys.exit(1)

    # Create safe execution environment
    safe_globals = create_safe_globals()

    # Execute user code
    try:
        exec(code, safe_globals)
    except SystemExit as e:
        # Generated code calling exit(0)/quit() should not fake success by
        # skipping verification; only a clean exit continues to verification.
        if e.code in (None, 0):
            pass
        else:
            sys.exit(e.code if isinstance(e.code, int) else 1)
    except SecurityError as e:
        print(f"Security Violation: {e}", file=sys.stderr)
        sys.exit(1)
    # Let other exceptions propagate normally (exit 1)

    # Verification payloads: self-tests then user checks, in the same
    # namespace as the solution so they can call its functions/variables.
    for env_var, exit_code, label in [
        ("CYCLIC_TEST_PAYLOAD", 3, "self-test"),
        ("CYCLIC_CHECK_PAYLOAD", 4, "user check"),
    ]:
        payload = os.environ.get(env_var, "")
        if payload:
            _run_verification(payload, safe_globals, label, exit_code)


if __name__ == "__main__":
    main()
