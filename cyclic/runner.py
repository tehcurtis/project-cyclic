#!/usr/bin/env python3
"""
Secure runner script for executing untrusted Python code inside Docker containers.
Uses PEP 578 audit hooks to intercept dangerous system calls at runtime.
"""

import os.path
import sys
from typing import Any


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
        'AttributeError', 'RuntimeError',
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
    except SecurityError as e:
        print(f"Security Violation: {e}", file=sys.stderr)
        sys.exit(1)
    # Let other exceptions propagate normally


if __name__ == "__main__":
    main()

