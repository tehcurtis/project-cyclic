"""
Unit tests for the AST-based static safety scanner in cyclic.safety.
These are pure, synchronous tests with no Docker or network dependency.
"""

from cyclic.safety import scan_code


def test_safe_code_is_safe():
    """Test that simple, harmless code is reported as safe."""
    report = scan_code("x = 1 + 1")

    assert report.is_safe is True
    assert report.issues == []


def test_import_os_is_blocked():
    """Test that a direct 'import os' is flagged."""
    report = scan_code("import os")

    assert report.is_safe is False
    assert any("os" in issue for issue in report.issues)


def test_import_os_submodule_is_blocked():
    """Test that importing a submodule of a blocked module is flagged."""
    report = scan_code("import os.path")

    assert report.is_safe is False
    assert any("os" in issue for issue in report.issues)


def test_from_subprocess_import_is_blocked():
    """Test that 'from subprocess import X' is flagged."""
    report = scan_code("from subprocess import Popen")

    assert report.is_safe is False
    assert any("subprocess" in issue for issue in report.issues)


def test_eval_and_exec_calls_are_blocked():
    """Test that calls to eval and exec are flagged."""
    eval_report = scan_code("eval('1+1')")
    exec_report = scan_code("exec('pass')")

    assert eval_report.is_safe is False
    assert exec_report.is_safe is False


def test_open_is_not_blocked():
    """Test that 'open' alone (no blocked import) is allowed by this scanner.

    Runtime audit hooks restrict file access separately.
    """
    report = scan_code("open('/tmp/x', 'w')")

    assert report.is_safe is True


def test_invalid_syntax_is_unsafe():
    """Test that code with a syntax error is reported as unsafe."""
    report = scan_code("def f(:")

    assert report.is_safe is False
    assert report.issues[0].startswith("SyntaxError")
