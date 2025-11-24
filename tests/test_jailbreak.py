"""
Security tests to verify that the runtime audit hooks prevent jailbreak attempts.
Tests various obfuscation techniques that bypass AST-based static analysis.
"""

import pytest
from cyclic.sandbox import DockerSandbox


@pytest.mark.asyncio
async def test_obfuscated_os_system():
    """Test that os.system calls are blocked (AST catches import, but runtime would catch call)."""
    sandbox = DockerSandbox()

    # Import is blocked by AST scanner (defense-in-depth)
    code = """
import os
func = getattr(os, 'sys' + 'tem')
func('echo hacked')
"""

    result = await sandbox.run(code, timeout=5)

    # AST scanner catches the import first (defense-in-depth)
    assert result.exit_code != 0
    assert "Safety Violation" in result.stderr or "Safety Violation" in result.stdout
    assert "os" in result.stderr or "os" in result.stdout


@pytest.mark.asyncio
async def test_file_read_blocked():
    """Test that reading sensitive files is blocked by runtime audit hook."""
    sandbox = DockerSandbox()

    code = """
f = open('/etc/passwd', 'r')
print(f.read())
"""

    result = await sandbox.run(code, timeout=5)

    # Runtime audit hook should catch this
    assert result.exit_code != 0
    assert "Security Violation" in result.stderr or "Security Violation" in result.stdout
    assert "File access blocked" in result.stderr or "File access blocked" in result.stdout


@pytest.mark.asyncio
async def test_network_access_blocked():
    """Test that network socket creation is blocked (AST catches import)."""
    sandbox = DockerSandbox()

    code = """
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('example.com', 80))
"""

    result = await sandbox.run(code, timeout=5)

    # AST scanner catches the import first (defense-in-depth)
    assert result.exit_code != 0
    assert "Safety Violation" in result.stderr or "Safety Violation" in result.stdout
    assert "socket" in result.stderr or "socket" in result.stdout


@pytest.mark.asyncio
async def test_subprocess_blocked():
    """Test that subprocess creation is blocked (AST catches import)."""
    sandbox = DockerSandbox()

    code = """
import subprocess
subprocess.Popen(['echo', 'hacked'])
"""

    result = await sandbox.run(code, timeout=5)

    # AST scanner catches the import first (defense-in-depth)
    assert result.exit_code != 0
    assert "Safety Violation" in result.stderr or "Safety Violation" in result.stdout
    assert "subprocess" in result.stderr or "subprocess" in result.stdout


@pytest.mark.asyncio
async def test_tmp_access_allowed():
    """Test that /tmp directory access is allowed."""
    sandbox = DockerSandbox()

    code = """
f = open('/tmp/test.txt', 'w')
f.write('hello')
f.close()
f = open('/tmp/test.txt', 'r')
content = f.read()
f.close()
print(content)
"""

    result = await sandbox.run(code, timeout=5)

    # Should succeed
    assert result.exit_code == 0
    assert "hello" in result.stdout


@pytest.mark.asyncio
async def test_safe_code_executes():
    """Test that safe code executes successfully."""
    sandbox = DockerSandbox()

    code = """
result = sum(range(10))
print(f"Sum: {result}")
"""

    result = await sandbox.run(code, timeout=5)

    assert result.exit_code == 0
    assert "Sum: 45" in result.stdout

