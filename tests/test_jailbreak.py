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


@pytest.mark.asyncio
async def test_standard_library_imports():
    """Test that standard library imports work correctly."""
    sandbox = DockerSandbox()

    code = """
import heapq
import collections
import math

# Test heapq
heap = [3, 1, 4, 1, 5]
heapq.heapify(heap)
print(f"Heap: {heap}")

# Test collections
counter = collections.Counter([1, 1, 1, 2, 2, 3])
print(f"Counter: {dict(counter)}")

# Test math
print(f"Pi: {math.pi:.2f}")
"""

    result = await sandbox.run(code, timeout=5)

    assert result.exit_code == 0
    assert "Heap:" in result.stdout
    assert "Counter:" in result.stdout
    assert "Pi:" in result.stdout


@pytest.mark.asyncio
async def test_top_k_frequent_elements():
    """Test the Top K Frequent Elements algorithm that was failing."""
    sandbox = DockerSandbox()

    code = """
import heapq
from collections import Counter

def top_k_frequent(nums, k):
    # Count frequencies using Counter (HashMap)
    count = Counter(nums)

    # Use min-heap to get top k elements
    # Time complexity: O(N log k) where N is number of elements
    heap = []
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)

    # Extract elements from heap
    result = [num for freq, num in heap]
    return result

# Test case from the original prompt
nums = [1, 1, 1, 2, 2, 3]
k = 2
result = top_k_frequent(nums, k)
print(f"Result: {sorted(result)}")
assert sorted(result) == [1, 2], f"Expected [1, 2], got {result}"
"""

    result = await sandbox.run(code, timeout=5)

    assert result.exit_code == 0
    assert "Result: [1, 2]" in result.stdout


@pytest.mark.asyncio
async def test_class_based_code_executes():
    """Regression test for the missing __build_class__ builtin: class
    definitions (e.g. a LeetCode-295-style MedianFinder) must execute
    successfully end to end in the container."""
    sandbox = DockerSandbox()

    code = """
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # max-heap (negated)
        self.large = []  # min-heap

    def addNum(self, num):
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2

mf = MedianFinder()
mf.addNum(1)
mf.addNum(2)
print(f"Median: {mf.findMedian()}")
mf.addNum(3)
print(f"Median: {mf.findMedian()}")
"""

    result = await sandbox.run(code, timeout=5)

    assert result.exit_code == 0
    assert "Median: 1.5" in result.stdout
    assert "Median: 2" in result.stdout


@pytest.mark.asyncio
async def test_os_fork_blocked_at_runtime():
    """Test that os.fork is blocked by the audit hook via a route the AST scanner misses."""
    sandbox = DockerSandbox()

    # Aliasing __import__ evades both the import ban and the bare-name call
    # check in the static scanner, so only the runtime audit hook can stop this.
    code = """
imp = __import__
os_mod = imp('os')
os_mod.fork()
"""

    result = await sandbox.run(code, timeout=5)

    assert result.exit_code != 0
    combined = result.stderr + result.stdout
    assert "Security Violation" in combined
    assert "Process execution blocked (os.fork)" in combined
