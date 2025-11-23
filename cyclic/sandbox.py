import asyncio
import docker
from dataclasses import dataclass
from loguru import logger

@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int

class DockerSandbox:
    def __init__(self, image: str = "python:3.14-slim"):
        self.client = docker.from_env()
        self.image = image
        self._pull_image()

    def _pull_image(self):
        """Ensure the image exists locally to avoid runtime latency."""
        try:
            self.client.images.get(self.image)
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling Docker image: {self.image}...")
            self.client.images.pull(self.image)

    async def run(self, code: str, timeout: int = 10) -> ExecutionResult:
        """
        Runs the provided Python code in an isolated container.
        """
        return await asyncio.to_thread(self._run_sync, code, timeout)

    def _run_sync(self, code: str, timeout: int) -> ExecutionResult:
        """
        Synchronous implementation of container execution.
        """
        # 1. Prepare the container config
        # Run 'python -c code' to keep it simple for now.

        command = ["python", "-c", code]

        container = None
        try:
            # 2. Spin up the container
            # 'network_disabled=True' prevents the AI from making web requests (Safety)
            container = self.client.containers.run(
                self.image,
                command=command,
                detach=True,
                network_disabled=True,
                # Stop the container if it exceeds memory limits (128MB)
                mem_limit="128m",
                cpu_period=100000,
                cpu_quota=100000,
                # Security Hardening
                user="65534:65534",  # 'nobody' user
                working_dir="/tmp",
                read_only=True,
                tmpfs={'/tmp': ''},
            )

            # 3. Wait for result (with timeout)
            try:
                result = container.wait(timeout=timeout)
                exit_code = result.get("StatusCode", 1)
            except Exception:
                container.kill()
                return ExecutionResult("", f"Execution timed out after {timeout}s", 124)

            # 4. Capture logs
            # demux=True returns a tuple (stdout, stderr)
            stdout_bytes, stderr_bytes = container.logs(stdout=True, stderr=True, demux=True)

            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code
            )

        except docker.errors.ContainerError as e:
            return ExecutionResult("", str(e), 1)
        except Exception as e:
            logger.error(f"Sandbox Critical Failure: {e}")
            return ExecutionResult("", str(e), 255)
        finally:
            # Ensure cleanup happens even if python crashes
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

if __name__ == "__main__":
    # Quick Test Loop
    async def main():
        sandbox = DockerSandbox()

        # Test 1: Success
        print("Test 1 (Expect Success):", await sandbox.run("print('Hello from the Jail!')"))

        # Test 2: Failure
        print("Test 2 (Expect Error):", await sandbox.run("import non_existent_module"))

        # Test 3: Infinite Loop (Timeout)
        print("Test 3 (Expect Timeout):", await sandbox.run("while True: pass", timeout=2))

    asyncio.run(main())