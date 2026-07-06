import asyncio
import docker
import pathlib
from dataclasses import dataclass
from loguru import logger
from .safety import scan_code

EXIT_SELF_TEST_FAILED = 3   # keep in sync with cyclic/runner.py
EXIT_USER_CHECK_FAILED = 4
EXIT_TIMEOUT = 124

@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int

class DockerSandbox:
    def __init__(self, image: str = "python:3.12-slim"):
        self.client = docker.from_env()
        self.image = image
        self._runner_path = self._get_runner_path()
        self._pull_image()

    def _get_runner_path(self) -> pathlib.Path:
        """Get the absolute path to runner.py."""
        # Get the directory where this file (sandbox.py) is located
        current_file = pathlib.Path(__file__)
        runner_path = current_file.parent / "runner.py"
        return runner_path.resolve()

    def _pull_image(self):
        """Ensure the image exists locally to avoid runtime latency."""
        try:
            self.client.images.get(self.image)
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling Docker image: {self.image}...")
            self.client.images.pull(self.image)

    async def run(
        self,
        code: str,
        timeout: int = 10,
        test_code: str | None = None,
        user_checks: str | None = None,
    ) -> ExecutionResult:
        """
        Runs the provided Python code in an isolated container.
        Performs static analysis before execution to catch unsafe patterns.

        Args:
            code: The solution code to execute.
            timeout: Execution timeout in seconds.
            test_code: Optional agent self-test payload, run after ``code`` in
                the same namespace. Failures exit 3.
            user_checks: Optional user-supplied check payload, run after
                ``test_code``. Failures exit 4.
        """
        # 0. Static Safety Check - scan each payload separately for attribution.
        for payload, label in (
            (code, "generated code"),
            (test_code, "test code"),
            (user_checks, "user check"),
        ):
            if payload is None:
                continue
            report = scan_code(payload)
            if not report.is_safe:
                return ExecutionResult(
                    stdout="",
                    stderr=f"Safety Violation in {label}:\n" + "\n".join(report.issues),
                    exit_code=1
                )

        return await asyncio.to_thread(self._run_sync, code, timeout, test_code, user_checks)

    def _run_sync(
        self,
        code: str,
        timeout: int,
        test_code: str | None = None,
        user_checks: str | None = None,
    ) -> ExecutionResult:
        """
        Synchronous implementation of container execution.
        Uses runner.py with PEP 578 audit hooks for runtime security.
        """
        # Mount runner.py into container at /runner.py (read-only)
        volumes = {
            str(self._runner_path): {
                'bind': '/runner.py',
                'mode': 'ro'
            }
        }

        command = ["python", "/runner.py"]

        environment = {"CYCLIC_CODE_PAYLOAD": code}
        if test_code is not None:
            environment["CYCLIC_TEST_PAYLOAD"] = test_code
        if user_checks is not None:
            environment["CYCLIC_CHECK_PAYLOAD"] = user_checks

        container = None
        try:
            # Spin up the container with security hardening
            # Pass code via environment variable to avoid stdin race conditions
            try:
                container = self.client.containers.run(
                    self.image,
                    command=command,
                    detach=True,
                    network_disabled=True,  # Block network access
                    environment=environment,  # Pass code/tests/checks via env vars
                    # Resource limits
                    mem_limit="128m",
                    cpu_period=100000,
                    cpu_quota=100000,
                    # Security hardening
                    user="65534:65534",  # 'nobody' user
                    working_dir="/tmp",
                    read_only=True,  # Read-only filesystem
                    tmpfs={'/tmp': ''},  # Writable /tmp via tmpfs
                    volumes=volumes,  # Mount runner.py
                    cap_drop=['ALL'],  # Drop all Linux capabilities
                )
            except docker.errors.APIError as e:
                logger.error(f"Failed to create container: {e}")
                return ExecutionResult("", f"Failed to create container: {e}", 1)
            except Exception as e:
                logger.error(f"Unexpected error during container creation: {e}")
                return ExecutionResult("", f"Container creation failed: {e}", 1)

            if container is None:
                return ExecutionResult("", "Container creation returned None", 1)

            # Wait for result (with timeout)
            try:
                result = container.wait(timeout=timeout)
                exit_code = result.get("StatusCode", 1)
            except Exception:
                # Timeout or other wait error
                return ExecutionResult("", f"Execution timed out after {timeout}s", 124)

            # Capture logs
            logs = container.logs(stdout=True, stderr=True)

            # Split stdout and stderr
            # Docker combines them, but we can try to separate if needed
            # For now, treat all as stdout unless exit_code != 0
            log_output = logs.decode("utf-8", errors="replace") if logs else ""

            if exit_code == 0:
                stdout = log_output
                stderr = ""
            else:
                # On error, try to separate stderr (security violations go to stderr)
                # Since we can't perfectly separate, put everything in stderr for errors
                stdout = ""
                stderr = log_output

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

        # Test 2: Failure (Import Error)
        print("Test 2 (Expect Error):", await sandbox.run("import non_existent_module"))

        # Test 3: Infinite Loop (Timeout)
        print("Test 3 (Expect Timeout):", await sandbox.run("while True: pass", timeout=2))

        # Test 4: Safety Violation
        print("Test 4 (Expect Safety Violation):", await sandbox.run("import os; os.system('echo hacked')"))

    asyncio.run(main())