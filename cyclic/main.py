import asyncio
from dataclasses import dataclass
from typing import Literal, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from loguru import logger

from .agent import Agent, AgentResponse
from .cache import SemanticCache
from .sandbox import DockerSandbox, ExecutionResult

app = typer.Typer(help="Cyclic: Self-healing code execution agent")
cache_app = typer.Typer(help="Cache management commands")
app.add_typer(cache_app, name="cache")
console = Console()

FailureKind = Literal[
    "execution", "timeout", "self_test", "user_check", "output_mismatch", "missing_tests"
]


@dataclass
class LoopResult:
    """Outcome of a CyclicLoop.run() invocation."""
    agent_response: AgentResponse
    result: ExecutionResult
    attempts: int
    from_cache: bool
    verified: bool
    failure: Optional[FailureKind] = None


class CyclicLoop:
    """Self-healing loop that generates, validates, executes, and retries code."""

    # Shared sandbox instance across all loops
    _shared_sandbox: DockerSandbox | None = None

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        timeout: int = 10,
        max_retries: int = 3,
        verbose: bool = False,
        max_history_messages: int = 20,
        sandbox: DockerSandbox | None = None,
        cache: SemanticCache | None = None,
        verify: bool = True,
        user_checks: list[str] | None = None,
        expected_outputs: list[str] | None = None,
    ):
        self.verify = verify
        self.agent = Agent(model=model, require_tests=verify)

        if sandbox:
            self.sandbox = sandbox
        else:
            if CyclicLoop._shared_sandbox is None:
                CyclicLoop._shared_sandbox = DockerSandbox()
            self.sandbox = CyclicLoop._shared_sandbox

        self.cache = cache
        self.timeout = timeout
        self.max_retries = max_retries
        self.verbose = verbose
        self.max_history_messages = max_history_messages
        self.history: list[dict] = []
        self.user_checks = user_checks or []
        self.expected_outputs = expected_outputs or []

    async def run(self, prompt: str) -> LoopResult:
        """
        Execute the self-healing loop.

        Returns:
            LoopResult describing the final agent response, execution result,
            attempt count, cache/verification provenance, and (on exhaustion)
            the last failure kind.
        """
        if self.max_retries <= 0:
            raise ValueError("max_retries must be greater than 0")

        attempt = 0
        original_prompt = prompt
        check_payload = "\n\n".join(self.user_checks) if self.user_checks else None

        # These will be assigned in the loop before return
        agent_response: AgentResponse
        result: ExecutionResult
        kind: FailureKind | None = None

        # Check cache before generation (only on first attempt)
        if self.cache:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Checking cache...", total=None)
                cached = await self.cache.search(prompt)
                progress.update(task, description="[green]Cache check complete[/green]")

            if cached and self.expected_outputs:
                missing_expect = [
                    e for e in self.expected_outputs if e not in cached.execution_result.stdout
                ]
                if missing_expect:
                    console.print(
                        "[yellow]Cache hit did not satisfy --expect; regenerating[/yellow]"
                    )
                    cached = None

            serve_result: ExecutionResult | None = None
            if cached:
                serve_result = cached.execution_result

                if self.user_checks:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        task = progress.add_task(
                            "Re-verifying cache hit against --check...", total=None
                        )
                        recheck = await self.sandbox.run(
                            cached.code, timeout=self.timeout, user_checks=check_payload
                        )
                        progress.update(task, description="[green]Re-verification complete[/green]")

                    expect_ok = all(
                        e in recheck.stdout for e in self.expected_outputs
                    )
                    if recheck.exit_code == 0 and expect_ok:
                        serve_result = recheck
                    else:
                        cached = None
                        serve_result = None

            if cached and serve_result is not None:
                agent_response = AgentResponse(
                    code=cached.code,
                    reasoning=cached.reasoning,
                    confidence=cached.confidence,
                    test_code=cached.test_code,
                )

                self._display_result(
                    cached.code,
                    serve_result.stdout,
                    header=f"\n[bold cyan]✓ Cache hit![/bold cyan] (similarity: {cached.similarity:.2f})",
                    code_label="Cached Code",
                )

                self._record_success(prompt, agent_response, serve_result)

                return LoopResult(
                    agent_response=agent_response,
                    result=serve_result,
                    attempts=0,
                    from_cache=True,
                    verified=True,
                )

        while attempt < self.max_retries:
            attempt += 1

            if attempt > 1:
                console.print(f"\n[yellow]Retry attempt {attempt}/{self.max_retries}[/yellow]")

            # Generate
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Generating code...", total=None)
                try:
                    agent_response = await self.agent.generate(prompt, history=self.history)
                    progress.update(task, description="[green]Code generated[/green]")
                except Exception as e:
                    console.print(f"[red]Generation failed: {e}[/red]")
                    raise

            if self.verbose:
                console.print("\n[bold]Generated Code:[/bold]")
                console.print(Syntax(agent_response.code, "python", theme="monokai"))
                console.print(f"\n[dim]Reasoning: {agent_response.reasoning}[/dim]")
                console.print(f"[dim]Confidence: {agent_response.confidence:.2f}[/dim]")

            # Missing self-tests when verification is required: consume the
            # attempt without calling the sandbox.
            if self.verify and not agent_response.test_code.strip():
                kind = "missing_tests"
                feedback = self._format_feedback(None, kind)
                console.print(
                    f"\n[bold red]✗ Missing test_code (attempt {attempt}/{self.max_retries})[/bold red]"
                )
                self._add_failure_history(prompt, agent_response, feedback)
                prompt = f"Previous attempt failed: {feedback}. Please fix the code."
                continue

            # Guard & Execute (sandbox.run handles safety scanning)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Executing code in sandbox...", total=None)
                result = await self.sandbox.run(
                    agent_response.code,
                    timeout=self.timeout,
                    test_code=agent_response.test_code if self.verify else None,
                    user_checks=check_payload if self.verify else None,
                )
                progress.update(task, description="[green]Execution complete[/green]")

            # Evaluate
            if result.exit_code == 0:
                missing_expect = [e for e in self.expected_outputs if e not in result.stdout]
                if missing_expect:
                    kind = "output_mismatch"
                    feedback = self._format_feedback(result, kind, missing=missing_expect[0])
                    console.print(
                        f"\n[bold red]✗ Output mismatch (attempt {attempt}/{self.max_retries})[/bold red]"
                    )
                    self._add_failure_history(prompt, agent_response, feedback)
                    prompt = f"Previous attempt failed: {feedback}. Please fix the code."
                    continue

                self._display_result(
                    agent_response.code,
                    result.stdout,
                    header="\n[bold green]✓ Success![/bold green]",
                    code_label="Generated Code",
                )

                # Store in cache if enabled and verified (use original prompt,
                # not retry-modified prompt). Never cache under --no-verify.
                if self.verify and self.cache:
                    await self.cache.store(original_prompt, agent_response, result)

                self._record_success(prompt, agent_response, result)

                return LoopResult(
                    agent_response=agent_response,
                    result=result,
                    attempts=attempt,
                    from_cache=False,
                    verified=self.verify,
                )

            # Failure - prepare feedback for retry
            kind = self._classify_failure(result.exit_code)
            feedback = self._format_feedback(result, kind)
            console.print(f"\n[bold red]✗ Execution failed (attempt {attempt}/{self.max_retries})[/bold red]")

            if self.verbose:
                console.print("\n[bold]Error Details:[/bold]")
                if result.stderr:
                    console.print(Panel(result.stderr.strip(), border_style="red", title="stderr"))
                if result.stdout:
                    console.print(Panel(result.stdout.strip(), border_style="yellow", title="stdout"))

            self._add_failure_history(prompt, agent_response, feedback)

            # Update prompt for next iteration
            prompt = f"Previous attempt failed: {feedback}. Please fix the code."

        # Max retries exceeded
        console.print(f"\n[bold red]✗ Failed after {self.max_retries} attempts[/bold red]")

        # Ensure we have values to return if the loop ran at least once
        if 'agent_response' not in locals() or 'result' not in locals():
            raise RuntimeError("Loop completed without generating any code")

        return LoopResult(
            agent_response=agent_response,
            result=result,
            attempts=attempt,
            from_cache=False,
            verified=False,
            failure=kind,
        )

    def _display_result(self, code: str, stdout: str, header: str, code_label: str) -> None:
        """Render a successful result (fresh or cached) to the console."""
        console.print(header)
        console.print(f"\n[bold]{code_label}:[/bold]")
        console.print(Syntax(code, "python", theme="monokai"))

        if stdout:
            console.print("\n[bold]Output:[/bold]")
            console.print(Panel(stdout.strip(), border_style="green"))

    def _record_success(
        self, prompt: str, agent_response: AgentResponse, result: ExecutionResult
    ) -> None:
        """Append a successful interaction (fresh or cached) to conversation history."""
        self._add_to_history({"role": "user", "content": prompt})
        body = f"```python\n{agent_response.code}\n```"
        if result.stdout:
            body += f"\n\nOutput: {result.stdout}"
        self._add_to_history({"role": "assistant", "content": body})

    def _assistant_turn_content(self, agent_response: AgentResponse) -> str:
        """Render the assistant's turn for failure history: code, then tests."""
        content = f"```python\n{agent_response.code}\n```"
        if self.verify:
            content += f"\n\n# tests\n```python\n{agent_response.test_code}\n```"
        return content

    def _add_failure_history(
        self, prompt: str, agent_response: AgentResponse, feedback: str
    ) -> None:
        """Add a failed interaction (prompt, code+tests, feedback) to conversation history."""
        self._add_to_history({"role": "user", "content": prompt})
        self._add_to_history(
            {"role": "assistant", "content": self._assistant_turn_content(agent_response)}
        )
        self._add_to_history({
            "role": "user",
            "content": f"The code failed with error: {feedback}. Please fix it."
        })

    def _add_to_history(self, message: dict) -> None:
        """Add message to history with sliding window limit."""
        self.history.append(message)
        # Keep only the last max_history_messages
        if len(self.history) > self.max_history_messages:
            self.history = self.history[-self.max_history_messages:]

    def _classify_failure(self, exit_code: int) -> FailureKind:
        """Classify a nonzero exit code into a failure taxonomy bucket."""
        if exit_code == 124:
            return "timeout"
        if exit_code == 3:
            return "self_test"
        if exit_code == 4:
            return "user_check"
        return "execution"

    def _format_feedback(
        self,
        result: ExecutionResult | None,
        kind: FailureKind,
        missing: str | None = None,
    ) -> str:
        """Format failure feedback for the repair loop, per failure taxonomy."""
        if kind == "missing_tests":
            return (
                "Your response did not include test_code. "
                "You must return deterministic assert-based tests."
            )

        if kind == "self_test":
            return (
                "The code ran without raising errors but FAILED its verification tests:\n"
                f"{result.stderr.strip()}\n"
                "Either fix the code so the tests pass, or fix the tests if they were wrong. "
                "Return both code and test_code."
            )

        if kind == "user_check":
            check_text = "\n\n".join(self.user_checks)
            return (
                "The code ran but failed a user-required check:\n"
                f"{result.stderr.strip()}\n"
                "The check is fixed and cannot be modified — the code must satisfy it: "
                f"{check_text}."
            )

        if kind == "output_mismatch":
            stdout_trunc = result.stdout[:1000]
            return (
                f"The code ran but its output did not contain the required text {missing!r}. "
                f"Actual output: {stdout_trunc}."
            )

        # execution / timeout: unchanged wording from the legacy error formatter.
        if result.stderr:
            return result.stderr.strip()
        elif result.exit_code == 124:
            return f"Execution timed out after {self.timeout}s"
        elif result.exit_code != 0:
            return f"Exit code: {result.exit_code}"
        return "Unknown error"


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Task description for code generation"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="LLM model to use"),
    max_retries: int = typer.Option(3, "--max-retries", "-r", help="Maximum retry attempts"),
    timeout: int = typer.Option(10, "--timeout", "-t", help="Execution timeout in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Enable semantic cache"),
    cache_threshold: float = typer.Option(
        0.85,
        "--cache-threshold",
        help="Cache similarity threshold (0.0-1.0)",
        min=0.0,
        max=1.0,
    ),
    check: list[str] = typer.Option(
        None,
        "--check",
        help="Python assert snippet run after the code in the sandbox (repeatable; all must pass)",
    ),
    expect: list[str] = typer.Option(
        None,
        "--expect",
        help="Substring that must appear in the program's stdout (repeatable)",
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Require agent self-tests and gate success on verification",
    ),
):
    """Run the self-healing code execution loop."""
    check = check or []
    expect = expect or []

    if not verify and (check or expect):
        raise typer.BadParameter("--check/--expect require verification")

    console.print("\n[bold cyan]Cyclic: Self-Healing Code Execution[/bold cyan]")
    console.print(f"[dim]Prompt: {prompt}[/dim]\n")

    cache_instance = None
    if cache:
        try:
            cache_instance = SemanticCache(similarity_threshold=cache_threshold)
        except Exception as e:
            logger.warning(f"Failed to initialize cache: {e}, continuing without cache")
            cache_instance = None

    loop = CyclicLoop(
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        verbose=verbose,
        cache=cache_instance,
        verify=verify,
        user_checks=check,
        expected_outputs=expect,
    )

    try:
        loop_result = asyncio.run(loop.run(prompt))
        result = loop_result.result
        attempts = loop_result.attempts

        success = result.exit_code == 0 and (loop_result.verified or not verify)

        if success:
            if loop_result.from_cache:
                console.print("\n[green]Served from cache[/green]")
            else:
                console.print(f"\n[green]Completed in {attempts} attempt(s)[/green]")
        else:
            if loop_result.failure is not None and loop_result.failure not in (
                "execution",
                "timeout",
            ):
                console.print(f"[bold red]Verification failed:[/bold red] {result.stderr}")
            else:
                console.print(f"[bold red]Sandbox Error:[/bold red] {result.stderr}")
            console.print(f"\n[red]Failed after {attempts} attempt(s)[/red]")
            raise typer.Exit(code=1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(code=130)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        logger.exception("Fatal error in CLI")
        raise typer.Exit(code=1)


@cache_app.command()
def stats():
    """Show cache statistics."""
    try:
        cache = SemanticCache()
        stats = asyncio.run(cache.get_stats())
        console.print("\n[bold cyan]Cache Statistics[/bold cyan]\n")
        console.print(f"Total entries: {stats['total_entries']}")
        cache_size_mb = stats['cache_size_bytes'] / (1024 * 1024)
        console.print(f"Cache size: {cache_size_mb:.2f} MB")
        console.print(f"Cache directory: {stats['cache_dir']}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@cache_app.command()
def clear():
    """Clear the semantic cache."""
    try:
        cache = SemanticCache()
        asyncio.run(cache.clear())
        console.print("[green]Cache cleared successfully[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
