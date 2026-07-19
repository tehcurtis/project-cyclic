import asyncio
from contextlib import contextmanager

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from loguru import logger

from .agent import Agent, AgentResponse
from .cache import SemanticCache
from .memory import Memory
from .sandbox import DockerSandbox, ExecutionResult

app = typer.Typer(help="Cyclic: Self-healing code execution agent")
cache_app = typer.Typer(help="Cache management commands")
app.add_typer(cache_app, name="cache")
memory_app = typer.Typer(help="Memory management commands")
app.add_typer(memory_app, name="memory")
console = Console()


@contextmanager
def _cli_errors():
    """Shared error-reporting wrapper for CLI management commands."""
    try:
        yield
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


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
        memory: Memory | None = None,
    ):
        self.agent = Agent(model=model)

        if sandbox:
            self.sandbox = sandbox
        else:
            if CyclicLoop._shared_sandbox is None:
                CyclicLoop._shared_sandbox = DockerSandbox()
            self.sandbox = CyclicLoop._shared_sandbox

        self.cache = cache
        self.memory = memory
        self.timeout = timeout
        self.max_retries = max_retries
        self.verbose = verbose
        self.max_history_messages = max_history_messages
        self.history: list[dict] = []

    async def run(
        self, prompt: str
    ) -> tuple[AgentResponse, ExecutionResult, int, bool]:
        """
        Execute the self-healing loop.

        Returns:
            Tuple of (final_agent_response, final_execution_result, attempt_count, served_from_cache)
        """
        if self.max_retries <= 0:
            raise ValueError("max_retries must be greater than 0")

        attempt = 0
        original_prompt = prompt
        # These will be assigned in the loop before return
        agent_response: AgentResponse
        result: ExecutionResult

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

            if cached:
                self._display_result(
                    cached.code,
                    cached.execution_result.stdout,
                    header=f"\n[bold cyan]✓ Cache hit![/bold cyan] (similarity: {cached.similarity:.2f})",
                    code_label="Cached Code",
                )

                # Reconstruct AgentResponse from cache
                agent_response = AgentResponse(
                    code=cached.code,
                    reasoning=cached.reasoning,
                    confidence=cached.confidence,
                )
                result = cached.execution_result

                self._record_success(prompt, agent_response, result)

                if self.memory:
                    await self.memory.remember(
                        original_prompt, agent_response.code, agent_response.reasoning
                    )

                return agent_response, result, 0, True

        recall_block = None
        if self.memory:
            recall_block = await self.memory.recall_context(original_prompt)
            if recall_block and self.verbose:
                console.print("[dim]Recalled similar solutions from memory[/dim]")

        while attempt < self.max_retries:
            attempt += 1

            if attempt > 1:
                console.print(
                    f"\n[yellow]Retry attempt {attempt}/{self.max_retries}[/yellow]"
                )

            # Generate
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Generating code...", total=None)
                try:
                    gen_prompt = (
                        f"{recall_block}\n\nTask: {prompt}"
                        if (recall_block and attempt == 1)
                        else prompt
                    )
                    agent_response = await self.agent.generate(
                        gen_prompt, history=self.history
                    )
                    progress.update(task, description="[green]Code generated[/green]")
                except Exception as e:
                    console.print(f"[red]Generation failed: {e}[/red]")
                    raise

            if self.verbose:
                console.print("\n[bold]Generated Code:[/bold]")
                console.print(Syntax(agent_response.code, "python", theme="monokai"))
                console.print(f"\n[dim]Reasoning: {agent_response.reasoning}[/dim]")
                console.print(f"[dim]Confidence: {agent_response.confidence:.2f}[/dim]")

            # Guard & Execute (sandbox.run handles safety scanning)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Executing code in sandbox...", total=None)
                result = await self.sandbox.run(
                    agent_response.code, timeout=self.timeout
                )
                progress.update(task, description="[green]Execution complete[/green]")

            # Evaluate
            if result.exit_code == 0:
                self._display_result(
                    agent_response.code,
                    result.stdout,
                    header="\n[bold green]✓ Success![/bold green]",
                    code_label="Generated Code",
                )

                # Store in cache if enabled (use original prompt, not retry-modified prompt)
                if self.cache:
                    await self.cache.store(original_prompt, agent_response, result)

                if self.memory:
                    await self.memory.remember(
                        original_prompt, agent_response.code, agent_response.reasoning
                    )

                self._record_success(prompt, agent_response, result)

                return agent_response, result, attempt, False

            # Failure - prepare feedback for retry
            error_msg = self._format_error(result)
            console.print(
                f"\n[bold red]✗ Execution failed (attempt {attempt}/{self.max_retries})[/bold red]"
            )

            if self.verbose:
                console.print("\n[bold]Error Details:[/bold]")
                if result.stderr:
                    console.print(
                        Panel(result.stderr.strip(), border_style="red", title="stderr")
                    )
                if result.stdout:
                    console.print(
                        Panel(
                            result.stdout.strip(), border_style="yellow", title="stdout"
                        )
                    )

            # Add failure to history for context
            self._add_to_history({"role": "user", "content": prompt})
            self._add_to_history(
                {
                    "role": "assistant",
                    "content": f"```python\n{agent_response.code}\n```",
                }
            )
            self._add_to_history(
                {
                    "role": "user",
                    "content": f"The code failed with error: {error_msg}. Please fix it.",
                }
            )

            # Update prompt for next iteration
            prompt = f"Previous attempt failed: {error_msg}. Please fix the code."

        # Max retries exceeded
        console.print(
            f"\n[bold red]✗ Failed after {self.max_retries} attempts[/bold red]"
        )

        # Ensure we have values to return if the loop ran at least once
        if "agent_response" not in locals() or "result" not in locals():
            raise RuntimeError("Loop completed without generating any code")

        return agent_response, result, attempt, False

    def _display_result(
        self, code: str, stdout: str, header: str, code_label: str
    ) -> None:
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

    def _add_to_history(self, message: dict) -> None:
        """Add message to history with sliding window limit."""
        self.history.append(message)
        # Keep only the last max_history_messages
        if len(self.history) > self.max_history_messages:
            self.history = self.history[-self.max_history_messages :]

    def _format_error(self, result: ExecutionResult) -> str:
        """Format execution error for feedback."""
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
    max_retries: int = typer.Option(
        3, "--max-retries", "-r", help="Maximum retry attempts"
    ),
    timeout: int = typer.Option(
        10, "--timeout", "-t", help="Execution timeout in seconds"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    cache: bool = typer.Option(
        True, "--cache/--no-cache", help="Enable semantic cache"
    ),
    cache_threshold: float = typer.Option(
        0.85,
        "--cache-threshold",
        help="Cache similarity threshold (0.0-1.0)",
        min=0.0,
        max=1.0,
    ),
    no_memory: bool = typer.Option(
        False, "--no-memory", help="Disable solution memory"
    ),
):
    """Run the self-healing code execution loop."""
    console.print("\n[bold cyan]Cyclic: Self-Healing Code Execution[/bold cyan]")
    console.print(f"[dim]Prompt: {prompt}[/dim]\n")

    cache_instance = None
    if cache:
        try:
            cache_instance = SemanticCache(similarity_threshold=cache_threshold)
        except Exception as e:
            logger.warning(f"Failed to initialize cache: {e}, continuing without cache")
            cache_instance = None

    memory_instance = None
    if not no_memory:
        try:
            memory_instance = Memory()
        except Exception as e:
            logger.warning(
                f"Failed to initialize memory: {e}, continuing without memory"
            )

    loop = CyclicLoop(
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        verbose=verbose,
        cache=cache_instance,
        memory=memory_instance,
    )

    try:
        agent_response, result, attempts, served_from_cache = asyncio.run(
            loop.run(prompt)
        )

        if result.exit_code == 0:
            if served_from_cache:
                console.print("\n[green]Served from cache[/green]")
            else:
                console.print(f"\n[green]Completed in {attempts} attempt(s)[/green]")
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
    with _cli_errors():
        cache = SemanticCache()
        stats = asyncio.run(cache.get_stats())
        console.print("\n[bold cyan]Cache Statistics[/bold cyan]\n")
        console.print(f"Total entries: {stats['total_entries']}")
        cache_size_mb = stats["cache_size_bytes"] / (1024 * 1024)
        console.print(f"Cache size: {cache_size_mb:.2f} MB")
        console.print(f"Cache directory: {stats['cache_dir']}")


@cache_app.command()
def clear():
    """Clear the semantic cache."""
    with _cli_errors():
        cache = SemanticCache()
        asyncio.run(cache.clear())
        console.print("[green]Cache cleared successfully[/green]")


@memory_app.command("stats")
def memory_stats():
    """Show memory statistics."""
    with _cli_errors():
        mem = Memory()
        console.print(f"Stored solutions: {asyncio.run(mem.count())}")


@memory_app.command("clear")
def memory_clear():
    """Clear all stored solutions."""
    with _cli_errors():
        mem = Memory()
        asyncio.run(mem.clear())
        console.print("Memory cleared.")


if __name__ == "__main__":
    app()
