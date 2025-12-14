import asyncio
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
    ):
        self.agent = Agent(model=model)

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

    async def run(self, prompt: str) -> tuple[AgentResponse, ExecutionResult, int]:
        """
        Execute the self-healing loop.

        Returns:
            Tuple of (final_agent_response, final_execution_result, attempt_count)
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
                cache_hit = await self.cache.search(prompt)
                progress.update(task, description="[green]Cache check complete[/green]")

            if cache_hit:
                console.print(f"\n[bold cyan]✓ Cache hit![/bold cyan] (similarity: {cache_hit.similarity:.2f})")
                console.print("\n[bold]Cached Code:[/bold]")
                console.print(Syntax(cache_hit.code, "python", theme="monokai"))

                if cache_hit.execution_result.stdout:
                    console.print("\n[bold]Output:[/bold]")
                    console.print(Panel(cache_hit.execution_result.stdout.strip(), border_style="green"))

                # Reconstruct AgentResponse from cache
                agent_response = AgentResponse(
                    code=cache_hit.code,
                    reasoning=cache_hit.reasoning,
                    confidence=cache_hit.confidence,
                )
                result = cache_hit.execution_result

                # Add to history
                self._add_to_history({"role": "user", "content": prompt})
                self._add_to_history({
                    "role": "assistant",
                    "content": f"```python\n{agent_response.code}\n```\n\nOutput: {result.stdout}"
                })

                return agent_response, result, 1

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

            # Guard & Execute (sandbox.run handles safety scanning)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Executing code in sandbox...", total=None)
                result = await self.sandbox.run(agent_response.code, timeout=self.timeout)
                progress.update(task, description="[green]Execution complete[/green]")

            # Evaluate
            if result.exit_code == 0:
                console.print("\n[bold green]✓ Success![/bold green]")

                # Print the generated code so user can see comments/implementation
                console.print("\n[bold]Generated Code:[/bold]")
                console.print(Syntax(agent_response.code, "python", theme="monokai"))

                if result.stdout:
                    console.print("\n[bold]Output:[/bold]")
                    console.print(Panel(result.stdout.strip(), border_style="green"))

                # Store in cache if enabled (use original prompt, not retry-modified prompt)
                if self.cache:
                    await self.cache.store(original_prompt, agent_response, result)

                # Add successful interaction to history
                self._add_to_history({"role": "user", "content": prompt})
                self._add_to_history({
                    "role": "assistant",
                    "content": f"```python\n{agent_response.code}\n```\n\nOutput: {result.stdout}"
                })

                return agent_response, result, attempt

            # Failure - prepare feedback for retry
            error_msg = self._format_error(result)
            console.print(f"\n[bold red]✗ Execution failed (attempt {attempt}/{self.max_retries})[/bold red]")

            if self.verbose:
                console.print("\n[bold]Error Details:[/bold]")
                if result.stderr:
                    console.print(Panel(result.stderr.strip(), border_style="red", title="stderr"))
                if result.stdout:
                    console.print(Panel(result.stdout.strip(), border_style="yellow", title="stdout"))

            # Add failure to history for context
            self._add_to_history({"role": "user", "content": prompt})
            self._add_to_history({
                "role": "assistant",
                "content": f"```python\n{agent_response.code}\n```"
            })
            self._add_to_history({
                "role": "user",
                "content": f"The code failed with error: {error_msg}. Please fix it."
            })

            # Update prompt for next iteration
            prompt = f"Previous attempt failed: {error_msg}. Please fix the code."

        # Max retries exceeded
        console.print(f"\n[bold red]✗ Failed after {self.max_retries} attempts[/bold red]")

        # Ensure we have values to return if the loop ran at least once
        if 'agent_response' not in locals() or 'result' not in locals():
             raise RuntimeError("Loop completed without generating any code")

        return agent_response, result, attempt

    def _add_to_history(self, message: dict) -> None:
        """Add message to history with sliding window limit."""
        self.history.append(message)
        # Keep only the last max_history_messages
        if len(self.history) > self.max_history_messages:
            self.history = self.history[-self.max_history_messages:]

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
    max_retries: int = typer.Option(3, "--max-retries", "-r", help="Maximum retry attempts"),
    timeout: int = typer.Option(10, "--timeout", "-t", help="Execution timeout in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Enable semantic cache"),
    cache_threshold: float = typer.Option(0.85, "--cache-threshold", help="Cache similarity threshold (0.0-1.0)"),
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

    loop = CyclicLoop(
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        verbose=verbose,
        cache=cache_instance,
    )

    try:
        agent_response, result, attempts = asyncio.run(loop.run(prompt))

        if result.exit_code == 0:
            console.print(f"\n[green]Completed in {attempts} attempt(s)[/green]")
        else:
            console.print(f"[bold red]Sandbox Error:[/bold red] {result.stderr}")
            console.print(f"\n[red]Failed after {attempts} attempt(s)[/red]")
            raise typer.Exit(code=1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(code=130)
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

