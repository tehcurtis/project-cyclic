# Cyclic

A self-healing code-execution agent for Python. Cyclic asks an LLM (via [LiteLLM](https://github.com/BerriAI/litellm)) for code that solves a task you describe in plain English, runs it in a hardened Docker sandbox, and — when execution fails — feeds the error back to the LLM and tries again until the code succeeds or `--max-retries` is exhausted.

## What it does

```
prompt ──▶ LLM ──▶ Python code ──▶ AST scan ──▶ Docker sandbox ──▶ exit 0?
                       ▲                                              │
                       └──────────── error feedback ◀─────────────────┘
```

Three independent safety layers sit between the model's output and your machine:

- **Static AST scan** (`cyclic/safety.py`) rejects code with obviously dangerous imports or builtins before anything runs.
- **PEP 578 runtime audit hooks** (`cyclic/runner.py`) execute inside the container and intercept dynamic socket, subprocess, ctypes, and unsafe file operations that the static scan can't catch.
- **Docker hardening** (`cyclic/sandbox.py`) drops all Linux capabilities, disables the network, caps memory and CPU, and runs as the `nobody` user on a read-only root filesystem.

## How the loop works

`CyclicLoop.run()` in `cyclic/main.py`:

1. **Generate.** `Agent.generate()` calls the configured LLM with a system prompt that asks for JSON containing `code`, `reasoning`, and `confidence`.
2. **Scan.** `DockerSandbox.run()` calls `scan_code()` first; a safety violation short-circuits with `exit_code=1`.
3. **Execute.** The code is passed to a fresh container via the `CYCLIC_CODE_PAYLOAD` env var, executed by `runner.py` under the audit hooks.
4. **Retry.** On non-zero exit, the error is appended to the conversation history (sliding window of 20 messages) and the loop iterates. On success, the loop returns the code, stdout, and attempt count.

## Prerequisites

- **Python 3.12** (per `pyproject.toml` and `mise.toml`). Note: `.python-version` currently says `3.14` and is out of sync; the project targets 3.12.
- **Docker** daemon running locally — the sandbox uses `docker.from_env()`.
- **An LLM API key** — either `OPENAI_API_KEY` or `LITELLM_API_KEY` in your environment.
- **[uv](https://github.com/astral-sh/uv)** is recommended for dependency management (the repo ships `uv.lock`).

## Install

```bash
uv sync
```

## Configure

| Variable           | Required | Purpose                                                      |
| ------------------ | -------- | ------------------------------------------------------------ |
| `OPENAI_API_KEY`   | one of   | Picked up by `Agent.__init__` if `LITELLM_API_KEY` is unset. |
| `LITELLM_API_KEY`  | one of   | Preferred when set.                                          |

Any model that LiteLLM supports (OpenAI, Anthropic, local, etc.) can be passed with `--model`. The default is `gpt-4o-mini`.

## Usage

```bash
uv run python -m cyclic.main run "Find the two most frequent elements in [1,1,1,2,2,3]"
```

### Options

| Flag                 | Default       | Description                          |
| -------------------- | ------------- | ------------------------------------ |
| `--model`, `-m`      | `gpt-4o-mini` | Any LiteLLM-supported model ID.      |
| `--max-retries`, `-r`| `3`           | Maximum retry attempts on failure.   |
| `--timeout`, `-t`    | `10`          | Per-execution timeout in seconds.    |
| `--verbose`, `-v`    | off           | Print generated code, reasoning, confidence, and full stderr/stdout on failure. |

### Example with `--verbose`

```bash
uv run python -m cyclic.main run \
  "Compute the SHA-256 of the string 'cyclic'" \
  --model gpt-4o-mini \
  --max-retries 5 \
  --timeout 15 \
  --verbose
```

You'll see the generated code with syntax highlighting, the model's reasoning and confidence, a sandbox-execution spinner, and either the program output or a red error panel.

## Safety model

### Static AST scan — `cyclic/safety.py`

`SafetyScanner` walks the AST and rejects:

- Imports of `os`, `subprocess`, `sys`, `socket`, `shutil`, `importlib` (including submodules like `os.path`).
- Calls to `eval`, `exec`, `__import__`.

`open()` is deliberately **not** blocked here — runtime audit hooks restrict file access at execution time, allowing legitimate uses inside `/tmp` while catching escape attempts.

### Runtime audit hooks — `cyclic/runner.py`

Runs inside the container and uses PEP 578 (`sys.addaudithook`) to intercept dynamic behavior the static scan can't see: socket creation, subprocess spawning, `ctypes`, file operations outside permitted paths.

### Docker hardening — `cyclic/sandbox.py`

Every execution gets a fresh container with:

- `network_disabled=True` — no outbound network.
- `mem_limit="128m"` and CPU capped to one core.
- `read_only=True` root filesystem; only `/tmp` is writable (via `tmpfs`).
- `user="65534:65534"` (the `nobody` user).
- `cap_drop=['ALL']` — no Linux capabilities.
- Only `runner.py` is mounted into the container, read-only.
- Code is passed via the `CYCLIC_CODE_PAYLOAD` env var (no stdin races).

The container is forcibly removed in a `finally` block.

## Project layout

| File                  | Role                                                                                   |
| --------------------- | -------------------------------------------------------------------------------------- |
| `cyclic/main.py`      | Typer CLI and the `CyclicLoop` orchestrator.                                           |
| `cyclic/agent.py`     | LiteLLM wrapper; produces structured `AgentResponse` (`code`, `reasoning`, `confidence`). |
| `cyclic/sandbox.py`   | `DockerSandbox` — container lifecycle and hardening.                                   |
| `cyclic/safety.py`    | `SafetyScanner` AST visitor and `scan_code()`.                                         |
| `cyclic/runner.py`    | In-container entry point that installs PEP 578 audit hooks before `exec`-ing payload. |
| `tests/test_jailbreak.py` | Security regression tests (obfuscated imports, network attempts, file escapes). |

## Tests

```bash
uv run pytest tests/
```

Tests require a running Docker daemon — `test_jailbreak.py` exercises the real sandbox.

## Caveats

- `DockerSandbox` currently pins `python:3.14-slim` as its container image, while the host project targets Python 3.12. If the image isn't available for your platform, override it by instantiating `DockerSandbox(image="python:3.12-slim")`.
- `chromadb`, `tenacity`, and `python-dotenv` are listed in `pyproject.toml` but not yet wired into the visible code paths.
- Static and runtime safety layers reduce risk; they don't eliminate it. Don't run Cyclic against untrusted prompts on a machine you care about without reviewing the configuration first.

## License

MIT — see `LICENSE`.
