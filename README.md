# Cyclic

Self-healing code execution agent with a semantic cache and sandboxed runtime.

## Requirements
- Python 3.12 (pinned to <3.13)
- [uv](https://github.com/astral-sh/uv)
- Docker (for the sandbox)
- API key for code generation: set `LITELLM_API_KEY` or `OPENAI_API_KEY`

## Setup
```bash
uv sync
```

## Usage
Run the self-healing loop:
```bash
uv run cyclic run "write a fizzbuzz script"
```
Options:
- `--cache/--no-cache` (default: cache enabled)
- `--cache-threshold <float>` similarity threshold (default 0.85)
- `--model`, `--timeout`, `--max-retries`, `--verbose`
- `--verify/--no-verify` (default: verification enabled) - require agent-written
  self-tests and gate success on them passing
- `--check <assert-snippet>` (repeatable) - a user-supplied assert run after the
  code and self-tests in the sandbox; all must pass
- `--expect <substring>` (repeatable) - a substring that must appear in the
  program's stdout; checked host-side after a clean exit

Cache management:
```bash
uv run cyclic cache stats
uv run cyclic cache clear
```

## Testing
```bash
uv run --python 3.12 pytest tests/test_cache.py -v
```

## Security
- Code executes in a Docker sandbox with network disabled and strict audit hooks.
- Static safety scan blocks dangerous imports/functions before execution.

## Verification
- Success now means *verified*, not just "exited 0": the agent must supply
  deterministic self-tests (`test_code`) that run after the generated code in
  the same sandbox, and any `--check`/`--expect` constraints must also pass.
  Self-test or check failures feed back into the repair loop like execution
  errors. Use `--no-verify` to restore the old "exit 0 is success" behavior
  (in that mode, results are never written to the semantic cache).
- Only verified successes are admitted to the semantic cache, and cache hits
  are re-checked against any `--expect`/`--check` constraints on the current
  run before being served.

## Notes
- Semantic cache uses ChromaDB + LiteLLM embeddings; cache stored at `~/.cyclic/cache/`.
- Pydantic deprecation warning in litellm dependencies is upstream and harmless.