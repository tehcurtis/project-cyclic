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

## Notes
- Semantic cache uses ChromaDB + LiteLLM embeddings; cache stored at `~/.cyclic/cache/`.
- Pydantic deprecation warning in litellm dependencies is upstream and harmless.