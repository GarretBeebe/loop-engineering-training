# Agent Loop Engineering — Training

Seven progressive examples that build a complete mental model for how production agent loops work. Each example runs against either the Anthropic API or a locally-hosted Ollama model — no frameworks, just the raw loop.

## What's covered

| # | File | Concept |
|---|------|---------|
| 01 | `examples/01_bare_loop.py` | The `while True` / `break` structure; messages list as memory |
| 02 | `examples/02_tool_loop.py` | ReAct pattern; `stop_reason` as the loop signal |
| 03 | `examples/03_multi_tool.py` | Batching multiple tool calls in one turn |
| 04 | `examples/04_stopping.py` | Max-iteration ceiling, keyword signal, judge model |
| 05 | `examples/05_error_recovery.py` | Catching tool errors and returning them as `tool_result` |
| 06 | `examples/06_orchestrator.py` | Sub-agents: wrapping an API call as a tool |
| 07 | `examples/07_human_in_loop.py` | Intercepting tool calls for human approval |

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

**Anthropic** (default):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Ollama** (local):
```bash
ollama pull llama3.2   # requires https://ollama.com
```

## Run an example

```bash
# Anthropic (default)
uv run python examples/01_bare_loop.py

# Ollama
uv run python examples/01_bare_loop.py --provider ollama
uv run python examples/02_tool_loop.py --provider ollama --model qwen2.5
```

Every example accepts `--provider {anthropic,ollama}` and `--model <name>`. You can also set defaults via environment variables:

```bash
export PROVIDER=ollama
export MODEL=qwen2.5
uv run python examples/02_tool_loop.py
```

Example 06 (orchestrator) has separate flags for each model tier:

```bash
uv run python examples/06_orchestrator.py --provider ollama \
  --orchestrator-model llama3.1:8b --worker-model llama3.2
```

Example 07 requires interactive input — run it separately:

```bash
uv run python examples/07_human_in_loop.py
```

> **Ollama + tools:** Examples 02–07 use function calling. Use a model that supports it — `llama3.2`, `qwen2.5`, and `mistral-nemo` work well. `phi3` does not.

## Smoke test (non-interactive examples)

```bash
for f in examples/0{1,2,3,4,5,6}_*.py; do
    echo "=== $f ===" && uv run python "$f"
done
```

## Full guide

See [GUIDE.md](GUIDE.md) for detailed explanations, key lines to study, and exercises for each example.
