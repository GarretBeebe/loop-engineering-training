# Agent Loop Engineering — Training

Seven progressive examples that build a complete mental model for how production agent loops work, using the Anthropic API directly (no frameworks).

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

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run an example

```bash
uv run python examples/01_bare_loop.py
```

Run all non-interactive examples back to back:

```bash
for f in examples/0{1,2,3,4,5,6}_*.py; do
    echo "=== $f ===" && uv run python "$f"
done
```

Example 07 requires interactive input — run it separately:

```bash
uv run python examples/07_human_in_loop.py
```

## Full guide

See [GUIDE.md](GUIDE.md) for detailed explanations, key lines to study, and exercises for each example.
