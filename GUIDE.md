# Agent Loop Engineering — A Practical Guide

Work through these examples in order. Each one adds exactly one new idea.
By the end you'll have the full mental model for how production agent loops work.

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd agentic-loop-examples
uv sync

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Example 01 — The Bare Loop

**File:** `examples/01_bare_loop.py`

**Concept:** Every agent loop is a `while True` with one exit condition. No tools needed to understand the shape. The messages list is the agent's memory — it grows with every turn.

```
messages = [user_turn]
while True:
    response = api_call(messages)
    messages.append(assistant_turn)   # model remembers what it said
    if done: break
    messages.append(next_user_turn)   # push it forward
```

**Run it:**
```bash
uv run python examples/01_bare_loop.py
```

**Key lines to study:**
- The `while True` / `break` structure (lines 52–72)
- How `messages.append(...)` grows the context each turn
- `[turn N]` in the output — watch how many API calls happen

**Exercises:**
1. Change `LINES_WANTED` to 2 or 6 and re-run. What changes?
2. Print `len(messages)` inside the loop. How many messages accumulate per turn?

---

## Example 02 — The Tool Loop (ReAct)

**File:** `examples/02_tool_loop.py`

**Concept:** The canonical ReAct pattern. `stop_reason` is the signal that drives the loop. `"tool_use"` means the model wants to call a tool and is not done. `"end_turn"` means it's finished.

```
while True:
    response = api_call(messages, tools=TOOLS)
    messages.append(response)

    if response.stop_reason == "end_turn":
        break                           # done — extract the final text

    if response.stop_reason == "tool_use":
        results = run_all_tools(response)
        messages.append(tool_results)   # feed results back, loop again
```

Tool results go back as a **user** message with `type: "tool_result"`. The model reads them in the next turn and decides whether to call another tool or answer.

**Run it:**
```bash
uv run python examples/02_tool_loop.py
```

**Key lines to study:**
- The `stop_reason` branch (lines 44–48) — this is the heart of every tool loop
- How `tool_results` is structured (lines 54–62) — notice `tool_use_id` links result to call
- The printed `messages in context: N` counter — it grows by 2 per tool round-trip

**Exercises:**
1. Change the task to a pure text question (no math). How many turns does it take?
2. Add a `print(messages)` dump at the end. Study the full structure of a complete loop.

---

## Example 03 — Multiple Tools

**File:** `examples/03_multi_tool.py`

**Concept:** Claude can call multiple tools in a single turn — one `tool_use` block per call. You must collect **all** of them and return **all** results in one user message. Returning results one at a time is an API error.

The agent autonomously picks which tools to use and in what order. You just define the interface and run whatever it asks for.

**Run it:**
```bash
uv run python examples/03_multi_tool.py
```

**Key lines to study:**
- `[b for b in response.content if b.type == "tool_use"]` (line 47) — collect all tool calls
- The single `messages.append(...)` for all results (line 60) — they must travel together
- The output line `Claude requested N tool call(s)` — does it batch them or call one at a time?

**Exercises:**
1. Remove one tool from the `TOOLS` list. Does Claude adapt or error?
2. Ask a task that needs only one tool. Confirm it doesn't call the others unnecessarily.

---

## Example 04 — Stopping Conditions

**File:** `examples/04_stopping.py`

**Concept:** Three strategies for deciding when the loop is done:

| Strategy | Mechanism | Best for |
|---|---|---|
| **Max iterations** | Count turns, break at ceiling | Budget guard, known-length tasks |
| **Keyword signal** | Model emits `DONE:` when satisfied | Open-ended tasks, cheap to implement |
| **Confidence judge** | Second model call evaluates completion | High-stakes tasks, unreliable signals |

Every loop needs at least a max-iterations safety ceiling even if you use another strategy.

**Run it:**
```bash
uv run python examples/04_stopping.py
```

**Key lines to study:**
- Strategy A: the `for turn in range(1, max_turns + 1)` ceiling (line 73)
- Strategy B: `if text.strip().startswith("DONE:")` (line 104) — keyword check
- Strategy C: `is_complete()` (lines 116–122) — the judge uses its own API call

**Exercises:**
1. For Strategy B, change the signal to `"FINAL:"` — update both the system prompt and the check.
2. For Strategy C, modify the judge prompt to be stricter. Does it run more iterations?

---

## Example 05 — Error Recovery

**File:** `examples/05_error_recovery.py`

**Concept:** Tools fail. Never let a tool exception crash the loop. Catch it, format it as a string, and return it as a `tool_result`. The model reads the error and decides what to do next — retry, fallback, or give up.

Two levels:
1. **Tool errors** — catch in your dispatcher, return `"ERROR: ..."` as the tool result
2. **API errors** — wrap `client.messages.create` in retry+backoff for rate limits

**Run it:**
```bash
uv run python examples/05_error_recovery.py
```

**Key lines to study:**
- `except (FileNotFoundError, ...) as e: result = f"ERROR: ..."` (lines 65–68) — the catch
- The tool result still has `tool_use_id` — even errors must be paired with their call
- `call_api_with_retry()` (lines 26–37) — exponential backoff pattern

**Exercises:**
1. Change the task to give Claude three bad filenames. How far does it search?
2. Catch `Exception` broadly instead of specific types. What's the risk?

---

## Example 06 — Orchestrator + Worker

**File:** `examples/06_orchestrator.py`

**Concept:** Sub-agents are API calls wrapped in a tool. The orchestrator sees a tool called `ask_worker`. When it calls it, your code makes a separate API call to a cheaper model. The orchestrator never sees the worker's chain-of-thought — only the final answer.

```
Orchestrator (sonnet) --[ask_worker tool]--> Python function
                                               -> Worker (haiku) API call
                                               <- worker's answer
                       <-[tool_result]--------
```

This pattern scales: workers can have their own tool loops, you can run multiple workers in parallel with `concurrent.futures`, and you can chain orchestrators.

**Run it:**
```bash
uv run python examples/06_orchestrator.py
```

**Key lines to study:**
- `run_worker()` (lines 30–38) — the worker is just a function that returns a string
- `WORKER_TOOL_DEF` (lines 41–57) — this is what makes it look like a tool to the orchestrator
- The `[worker]` log lines in output — shows the two-model system working

**Exercises:**
1. Swap the orchestrator to `claude-haiku-4-5-20251001`. How does the quality change?
2. Give the worker its own tool (e.g. `calculate`). Now it's a full sub-agent with a tool loop.

---

## Example 07 — Human-in-the-Loop

**File:** `examples/07_human_in_loop.py`

**Concept:** Intercept tool calls before executing them. Show the human what the model wants to do and require approval. On rejection, return a rejection string — the model reads it and must propose an alternative. The loop continues until the human approves a safe action or the model gives up.

This is the safety layer for any irreversible action: file deletes, emails, API writes, deployments.

**Run it:**
```bash
uv run python examples/07_human_in_loop.py
```
When prompted `Allow? [y/n]`, try `n` first to see the model adapt, then `y` to let it succeed.

**Key lines to study:**
- `human_approve()` (lines 46–51) — this is where execution pauses
- The rejection result string (line 70) — what you tell the model matters
- `execute_command()` (lines 40–42) — stubbed here; in production, use `subprocess.run`

**Exercises:**
1. Change the rejection message to be more specific: `"REJECTED: too destructive. Use a safer rm."`. Does the model's alternative improve?
2. Add an `is_dangerous(command)` heuristic that auto-rejects any command containing `rm -rf` without asking.

---

## The Full Mental Model

```
┌─────────────────────────────────────────────────────┐
│                  Agent Loop                          │
│                                                      │
│  messages = [user_task]                              │
│                                                      │
│  while True:                                         │
│    response = api(messages, tools)    ← 02, 03       │
│    messages.append(response)                         │
│                                                      │
│    if stop_reason == "end_turn": break  ← 01, 02    │
│    if stop_reason == "tool_use":                     │
│      for each tool_call:                             │
│        if needs_human_approval:        ← 07          │
│          approved = ask_human()                      │
│        try:                                          │
│          result = run_tool(call)       ← 03          │
│        except:                                       │
│          result = format_error(e)      ← 05          │
│      messages.append(all_results)                    │
│                                                      │
│    if max_iter or signal or judge: break  ← 04      │
│                                                      │
│  # run_tool() can itself be an API call  ← 06       │
└─────────────────────────────────────────────────────┘
```

Each example peels off one layer of this diagram. Once you can write this loop from memory, you understand the core mechanics behind every agent framework (LangChain, LlamaIndex, Claude Code itself).

---

## Smoke Test

Run all non-interactive examples back to back:

```bash
for f in examples/0{1,2,3,4,5,6}_*.py; do
    echo "Running $f..."
    uv run python "$f" && echo "OK: $f" || echo "FAILED: $f"
    echo "---"
done
```

Example 07 requires interactive input — run it separately:
```bash
uv run python examples/07_human_in_loop.py
```
