"""
Example 06: Orchestrator + Worker (Multi-Agent)
================================================
Sub-agents are just API calls wrapped in a tool interface.

Pattern:
    Orchestrator (smart, expensive model) breaks a task into sub-questions
    and calls a "worker" tool for each. The worker tool is a Python function
    that internally runs its own agent loop (or single API call). The
    orchestrator sees only the worker's final answer, not its reasoning steps.

This gives you:
    - Clean separation of planning (orchestrator) vs execution (worker)
    - Different models per layer (cheap worker, capable orchestrator)
    - Workers can be run in parallel (not shown here, but the pattern extends)

The orchestrator model here is claude-sonnet-4-6 because it needs to do
real multi-step reasoning. The worker uses haiku since each sub-task is simple.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anthropic

ORCHESTRATOR_MODEL = "claude-sonnet-4-6"
WORKER_MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Worker: a standalone mini-agent (single call for simplicity here)
# ---------------------------------------------------------------------------

def run_worker(client: anthropic.Anthropic, prompt: str) -> str:
    """Answer a focused sub-question. Returns the response text."""
    print(f"    [worker] prompt: {prompt[:80]}...")
    response = client.messages.create(
        model=WORKER_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.content[0].text.strip()
    print(f"    [worker] answer: {answer[:80]}...")
    return answer


# ---------------------------------------------------------------------------
# Worker tool definition (what the orchestrator sees)
# ---------------------------------------------------------------------------

WORKER_TOOL_DEF = {
    "name": "ask_worker",
    "description": (
        "Delegate a focused sub-question to a worker agent and get back a concise answer. "
        "Use this to break a complex task into smaller, answerable questions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "A specific, self-contained question for the worker to answer.",
            }
        },
        "required": ["question"],
    },
}


# ---------------------------------------------------------------------------
# Orchestrator loop
# ---------------------------------------------------------------------------

def run():
    client = anthropic.Anthropic()

    task = (
        "Summarize the key pros and cons of using async Python (asyncio/FastAPI) "
        "versus sync Python (Flask/Django) for building web APIs. "
        "Use your worker tool to research each side before synthesizing."
    )

    messages = [{"role": "user", "content": task}]
    print(f"Task: {task}\n")
    print(f"Orchestrator: {ORCHESTRATOR_MODEL}")
    print(f"Worker:       {WORKER_MODEL}\n")

    turn = 0
    while True:
        turn += 1
        print(f"--- orchestrator turn {turn} ---")

        response = client.messages.create(
            model=ORCHESTRATOR_MODEL,
            max_tokens=2048,
            tools=[WORKER_TOOL_DEF],
            messages=messages,
        )

        print(f"stop_reason: {response.stop_reason}")
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason in ("end_turn", "max_tokens"):
            if response.stop_reason == "max_tokens":
                print("  [warning] response truncated — increase max_tokens if needed")
            for block in response.content:
                if block.type == "text":
                    print(f"\n=== Final synthesis ===\n{block.text}")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                print(f"  Orchestrator delegating: {block.input['question'][:80]}")
                worker_answer = run_worker(client, block.input["question"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": worker_answer,
                })

            messages.append({"role": "user", "content": tool_results})
            print()


if __name__ == "__main__":
    run()
