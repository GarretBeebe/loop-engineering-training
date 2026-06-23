"""
Example 05: Error Recovery
==========================
Tools fail. The loop must handle this gracefully.

Two levels of recovery:

Level 1 — Tool errors:
    Catch exceptions in the tool runner and return them as a tool_result
    with an "error" prefix. Never let a tool exception crash the loop.
    The model sees the error message and can adapt (try a different path,
    rephrase the request, or give up gracefully).

Level 2 — API errors (rate limits, transient failures):
    Wrap the client.messages.create call in a retry with exponential backoff.
    The Anthropic SDK raises anthropic.RateLimitError on 429s.

In this example:
    - The model is given a bad filename first ("missing.txt")
    - The read_file tool raises FileNotFoundError
    - The error is returned as a tool result
    - The model recovers by trying "hello.txt" instead
"""

import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anthropic
from utils.tools import READ_FILE_DEF, run_tool

MODEL = "claude-haiku-4-5-20251001"
MAX_RETRIES = 3


def call_api_with_retry(client: anthropic.Anthropic, **kwargs) -> anthropic.types.Message:
    """Wrap API calls with exponential backoff on rate limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            print(f"  [rate limited] retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def run():
    client = anthropic.Anthropic()

    task = (
        "Please read the file 'missing.txt' and tell me what's in it. "
        "If that file doesn't exist, try 'hello.txt' instead."
    )
    messages = [{"role": "user", "content": task}]

    print(f"Task: {task}\n")

    turn = 0
    while True:
        turn += 1
        print(f"--- turn {turn} ---")

        response = call_api_with_retry(
            client,
            model=MODEL,
            max_tokens=512,
            tools=[READ_FILE_DEF],
            messages=messages,
        )

        print(f"stop_reason: {response.stop_reason}")
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    print(f"\nFinal answer: {block.text}")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                print(f"  Tool call: {block.name}({block.input})")
                try:
                    result = run_tool(block.name, block.input)
                    print(f"  Tool result: {result[:80]}...")
                except (FileNotFoundError, PermissionError, ValueError) as e:
                    # Return the error as a string — the model will read it and adapt
                    result = f"ERROR: {type(e).__name__}: {e}"
                    print(f"  Tool error returned to model: {result}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})
            print()


if __name__ == "__main__":
    run()
