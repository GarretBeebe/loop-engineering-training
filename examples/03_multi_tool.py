"""
Example 03: Multiple Tools
==========================
The agent picks which tool(s) to call — and Claude can call several in one turn.

Key difference from 02: a single response may contain multiple tool_use blocks.
The loop must collect ALL of them, run each, and return ALL results together.

Returning results one at a time (separate user messages) will cause an API error —
all tool_use blocks in a response must be resolved before continuing.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.provider import make_client, resolve_model, parse_args
from utils.tools import CALCULATE_DEF, GET_CURRENT_TIME_DEF, REVERSE_STRING_DEF, run_tool

TOOLS = [CALCULATE_DEF, GET_CURRENT_TIME_DEF, REVERSE_STRING_DEF]


def run(provider="anthropic", model=None):
    client = make_client(provider)
    MODEL = resolve_model(provider, model)

    task = (
        "Please do three things: "
        "(1) reverse the string 'hello world', "
        "(2) tell me the current time, and "
        "(3) calculate 7 * 8 + 3. "
        "Use your tools."
    )
    messages = [{"role": "user", "content": task}]

    print(f"Task: {task}\n")

    turn = 0
    while True:
        turn += 1
        print(f"--- turn {turn} ---")

        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            tools=TOOLS,
            messages=messages,
        )

        print(f"stop_reason: {response.stop_reason}")
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    print(f"\nFinal answer:\n{block.text}")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            print(f"  Claude requested {len(tool_use_blocks)} tool call(s):")

            for block in tool_use_blocks:
                print(f"    {block.name}({block.input})")
                try:
                    result = run_tool(block.name, block.input)
                except Exception as e:
                    result = f"ERROR: {e}"
                print(f"    -> {result}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            # ALL results returned in one user message
            messages.append({"role": "user", "content": tool_results})
            print()


if __name__ == "__main__":
    args = parse_args()
    run(provider=args.provider, model=args.model)
