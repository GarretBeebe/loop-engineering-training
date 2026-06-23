"""
Example 02: The Tool Loop (ReAct Pattern)
=========================================
This is the core pattern for tool-using agents.

The loop:
    1. Call the API
    2. If stop_reason == "tool_use": run the tool, append result, go to 1
    3. If stop_reason == "end_turn": we're done

Tool results are appended as a "user" message with type "tool_result".
The message list grows with every iteration — the model sees the full history.

Watch the printed messages list to see it expand across turns.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anthropic
from utils.tools import CALCULATE_DEF, run_tool

MODEL = "claude-haiku-4-5-20251001"


def run():
    client = anthropic.Anthropic()

    task = "What is (17 * 4) + (sqrt(144) / 3)? Show your reasoning."
    messages = [{"role": "user", "content": task}]

    print(f"Task: {task}\n")

    turn = 0
    while True:
        turn += 1
        print(f"--- turn {turn} | messages in context: {len(messages)} ---")

        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            tools=[CALCULATE_DEF],
            messages=messages,
        )

        print(f"stop_reason: {response.stop_reason}")

        # Always append the assistant's response to history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract and print the final text answer
            for block in response.content:
                if block.type == "text":
                    print(f"\nFinal answer: {block.text}")
            break

        if response.stop_reason == "tool_use":
            # Handle every tool call in this response
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  Tool call: {block.name}({block.input})")
                    try:
                        result = run_tool(block.name, block.input)
                        print(f"  Tool result: {result}")
                    except Exception as e:
                        result = f"ERROR: {e}"
                        print(f"  Tool error: {result}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Tool results go back as a user message
            messages.append({"role": "user", "content": tool_results})
            print()


if __name__ == "__main__":
    run()
