"""
Example 07: Human-in-the-Loop
==============================
The loop pauses at a decision point and requires human approval.

Pattern:
    When the model calls a "dangerous" tool, intercept it before execution.
    Show the human what the model wants to do and prompt for [y/n].
    If approved: run the tool normally, return the result.
    If rejected: return a "rejected" tool result. The model reads this and
                 must propose an alternative.

This pattern applies to anything irreversible: shell commands, file deletes,
API writes, emails, payments. The model plans; the human approves.

Note: This example does NOT actually execute any shell commands —
execute_command() is stubbed to simulate what would happen.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.provider import make_client, resolve_model, parse_args

EXECUTE_COMMAND_DEF = {
    "name": "execute_command",
    "description": "Execute a shell command on the local machine.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run.",
            }
        },
        "required": ["command"],
    },
}


def execute_command(command: str) -> str:
    """Stubbed — in a real system this would run subprocess.run(command, shell=True)."""
    return f"[simulated] Command '{command}' executed successfully."


def human_approve(command: str) -> bool:
    """Pause and ask the human whether to allow this command."""
    print(f"\n  *** APPROVAL REQUIRED ***")
    print(f"  The agent wants to run: {command!r}")
    answer = input("  Allow? [y/n]: ").strip().lower()
    return answer == "y"


def run(provider="anthropic", model=None):
    client = make_client(provider)
    MODEL = resolve_model(provider, model)

    task = (
        "I need you to clean up my /tmp/test directory. "
        "Use the execute_command tool to do it."
    )
    messages = [{"role": "user", "content": task}]

    print(f"Task: {task}\n")
    print("(You will be prompted to approve or reject each command.)\n")

    turn = 0
    while True:
        turn += 1
        print(f"--- turn {turn} ---")

        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            tools=[EXECUTE_COMMAND_DEF],
            messages=messages,
        )

        print(f"stop_reason: {response.stop_reason}")
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    print(f"\nAgent: {block.text}")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                command = block.input["command"]

                if human_approve(command):
                    result = execute_command(command)
                    print(f"  Approved. Result: {result}")
                else:
                    # Inject a rejection — the model must propose an alternative
                    result = "REJECTED by user. Please suggest a safer alternative."
                    print(f"  Rejected. Telling model to try something else.")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})
            print()


if __name__ == "__main__":
    args = parse_args()
    run(provider=args.provider, model=args.model)
