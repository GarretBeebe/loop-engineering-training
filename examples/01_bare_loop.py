"""
Example 01: The Bare Loop
=========================
The skeleton every agent loop is built on.

Pattern:
    messages = [user_turn]
    while True:
        response = client.messages.create(messages=messages)
        messages.append(assistant_turn)
        if done: break
        messages.append(next_user_turn)

With no tools, stop_reason is always "end_turn" — so the exit condition
here is content-based: we count turns and send "continue" to force
multiple rounds. This makes the loop structure visible without tool complexity.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.provider import make_client, resolve_model, parse_args

LINES_WANTED = 3  # a haiku has exactly 3 lines


def run(provider="anthropic", model=None):
    client = make_client(provider)
    MODEL = resolve_model(provider, model)

    messages = [
        {
            "role": "user",
            "content": (
                f"Write me a haiku about programming, but give me exactly one line per reply. "
                f"Stop after each line and wait. I will say 'next' to get the next line. "
                f"Start with line 1 now."
            ),
        }
    ]

    print(f"User: {messages[0]['content']}\n")

    collected_lines = []
    turn = 0

    while True:
        turn += 1
        print(f"--- turn {turn} ---")

        response = client.messages.create(
            model=MODEL,
            max_tokens=128,
            messages=messages,
        )

        # A bare loop response is always a single text block
        assistant_text = next((b.text for b in response.content if b.type == "text"), "").strip()
        print(f"Assistant: {assistant_text}")
        print(f"stop_reason: {response.stop_reason}\n")

        collected_lines.append(assistant_text)

        # Append assistant turn so the model remembers what it said
        messages.append({"role": "assistant", "content": response.content})

        # Exit when we have all the lines we wanted
        if len(collected_lines) >= LINES_WANTED:
            break

        # Otherwise, push the conversation forward
        messages.append({"role": "user", "content": "next"})

    print("=== Final poem ===")
    for line in collected_lines:
        print(line)


if __name__ == "__main__":
    args = parse_args()
    run(provider=args.provider, model=args.model)
