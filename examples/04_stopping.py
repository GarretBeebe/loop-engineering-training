"""
Example 04: Stopping Conditions
================================
Three strategies for terminating an agent loop — demonstrated side by side.

Strategy A — Max iterations:
    Hard ceiling. Simple and safe. The model doesn't need to know about it.
    Use when tasks have a known upper bound or you want a budget guard.

Strategy B — Keyword signal:
    Instruct the model to emit a special token (e.g. "DONE:") when finished.
    The loop watches for it in the text. Cheap — no extra API call.
    Risk: the model may forget the signal or use it prematurely.

Strategy C — Confidence judge:
    After each turn, call a second cheap model to evaluate whether the task
    is complete. More reliable but costs an extra call per iteration.
    Use for open-ended tasks where "done" is hard to define structurally.

All three run on the same task: iteratively improve a one-sentence summary.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.provider import make_client, resolve_model, parse_args

TEXT_TO_SUMMARIZE = (
    "Python is a high-level, interpreted programming language known for its clear syntax "
    "and readability. It supports multiple programming paradigms including procedural, "
    "object-oriented, and functional programming. Python's extensive standard library and "
    "vibrant ecosystem of third-party packages make it suitable for web development, "
    "data science, machine learning, automation, and many other domains."
)


# ---------------------------------------------------------------------------
# Strategy A: Max iterations
# ---------------------------------------------------------------------------

def strategy_a_max_iterations(client, model: str, max_turns: int = 3):
    print("\n=== Strategy A: Max Iterations ===")
    messages = [
        {
            "role": "user",
            "content": (
                f"Summarize this text in one sentence, then on the next turn improve it. "
                f"Keep refining.\n\nText: {TEXT_TO_SUMMARIZE}"
            ),
        }
    ]

    for turn in range(1, max_turns + 1):
        response = client.messages.create(
            model=model, max_tokens=256, messages=messages
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        print(f"Turn {turn}: {text[:120]}...")
        messages.append({"role": "assistant", "content": response.content})

        if turn < max_turns:
            messages.append({"role": "user", "content": "Improve your summary."})

    print(f"Stopped: reached max {max_turns} iterations.")


# ---------------------------------------------------------------------------
# Strategy B: Keyword signal
# ---------------------------------------------------------------------------

def strategy_b_keyword(client, model: str):
    print("\n=== Strategy B: Keyword Signal ===")
    messages = [
        {
            "role": "user",
            "content": (
                f"Summarize this text in one sentence, then improve it each turn. "
                f"When your summary is as good as it can be, start your reply with 'DONE:'. "
                f"Otherwise start with 'DRAFT:'.\n\nText: {TEXT_TO_SUMMARIZE}"
            ),
        }
    ]

    turn = 0
    while True:
        turn += 1
        response = client.messages.create(
            model=model, max_tokens=256, messages=messages
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        print(f"Turn {turn}: {text[:120]}...")
        messages.append({"role": "assistant", "content": response.content})

        if text.strip().startswith("DONE:"):
            print("Stopped: model signaled completion.")
            break

        messages.append({"role": "user", "content": "Keep improving."})

        if turn >= 10:  # safety ceiling
            print("Stopped: safety ceiling hit.")
            break


# ---------------------------------------------------------------------------
# Strategy C: Confidence judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are evaluating whether a one-sentence summary is complete and polished.
Reply with only "YES" if it is ready, or "NO" if it could still be improved."""


def is_complete(client, model: str, summary: str) -> bool:
    response = client.messages.create(
        model=model,
        max_tokens=8,
        system=JUDGE_PROMPT,
        messages=[{"role": "user", "content": f'Summary: "{summary}"'}],
    )
    verdict = next((b.text for b in response.content if b.type == "text"), "").strip().upper()
    return verdict.startswith("YES")


def strategy_c_judge(client, model: str):
    print("\n=== Strategy C: Confidence Judge ===")
    messages = [
        {
            "role": "user",
            "content": f"Summarize this text in one sentence:\n\n{TEXT_TO_SUMMARIZE}",
        }
    ]

    turn = 0
    while True:
        turn += 1
        response = client.messages.create(
            model=model, max_tokens=256, messages=messages
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        print(f"Turn {turn}: {text[:120]}...")
        messages.append({"role": "assistant", "content": response.content})

        if is_complete(client, model, text):
            print("Stopped: judge approved the summary.")
            break

        messages.append({"role": "user", "content": "Can you improve this summary?"})

        if turn >= 5:
            print("Stopped: safety ceiling hit.")
            break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(provider="anthropic", model=None):
    client = make_client(provider)
    MODEL = resolve_model(provider, model)
    strategy_a_max_iterations(client, MODEL)
    strategy_b_keyword(client, MODEL)
    strategy_c_judge(client, MODEL)


if __name__ == "__main__":
    args = parse_args()
    run(provider=args.provider, model=args.model)
