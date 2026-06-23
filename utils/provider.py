"""
Provider abstraction — run examples against Anthropic or a local Ollama instance.

Usage:
    from utils.provider import make_client, resolve_model, parse_args, RATE_LIMIT_ERRORS

    args   = parse_args()
    client = make_client(args.provider)
    model  = resolve_model(args.provider, args.model)

    response = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": "Hello"}],
        tools=[...],   # optional — Anthropic tool-definition format
        system="...",  # optional
    )

    # response.stop_reason : "end_turn" | "tool_use" | "max_tokens"
    # response.content     : list of TextBlock / ToolUseBlock
    for block in response.content:
        if block.type == "text":
            print(block.text)
        elif block.type == "tool_use":
            print(block.name, block.input, block.id)

The Ollama adapter translates Anthropic-format messages and tool definitions to
OpenAI's chat-completions format (which Ollama exposes at /v1) and maps the
response back to the same shape the examples expect.
"""

from __future__ import annotations

import json
import os
import types
from dataclasses import dataclass, field
from typing import Any

import anthropic

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.2",
}

# ---------------------------------------------------------------------------
# Rate-limit exception tuple — covers both providers for example 05.
# _openai is set to None when the package is absent so the rest of the
# module degrades gracefully (Anthropic path still works; Ollama errors at
# client construction time with a clear message).
# ---------------------------------------------------------------------------

try:
    import openai as _openai
    RATE_LIMIT_ERRORS: tuple[type[Exception], ...] = (
        anthropic.RateLimitError,
        _openai.RateLimitError,
    )
except ImportError:
    _openai = None  # type: ignore[assignment]
    RATE_LIMIT_ERRORS = (anthropic.RateLimitError,)

# ---------------------------------------------------------------------------
# Normalized response types (Anthropic-compatible shape)
# The Ollama path returns these dataclasses; the Anthropic path returns
# the real SDK objects — both have the same attributes the examples access.
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class ToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class Response:
    stop_reason: str = "end_turn"
    content: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Translation: Anthropic format → OpenAI format
# ---------------------------------------------------------------------------

def _block_attr(block: Any, attr: str) -> Any:
    """Read an attribute from either a dict block or an object block."""
    return block[attr] if isinstance(block, dict) else getattr(block, attr)


def _to_openai_tools(tools: list[dict] | None) -> list[dict] | None:
    """Anthropic tool defs → OpenAI function defs.
    input_schema and parameters are both JSON Schema — only the wrapper changes.
    """
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _to_openai_messages(messages: list[dict], system: str | None) -> list[dict]:
    """Translate the Anthropic-format message history to OpenAI chat format.

    Key differences:
    - Anthropic assistant content is a list of block objects; OpenAI wants a
      string for .content and a tool_calls list.
    - Anthropic tool results are a list inside one user message; OpenAI uses
      separate messages with role="tool" for each result.
    """
    result: list[dict] = []

    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            if isinstance(content, str):
                result.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # Collect tool results in one pass; fall through to text if none.
                tool_msgs = [
                    {
                        "role": "tool",
                        "tool_call_id": item["tool_use_id"],
                        "content": str(item["content"]),
                    }
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "tool_result"
                ]
                if tool_msgs:
                    result.extend(tool_msgs)
                else:
                    parts = [
                        _block_attr(item, "text")
                        for item in content
                        if _block_attr(item, "type") == "text"
                    ]
                    result.append({"role": "user", "content": " ".join(parts)})

        elif role == "system":
            result.append({
                "role": "system",
                "content": content if isinstance(content, str) else "",
            })

        elif role == "assistant":
            if isinstance(content, str):
                result.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text_parts: list[str] = []
                tool_calls: list[dict] = []
                for block in content:
                    btype = _block_attr(block, "type")
                    if btype == "text":
                        t = _block_attr(block, "text")
                        if t:
                            text_parts.append(t)
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id": _block_attr(block, "id"),
                            "type": "function",
                            "function": {
                                "name": _block_attr(block, "name"),
                                "arguments": json.dumps(_block_attr(block, "input")),
                            },
                        })

                assistant_msg: dict = {
                    "role": "assistant",
                    "content": " ".join(text_parts) or "",
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                result.append(assistant_msg)

    return result


def _from_openai_response(oai_response: Any) -> Response:
    """Map an OpenAI chat-completion response back to our normalized Response."""
    choice = oai_response.choices[0]
    message = choice.message

    content: list = []
    if message.content:
        content.append(TextBlock(text=message.content))
    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                tool_input = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                tool_input = {}
            content.append(ToolUseBlock(
                id=tc.id,
                name=tc.function.name,
                input=tool_input,
            ))

    # Derive stop_reason from what was actually produced, not finish_reason alone.
    # Some Ollama builds set finish_reason="tool_calls" before populating tool_calls,
    # which would otherwise send example loops into an infinite spin.
    if any(isinstance(b, ToolUseBlock) for b in content):
        stop_reason = "tool_use"
    elif choice.finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    return Response(stop_reason=stop_reason, content=content)


# ---------------------------------------------------------------------------
# Ollama client (wraps openai.OpenAI pointed at localhost:11434)
# ---------------------------------------------------------------------------

class _OllamaMessages:
    def __init__(self, base_url: str) -> None:
        if _openai is None:
            raise ImportError(
                "The 'openai' package is required for Ollama support.\n"
                "Install it with:  uv add openai  or  pip install openai"
            )
        self._client = _openai.OpenAI(base_url=base_url, api_key="ollama")

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
        **_: Any,
    ) -> Response:
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _to_openai_messages(messages, system),
        }
        oai_tools = _to_openai_tools(tools)
        if oai_tools:
            kwargs["tools"] = oai_tools

        return _from_openai_response(self._client.chat.completions.create(**kwargs))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def make_client(
    provider: str = "anthropic",
    base_url: str = "http://localhost:11434/v1",
) -> Any:
    """Return a client for the given provider.

    Both clients expose client.messages.create(**kwargs) with Anthropic-style
    keyword arguments. Responses have .stop_reason and .content.
    """
    if provider == "anthropic":
        return anthropic.Anthropic()
    if provider == "ollama":
        url = os.getenv("OLLAMA_BASE_URL", base_url)
        messages_proxy = _OllamaMessages(url)
        return types.SimpleNamespace(messages=messages_proxy)
    raise ValueError(f"Unknown provider {provider!r}. Choose 'anthropic' or 'ollama'.")


def resolve_model(provider: str, model: str | None = None) -> str:
    """Return model, or the default model for provider when model is None."""
    return model or DEFAULT_MODELS[provider]


def parse_args(description: str | None = None):
    """Parse --provider and --model CLI flags shared by all examples."""
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--provider",
        default=os.getenv("PROVIDER", "anthropic"),
        choices=list(DEFAULT_MODELS),
    )
    p.add_argument("--model", default=os.getenv("MODEL"))
    return p.parse_args()
