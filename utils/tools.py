"""
Shared toy tool implementations used across the examples.

Each tool has two parts:
  1. A Python function that does the actual work
  2. A TOOL_DEF dict in Anthropic's tool-definition schema

Import both and pass the TOOL_DEF list to client.messages.create(tools=...).
When Claude calls a tool, dispatch to the matching function.
"""

import math
import json
import os
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Tool: calculate
# ---------------------------------------------------------------------------

def calculate(expression: str) -> str:
    """Evaluate a math expression safely using Python's math module."""
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed["abs"] = abs
    allowed["round"] = round
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return str(result)
    except Exception as e:
        raise ValueError(f"Could not evaluate '{expression}': {e}") from e


CALCULATE_DEF = {
    "name": "calculate",
    "description": "Evaluate a mathematical expression. Supports standard math functions (sqrt, sin, cos, log, etc.) and arithmetic operators.",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A Python math expression, e.g. '17 * 4' or 'sqrt(144) / 3'",
            }
        },
        "required": ["expression"],
    },
}

# ---------------------------------------------------------------------------
# Tool: get_current_time
# ---------------------------------------------------------------------------

def get_current_time() -> str:
    return datetime.now().isoformat()


GET_CURRENT_TIME_DEF = {
    "name": "get_current_time",
    "description": "Return the current local date and time in ISO 8601 format.",
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

# ---------------------------------------------------------------------------
# Tool: reverse_string
# ---------------------------------------------------------------------------

def reverse_string(text: str) -> str:
    return text[::-1]


REVERSE_STRING_DEF = {
    "name": "reverse_string",
    "description": "Reverse the characters in a string.",
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The string to reverse.",
            }
        },
        "required": ["text"],
    },
}

# ---------------------------------------------------------------------------
# Tool: read_file
# ---------------------------------------------------------------------------

_ALLOWED_DIR = Path(__file__).parent.parent / "sample_files"


def read_file(path: str) -> str:
    """Read a file from the sample_files/ directory only (no path traversal)."""
    target = (_ALLOWED_DIR / path).resolve()
    if not str(target).startswith(str(_ALLOWED_DIR.resolve())):
        raise PermissionError(f"Access denied: '{path}' is outside sample_files/")
    return target.read_text()


READ_FILE_DEF = {
    "name": "read_file",
    "description": "Read the contents of a file from the sample_files/ directory.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Filename inside sample_files/, e.g. 'hello.txt'",
            }
        },
        "required": ["path"],
    },
}

# ---------------------------------------------------------------------------
# Dispatcher: call any tool by name
# ---------------------------------------------------------------------------

TOOL_DEFS = [CALCULATE_DEF, GET_CURRENT_TIME_DEF, REVERSE_STRING_DEF, READ_FILE_DEF]

_DISPATCH = {
    "calculate": lambda args: calculate(args["expression"]),
    "get_current_time": lambda args: get_current_time(),
    "reverse_string": lambda args: reverse_string(args["text"]),
    "read_file": lambda args: read_file(args["path"]),
}


def run_tool(name: str, args: dict) -> str:
    """Dispatch a tool call by name. Returns a string result or raises."""
    if name not in _DISPATCH:
        raise ValueError(f"Unknown tool: {name!r}")
    return _DISPATCH[name](args)
