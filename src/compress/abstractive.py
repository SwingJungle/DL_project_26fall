"""Abstractive message finalization (P3.1).

Only whitespace is stripped. Any other post-processing would turn the
abstractive condition into a different method and bias the Method contrast.
"""
from __future__ import annotations

import re

_PREAMBLE_RE = re.compile(r"^(here|sure|certainly|of course)\b", re.IGNORECASE)


class AbstractiveFormatError(ValueError):
    pass


def finalize(text: str) -> tuple[str, dict]:
    stripped = text.strip()
    if not stripped:
        raise AbstractiveFormatError("empty output")
    if stripped.startswith("```") or stripped.endswith("```"):
        raise AbstractiveFormatError("output wrapped in a code fence")
    if stripped.startswith("{") and stripped.endswith("}"):
        raise AbstractiveFormatError("output is a JSON object; write plain prose")
    if _PREAMBLE_RE.match(stripped):
        raise AbstractiveFormatError("output starts with a conversational preamble")
    return stripped, {"fence": False, "json": False, "preamble": False}
