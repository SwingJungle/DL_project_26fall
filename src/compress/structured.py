"""Structured message: parse, validate and canonicalize (P3.2 / P3.3).

The measured message is always the canonical compact serialization, never the
raw model output: key order is fixed, keys are dropped only by rejecting (never
silently), and every byte of JSON syntax counts toward the budget.
"""
from __future__ import annotations

import json

ALLOWED_KINDS = ("E", "D", "R", "N", "C", "T", "U", "S")
TOP_KEYS = ("v", "f")
ITEM_KEYS = ("k", "x")
MAX_FACTS = 10


class StructuredFormatError(ValueError):
    pass


def parse_message(text: str) -> dict:
    stripped = text.strip()
    if not stripped:
        raise StructuredFormatError("empty output")
    if stripped.startswith("```") or stripped.endswith("```"):
        raise StructuredFormatError("output wrapped in a code fence")
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise StructuredFormatError(f"invalid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise StructuredFormatError(f"top level is {type(data).__name__}, expected object")

    extra = [k for k in data if k not in TOP_KEYS]
    if extra:
        raise StructuredFormatError(f"unexpected top-level keys: {extra}")
    missing = [k for k in TOP_KEYS if k not in data]
    if missing:
        raise StructuredFormatError(f"missing top-level keys: {missing}")
    if data["v"] != 1:
        raise StructuredFormatError(f"v must be 1, got {data['v']!r}")

    facts = data["f"]
    if not isinstance(facts, list):
        raise StructuredFormatError("f must be an array")
    if not 1 <= len(facts) <= MAX_FACTS:
        raise StructuredFormatError(f"f length {len(facts)} outside 1..{MAX_FACTS}")
    seen = []
    for i, item in enumerate(facts):
        if not isinstance(item, dict):
            raise StructuredFormatError(f"f[{i}] is not an object")
        extra = [k for k in item if k not in ITEM_KEYS]
        if extra:
            raise StructuredFormatError(f"f[{i}] has unexpected keys: {extra}")
        if "k" not in item or "x" not in item:
            raise StructuredFormatError(f"f[{i}] missing k or x")
        if item["k"] not in ALLOWED_KINDS:
            raise StructuredFormatError(f"f[{i}].k={item['k']!r} not in {ALLOWED_KINDS}")
        if not isinstance(item["x"], str) or len(item["x"].strip()) < 3:
            raise StructuredFormatError(f"f[{i}].x must be a string of length >= 3")
        seen.append({"k": item["k"], "x": item["x"].strip()})
    return {"v": 1, "f": seen}


def serialize(message: dict) -> str:
    return json.dumps(message, ensure_ascii=False, separators=(",", ":"))


def canonicalize(text: str) -> tuple[str, dict]:
    message = parse_message(text)
    return serialize(message), message


def render_schema_doc() -> str:
    """Human-readable schema block reused by tests and docs."""
    return (
        '{"v":1,"f":[{"k":"E","x":"..."},...]}\n'
        "k: E entity | D number/date | R positive relation | N negation | "
        "C condition/exception | T temporal order | U attribution/uncertainty | S causal"
    )
