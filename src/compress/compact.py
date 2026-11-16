"""Compact structured message: array-of-pairs serialization (P6.0).

Motivation (G2, protocol tax): the full schema spends about 19 bytes per item on
JSON syntax (``{"k":"E","x":""}`` plus separators). The compact schema keeps the
*same* semantics -- one kind letter and one statement per fact -- but serializes
each item as a two-element array::

    [["E","Peothmem founded the Tergi Survey"],["D","Moved in 2046"]]

Per-item overhead drops from ~19 B to ~9 B, so at the same 50% UTF-8 byte
budget the sender has more room for statements. This is the single design
choice tested as the course-required "small improvement" ablation.

The measured message is always the canonical compact serialization, never the
raw model output: whitespace is normalized and validation is strict, exactly
like :mod:`src.compress.structured`. A malformed message is rejected (and
retried), never silently repaired.
"""
from __future__ import annotations

import json

from .structured import ALLOWED_KINDS, MAX_FACTS

# Empirical per-item syntax cost at n_units=7: outer brackets + 7 x 8 B items
# + 6 commas = 64 B, i.e. ~9.1 B/item (full schema: ~18.9 B/item).
PER_ITEM_OVERHEAD = 9


class CompactFormatError(ValueError):
    pass


def parse_message(text: str) -> dict:
    """Parse ``[["E","..."],...]`` into ``{"v": 1, "f": [{"k","x"},...]}``."""
    stripped = text.strip()
    if not stripped:
        raise CompactFormatError("empty output")
    if stripped.startswith("```") or stripped.endswith("```"):
        raise CompactFormatError("output wrapped in a code fence")
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise CompactFormatError(f"invalid JSON: {exc}") from None
    if not isinstance(data, list):
        raise CompactFormatError(
            f"top level is {type(data).__name__}, expected array"
        )
    if not 1 <= len(data) <= MAX_FACTS:
        raise CompactFormatError(f"array length {len(data)} outside 1..{MAX_FACTS}")

    facts = []
    for i, item in enumerate(data):
        if not isinstance(item, list):
            raise CompactFormatError(f"item {i} is not an array")
        if len(item) != 2:
            raise CompactFormatError(f"item {i} has {len(item)} elements, expected 2")
        kind, statement = item
        if kind not in ALLOWED_KINDS:
            raise CompactFormatError(f"item {i} kind {kind!r} not in {ALLOWED_KINDS}")
        if not isinstance(statement, str) or len(statement.strip()) < 3:
            raise CompactFormatError(f"item {i} statement must be a string of length >= 3")
        facts.append({"k": kind, "x": statement.strip()})
    return {"v": 1, "f": facts}


def serialize(message: dict) -> str:
    return json.dumps(
        [[item["k"], item["x"]] for item in message["f"]],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def canonicalize(text: str) -> tuple[str, dict]:
    message = parse_message(text)
    return serialize(message), message


def reserialize_full(message: dict) -> str:
    """Same facts, compact serialization -- the overhead-only control.

    Changing only the container (objects -> two-element arrays) while holding
    the selected facts fixed lets us separate two mechanisms behind any compact
    gain: (a) the sender fits more facts when syntax is cheaper, and (b) the
    receiver reads a compact encoding differently. This function isolates (b).
    """
    return serialize(message)
