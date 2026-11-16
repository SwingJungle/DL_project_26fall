"""Byte-budget validator with four counters (P1.6).

The experimental unit is UTF-8 bytes (tokenizer-independent, D-5). The other
three counters are diagnostics:

* bytes      - binding constraint, ``len(text.encode("utf-8"))``
* chars      - Unicode code points
* words      - whitespace-delimited tokens
* tokens_est - planning estimate (~4 bytes/token for English). True per-receiver
               token counts come from ``usage.prompt_tokens`` in calls.jsonl,
               because no tokenizer is shipped with the gateway.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"\S+")


def count_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def count_chars(text: str) -> int:
    return len(text)


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def count_tokens_est(text: str) -> int:
    """Planning estimate only; never report it as a measured token count."""
    return max(1, math.ceil(count_bytes(text) / 4.0))


def measure(text: str) -> dict:
    return {
        "bytes": count_bytes(text),
        "chars": count_chars(text),
        "words": count_words(text),
        "tokens_est": count_tokens_est(text),
    }


@dataclass(frozen=True)
class BudgetWindow:
    """Inclusive byte window derived from a source document.

    ``lo``/``hi`` are absolute byte bounds; ``tol_abs`` is an absolute fraction
    of source bytes (0.02 == +/-2 percentage points of the source).
    """

    source_bytes: int
    target: int
    lo: int
    hi: int
    tol_abs: float = 0.02

    def compliant(self, n_bytes: int) -> bool:
        return self.lo <= n_bytes <= self.hi

    def direction(self, n_bytes: int) -> str:
        if n_bytes > self.hi:
            return "over"
        if n_bytes < self.lo:
            return "under"
        return "ok"

    def ratio(self, n_bytes: int) -> float:
        return n_bytes / self.source_bytes if self.source_bytes else 0.0

    def to_dict(self) -> dict:
        return {
            "source_bytes": self.source_bytes,
            "target": self.target,
            "lo": self.lo,
            "hi": self.hi,
            "tol_abs": self.tol_abs,
            "ratio_lo": round(self.lo / self.source_bytes, 4) if self.source_bytes else 0.0,
            "ratio_hi": round(self.hi / self.source_bytes, 4) if self.source_bytes else 0.0,
        }


def window_from_ratio(source_bytes: int, ratio: float = 0.50, tol_abs: float = 0.02) -> BudgetWindow:
    """Experiment mode: target = ratio x source, band = target +/- tol_abs x source."""
    target = int(round(source_bytes * ratio))
    lo = max(1, math.ceil(target - tol_abs * source_bytes))
    hi = max(lo, math.floor(target + tol_abs * source_bytes))
    return BudgetWindow(source_bytes, target, lo, hi, tol_abs)


def window_from_b(
    source_bytes: int, n_facts: int, b_per_fact: float, tol_abs: float = 0.02
) -> BudgetWindow:
    """Calibration mode: target = b x n_facts (fixed compression difficulty).

    Used only by P0.9. The experiment always uses ``window_from_ratio(..., 0.50)``.
    """
    if n_facts <= 0:
        raise ValueError("n_facts must be positive")
    target = int(round(b_per_fact * n_facts))
    lo = max(1, math.ceil(target - tol_abs * source_bytes))
    hi = max(lo, math.floor(target + tol_abs * source_bytes))
    return BudgetWindow(source_bytes, target, lo, hi, tol_abs)


def budget_report(text: str, window: BudgetWindow) -> dict:
    counts = measure(text)
    counts.update(
        {
            "target": window.target,
            "lo": window.lo,
            "hi": window.hi,
            "ratio_source": round(counts["bytes"] / window.source_bytes, 4)
            if window.source_bytes
            else 0.0,
            "compliant": window.compliant(counts["bytes"]),
            "direction": window.direction(counts["bytes"]),
            "delta_bytes": counts["bytes"] - window.target,
        }
    )
    return counts
