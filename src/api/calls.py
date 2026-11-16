"""Per-run call log: one JSON line per logical request (P1.5).

Full request messages are stored for auditability. No credential is ever
included. Both cache hits and failed attempts are logged; ``status`` separates
``ok`` / ``error`` and ``cached`` marks cache hits.
"""
from __future__ import annotations

from pathlib import Path

from .config import run_dir
from ..common import JsonlWriter, now_iso


class CallLogger:
    def __init__(self, path: str | Path | None = None, *, run_id: str | None = None):
        if path is None:
            if run_id is None:
                raise ValueError("provide path or run_id")
            path = run_dir(run_id) / "calls.jsonl"
        self.path = Path(path)
        self._writer = JsonlWriter(self.path)

    def log(self, record: dict) -> dict:
        record = {"ts": now_iso(), **record}
        self._writer.append(record)
        return record
