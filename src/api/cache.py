"""SQLite completion cache keyed by the full request content hash (P1.3).

A cache hit performs no network call and is still written to the call log with
``cached=true`` so that every logical request is auditable.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from .config import cache_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS completions (
    cache_key      TEXT PRIMARY KEY,
    model_key      TEXT NOT NULL,
    provider       TEXT NOT NULL,
    api_model      TEXT NOT NULL,
    model_returned TEXT,
    request_json   TEXT NOT NULL,
    response_text  TEXT NOT NULL,
    usage_json     TEXT NOT NULL,
    request_id     TEXT,
    finish_reason  TEXT,
    ts             TEXT NOT NULL,
    latency_ms     INTEGER,
    cost_usd       REAL,
    prompt_version TEXT,
    tag            TEXT
);
CREATE INDEX IF NOT EXISTS idx_completions_model ON completions(model_key);
"""


class CompletionCache:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else cache_path()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def get(self, cache_key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT model_returned, response_text, usage_json, request_id, "
                "finish_reason, ts, latency_ms, cost_usd, prompt_version "
                "FROM completions WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "model_returned": row[0],
            "text": row[1],
            "usage_json": row[2],
            "request_id": row[3],
            "finish_reason": row[4],
            "ts": row[5],
            "latency_ms": row[6],
            "cost_usd": row[7],
            "prompt_version": row[8],
        }

    def put(self, cache_key: str, record: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO completions "
                "(cache_key, model_key, provider, api_model, model_returned, "
                " request_json, response_text, usage_json, request_id, finish_reason, "
                " ts, latency_ms, cost_usd, prompt_version, tag) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cache_key,
                    record["model_key"],
                    record["provider"],
                    record["api_model"],
                    record.get("model_returned"),
                    record["request_json"],
                    record["response_text"],
                    record["usage_json"],
                    record.get("request_id"),
                    record.get("finish_reason"),
                    record["ts"],
                    record.get("latency_ms"),
                    record.get("cost_usd"),
                    record.get("prompt_version"),
                    record.get("tag"),
                ),
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM completions").fetchone()[0])

    def stats(self) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), "
                "COALESCE(SUM(json_extract(usage_json,'$.prompt_tokens')),0), "
                "COALESCE(SUM(json_extract(usage_json,'$.completion_tokens')),0), "
                "COALESCE(SUM(cost_usd),0) FROM completions"
            ).fetchone()
        return {
            "cached_requests": int(row[0]),
            "prompt_tokens": int(row[1]),
            "completion_tokens": int(row[2]),
            "cost_usd": float(row[3] or 0.0),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
