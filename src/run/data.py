"""Dataset access for the runners."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..api.config import ROOT


def data_dir() -> Path:
    """Corpus directory; ``DL_DATA_DIR`` lets a probe use an alternate corpus."""
    return Path(os.environ.get("DL_DATA_DIR") or (ROOT / "data"))


def docs_path() -> Path:
    return data_dir() / "docs.jsonl"


def questions_path() -> Path:
    return data_dir() / "questions.jsonl"


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_docs(
    split: str | None = None,
    limit: int | None = None,
    doc_ids: list[str] | None = None,
) -> list[dict]:
    docs = load_jsonl(docs_path())
    if doc_ids:
        wanted = set(doc_ids)
        docs = [d for d in docs if d["doc_id"] in wanted]
    elif split:
        docs = [d for d in docs if d.get("split") == split]
    if limit is not None:
        docs = docs[:limit]
    return docs


def _qid_sort_key(qid: str) -> tuple:
    tail = qid.rsplit("q", 1)[-1]
    try:
        return (0, int(tail))
    except ValueError:
        return (1, qid)


def load_questions_by_doc() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for q in load_jsonl(questions_path()):
        out.setdefault(q["doc_id"], []).append(q)
    for qs in out.values():
        qs.sort(key=lambda q: _qid_sort_key(q["qid"]))
    return out


def doc_bytes(doc: dict) -> int:
    return int(doc.get("bytes") or len(doc["text"].encode("utf-8")))
