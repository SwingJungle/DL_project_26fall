"""Build the same-facts compact control for the G2 probe.

Takes the *full-schema* messages from the compact probe and re-serializes each
one as the compact array, without changing a single selected fact. Running the
receivers on these derived messages isolates the **encoding** effect from the
**budget** effect:

* compact vs full (regenerated): differs in serialization **and** in which
  facts the sender kept (cheaper syntax leaves room for more facts);
* compact-reserialized vs full: differs only in serialization (identical facts,
  shorter message).

Usage:
    python -m src.analysis.make_reserialize \
        --messages results/compact_probe/messages.jsonl \
        --out results/compact_reserialize/messages.jsonl
"""
from __future__ import annotations

import argparse
import json

from ..compress.compact import reserialize_full
from ..compress.structured import parse_message
from ..common import now_iso, read_jsonl
from ..eval.retention import retention_summary
from ..run.data import load_docs, load_questions_by_doc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", default="results/compact_probe/messages.jsonl")
    parser.add_argument("--out", default="results/compact_reserialize/messages.jsonl")
    parser.add_argument("--data-dir", default="data/_density_probe_paired/b100")
    args = parser.parse_args(argv)

    import os

    os.environ["DL_DATA_DIR"] = args.data_dir
    docs = {doc["doc_id"]: doc for doc in load_docs()}
    questions = load_questions_by_doc()

    sources = [m for m in read_jsonl(args.messages) if m["method"] == "structured" and m.get("message")]
    out = []
    bytes_saved = 0
    for msg in sources:
        parsed = parse_message(msg["message"])
        compact = reserialize_full(parsed)
        bytes_saved += int(msg["final_bytes"]) - len(compact.encode("utf-8"))
        doc = docs[msg["doc_id"]]
        window = msg["window"]
        n_bytes = len(compact.encode("utf-8"))
        compliant = int(window["lo"]) <= n_bytes <= int(window["hi"])
        direction = "ok" if compliant else ("over" if n_bytes > int(window["hi"]) else "under")
        rec = {
            "key": f"{msg['doc_id']}|{msg['model_key']}|compact_rs|{msg['level']}",
            "run_id": "compact_reserialize",
            "created_at": now_iso(),
            "doc_id": msg["doc_id"],
            "model_key": msg["model_key"],
            "method": "compact",
            "level": msg["level"],
            "level_kind": msg["level_kind"],
            "level_value": msg["level_value"],
            "window": msg["window"],
            "message": compact,
            "final_bytes": n_bytes,
            "ratio_source": round(n_bytes / int(window["source_bytes"]), 4)
            if int(window["source_bytes"])
            else 0.0,
            "compliant": compliant,
            "direction": direction,
            "n_items": len(parsed["f"]),
            "json_valid": True,
            "n_attempts": 0,
            "first_attempt_compliant": False,
            "prompt_version": "p6.0-compact-rs",
            "derived_from": "structured",
            "source_key": msg["key"],
            "doc_bytes": msg["doc_bytes"],
            "n_facts": msg["n_facts"],
            "style": "readable",
            "retention": retention_summary(doc, compact, questions.get(msg["doc_id"], []), "compact"),
        }
        out.append(rec)

    import os as _os

    _os.makedirs(_os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        fh.flush()
    print(
        json.dumps(
            {
                "messages": len(out),
                "out": args.out,
                "total_bytes_saved": bytes_saved,
                "mean_bytes_saved": round(bytes_saved / len(out), 1) if out else 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
