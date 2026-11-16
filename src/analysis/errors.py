"""Error examples for the report (course requirement: 5-10 concrete failures).

Each example is a question the receiver got wrong, tied back to the fact that
carried the answer, the source sentence, the compressed message and whether the
fact survived the budget. The point is to show *which kind of content* the
schema loses at high density (numbers, negations, conditions) and what the
receiver does instead (guesses, says unknown, confuses an entity).

Usage:
    python -m src.analysis.errors
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from ..eval.retention import fact_scores, message_tokens
from ..run.data import load_docs, load_questions_by_doc
from .common import read_jsonl

DATA_DIR = "data/_density_probe_paired/b100"


def _load(data_dir: str):
    import os

    os.environ["DL_DATA_DIR"] = data_dir
    docs = {d["doc_id"]: d for d in load_docs()}
    questions = load_questions_by_doc()
    return docs, questions


def _messages(path: str) -> dict[str, dict]:
    return {m["key"]: m for m in read_jsonl(path) if m.get("message")}


def _fid_of(question: dict) -> list[str]:
    return list(question.get("fact_ids") or [])


def _missing_keys(doc: dict, message: dict, q: dict) -> list[str]:
    """Fact keys (numbers, names, gold spans) absent from the message."""
    scores = {f["fid"]: f for f in fact_scores(doc, message["message"], message["method"])}
    tokens, numbers = message_tokens(message["message"])
    missing = []
    for fid in _fid_of(q):
        for key in scores.get(fid, {}).get("keys", []):
            present = key in numbers if key.replace(".", "", 1).isdigit() else key in tokens
            if not present and key not in missing:
                missing.append(key)
    return missing


def _evidence_survived(message: dict, q: dict) -> bool | None:
    retention = message.get("retention") or {}
    facts = {f["fid"]: f for f in retention.get("facts", [])}
    ids = _fid_of(q)
    if not ids or not facts:
        return None
    return all(facts.get(fid, {}).get("survived") for fid in ids)


def collect(
    *,
    message_path: str,
    prediction_path: str,
    methods: tuple[str, ...],
    limit: int = 8,
) -> list[dict]:
    docs, questions = _load(DATA_DIR)
    messages = _messages(message_path)
    predictions = read_jsonl(prediction_path)
    by_question = {q["qid"]: q for qs in questions.values() for q in qs}
    examples = []
    for rec in predictions:
        if rec["method"] not in methods:
            continue
        msg = messages.get(rec["key"].rsplit("|recv:", 1)[0])
        if not msg:
            continue
        for pq in rec.get("per_question") or []:
            if pq.get("correct"):
                continue
            q = by_question.get(pq["qid"])
            if not q:
                continue
            doc = docs.get(rec["doc_id"], {})
            facts = {f["fid"]: f for f in doc.get("facts", [])}
            fact_texts = [facts[fid]["text"] for fid in _fid_of(q) if fid in facts]
            examples.append(
                {
                    "doc_id": rec["doc_id"],
                    "method": rec["method"],
                    "level": rec["level"],
                    "sender": rec["sender_model"],
                    "question": q.get("text"),
                    "qid": q["qid"],
                    "demand": q.get("demand"),
                    "answerable": q.get("answerable"),
                    "gold": pq.get("gold"),
                    "pred": pq.get("pred"),
                    "fact_ids": _fid_of(q),
                    "source_fact": " | ".join(fact_texts),
                    "evidence_survived": _evidence_survived(msg, q),
                    "missing_keys": _missing_keys(doc, msg, q),
                    "message_bytes": rec["message_bytes"],
                    "message": msg["message"],
                    "ratio": msg["ratio_source"],
                }
            )
    # Prefer failures where the evidence fact was demonstrably dropped.
    dropped = [e for e in examples if e["evidence_survived"] is False]
    hallucinated = [
        e for e in examples if e["answerable"] is False and str(e["pred"]).strip().lower() != "unknown"
    ]
    ordered = dropped + [e for e in hallucinated if e not in dropped] + [e for e in examples if e not in dropped and e not in hallucinated]
    # Diversify: one example per document, then fill if needed.
    out: list[dict] = []
    seen_docs: set[str] = set()
    for e in ordered:
        if e["doc_id"] in seen_docs:
            continue
        seen_docs.add(e["doc_id"])
        out.append(e)
        if len(out) >= limit:
            return out
    for e in ordered:
        if e in out:
            continue
        out.append(e)
        if len(out) >= limit:
            break
    return out


def _select_contrast(
    *, structured_path: str, structured_predictions: str, compact_path: str, compact_predictions: str
) -> list[dict]:
    """Up to two examples where compact is right and full JSON is wrong on the
    same question (same document, sender, receiver)."""
    docs, questions = _load(DATA_DIR)
    by_question = {q["qid"]: q for qs in questions.values() for q in qs}
    messages = {
        "structured": _messages(structured_path),
        "compact": _messages(compact_path),
    }
    preds = {
        "structured": read_jsonl(structured_predictions),
        "compact": read_jsonl(compact_predictions),
    }

    def index(records):
        out = {}
        for rec in records:
            for q in rec.get("per_question") or []:
                key = (rec["doc_id"], rec["sender_model"], rec["receiver_model"], q["qid"])
                out[key] = {"record": rec, "q": q}
        return out

    idx_s = index([r for r in preds["structured"] if r["method"] == "structured" and r["pair_type"] == "self"])
    idx_c = index([r for r in preds["compact"] if r["method"] == "compact" and r["pair_type"] == "self"])
    out = []
    seen_docs: set[str] = set()
    for key in sorted(set(idx_s) & set(idx_c)):
        s, c = idx_s[key], idx_c[key]
        if s["q"].get("correct") or not c["q"].get("correct"):
            continue
        doc_id, sender, receiver, qid = key
        if doc_id in seen_docs:
            continue
        q = by_question.get(qid)
        if not q:
            continue
        seen_docs.add(doc_id)
        doc = docs.get(doc_id, {})
        facts = {f["fid"]: f for f in doc.get("facts", [])}
        msg_key_s = s["record"]["key"].rsplit("|recv:", 1)[0]
        msg_key_c = c["record"]["key"].rsplit("|recv:", 1)[0]
        out.append(
            {
                "doc_id": doc_id,
                "sender": sender,
                "receiver": receiver,
                "qid": qid,
                "question": q.get("text"),
                "gold": s["q"].get("gold"),
                "full_pred": s["q"].get("pred"),
                "compact_pred": c["q"].get("pred"),
                "source_fact": " | ".join(facts[fid]["text"] for fid in _fid_of(q) if fid in facts),
                "full_message": messages["structured"].get(msg_key_s, {}).get("message"),
                "compact_message": messages["compact"].get(msg_key_c, {}).get("message"),
                "full_evidence": _evidence_survived(messages["structured"].get(msg_key_s, {}), q),
                "compact_evidence": _evidence_survived(messages["compact"].get(msg_key_c, {}), q),
            }
        )
        if len(out) >= 2:
            break
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/error_examples.md")
    parser.add_argument("--json-out", default="reports/error_examples.json")
    args = parser.parse_args(argv)

    dropped = collect(
        message_path="results/v3probe4/messages.jsonl",
        prediction_path="results/v3probe4/predictions.jsonl",
        methods=("structured",),
        limit=6,
    )
    compact_wins = _select_contrast(
        structured_path="results/compact_probe/messages.jsonl",
        structured_predictions="results/compact_probe/predictions.jsonl",
        compact_path="results/compact_probe/messages.jsonl",
        compact_predictions="results/compact_probe/predictions.jsonl",
    )

    lines = [
        "# Error examples",
        "",
        "> Generated by `python -m src.analysis.errors` from frozen predictions.",
        "> Zero new API calls. Each row is a question the receiver answered wrong,",
        "> tied to the fact that carried the answer and whether it survived the",
        "> 50% budget.",
        "",
        "## A. Structured (full JSON) at b100: the dropped fact explains the error",
        "",
        "| # | Doc | QID | Question | Gold | Receiver answer | Fact that answered it | Missing keys | Evidence survived? | Message bytes |",
        "|---|---|---|---|---|---|---|---|---:|---:|",
    ]
    for i, e in enumerate(dropped, 1):
        lines.append(
            f"| {i} | {e['doc_id']} | {e['qid']} | {e['question']} | {e['gold']} | "
            f"{e['pred']} | {e['source_fact'][:220]} | {', '.join(e['missing_keys']) or '-'} | "
            f"{e['evidence_survived']} | {e['message_bytes']} |"
        )
    lines += ["", "### A2. Three full examples (copy-paste ready)", ""]
    for i, e in enumerate(dropped[:3], 1):
        lines += [
            f"**Example {i} — {e['doc_id']} / {e['method']} / {e['sender']} / "
            f"{e['message_bytes']} B ({100*e['ratio']:.0f}% of source)**",
            "",
            f"- Question: {e['question']}",
            f"- Gold: `{e['gold']}`; receiver answered `{e['pred']}`",
            f"- Source fact: {e['source_fact']}",
            f"- Missing keys: `{', '.join(e['missing_keys']) or '-'}`",
            "",
            "```json",
            e["message"],
            "```",
            "",
        ]
    lines += [
        "",
        "Notes on the failure kinds (read off the source fact and the message):",
        "",
        "- **Numbers/dates** are the most common casualty; when the year or",
        "  threshold is dropped the receiver answers a plausible sibling value.",
        "- **Negations and conditions** ('not X', 'only below N degrees') are",
        "  fragile: the message keeps the relation but drops the polarity or the",
        "  threshold, and the receiver answers the unmarked version.",
        "- **Unanswerable questions** become guessable when the message keeps",
        "  related entities; the receiver answers instead of saying unknown.",
        "",
        "## B. Same question: full JSON wrong, compact right (G2 probe)",
        "",
    ]
    if compact_wins:
        lines += [
            "| # | Doc | Question | Gold | Full JSON answer | Compact answer | Source fact |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, e in enumerate(compact_wins, 1):
            lines.append(
                f"| {i} | {e['doc_id']} | {e['question']} | {e['gold']} | {e['full_pred']} | "
                f"{e['compact_pred']} | {e['source_fact'][:220]} |"
            )
        lines += [
            "",
            "These are paired (same document, sender, receiver and question), so the",
            "difference is what the extra statement bytes bought: the compact message",
            "kept the number/condition that the full-schema message had to drop.",
            "",
        ]
    else:
        lines += [
            "No paired full-wrong/compact-right case found in the probe predictions.",
            "",
        ]
    lines += [
        "## C. What to quote in the report",
        "",
        "Pick 3–4 of the rows above for the paper: one dropped number, one dropped",
        "negation, one dropped condition, and one compact contrast. Show the",
        "original sentence, the message substring, and the receiver answer — the",
        "point is that the loss is a **whole fact**, not a truncation.",
        "",
    ]
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump({"dropped": dropped, "compact_wins": compact_wins}, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"wrote {args.out} and {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
