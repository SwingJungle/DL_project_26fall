"""4x4 pairing matrix analysis (pilot / main experiment).

Reads messages.jsonl + predictions.jsonl, computes accuracy by PairType, the
SelfAdvantage / FamilyAdvantage contrasts with a document-level bootstrap CI,
and the Method main effect. Outputs a markdown table and a JSON summary.

Usage:
    python -m src.run.matrix_report --messages results/pilot_trans/messages.jsonl \
        --predictions results/pilot_trans/predictions.jsonl \
        --out reports/pilot_trans_matrix.md --json-out reports/pilot_trans_matrix.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

PAIR_TYPES = ("self", "same", "cross_matched", "cross_unmatched")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def accuracy(rows: list[dict], demand: str | None = None) -> tuple[int, int]:
    q = [
        x
        for r in rows
        for x in r["per_question"]
        if demand is None or x["demand"] == demand
    ]
    return sum(x["correct"] for x in q), len(q)


def _boot_self_adv(predictions: list[dict], method: str, n_boot: int = 3000, seed: int = 7):
    docs = sorted({r["doc_id"] for r in predictions if r["level"] != "full"})
    per = {"self": defaultdict(lambda: [0, 0]), "cross_matched": defaultdict(lambda: [0, 0])}
    for r in predictions:
        if r["level"] == "full" or r["method"] != method:
            continue
        if r["pair_type"] not in per:
            continue
        for q in r["per_question"]:
            per[r["pair_type"]][r["doc_id"]][0] += int(q["correct"])
            per[r["pair_type"]][r["doc_id"]][1] += 1
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [rng.choice(docs) for _ in docs]
        sk = sum(per["self"][d][0] for d in sample)
        sn = sum(per["self"][d][1] for d in sample)
        ck = sum(per["cross_matched"][d][0] for d in sample)
        cn = sum(per["cross_matched"][d][1] for d in sample)
        if sn and cn:
            diffs.append(sk / sn - ck / cn)
    diffs.sort()
    if not diffs:
        return None
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]


def summarize(messages: list[dict], predictions: list[dict]) -> dict:
    by_method = {}
    for method in sorted({r["method"] for r in predictions if r["level"] != "full"}):
        rows = [r for r in predictions if r["method"] == method]
        cells = {}
        for pt in PAIR_TYPES:
            sub = [r for r in rows if r["pair_type"] == pt]
            k, n = accuracy(sub)
            cells[pt] = {
                "n_correct": k,
                "n_questions": n,
                "accuracy": k / n if n else None,
                "n_records": len(sub),
            }
        self_a = cells["self"]["accuracy"]
        matched = cells["cross_matched"]["accuracy"]
        same = cells["same"]["accuracy"]
        ci = _boot_self_adv(predictions, method)
        by_method[method] = {
            "cells": cells,
            "self_advantage": (self_a - matched) if self_a is not None and matched is not None else None,
            "self_advantage_ci": ci,
            "family_advantage": (same - matched) if same is not None and matched is not None else None,
        }

    full_rows = [r for r in predictions if r["level"] == "full"]
    fk, fn = accuracy(full_rows)
    method_main = {}
    for method, data in by_method.items():
        k = sum(c["n_correct"] for c in data["cells"].values())
        n = sum(c["n_questions"] for c in data["cells"].values())
        method_main[method] = {"n_correct": k, "n_questions": n, "accuracy": k / n if n else None}

    delta = None
    if "abstractive" in by_method and "structured" in by_method:
        a = by_method["abstractive"]["self_advantage"]
        s = by_method["structured"]["self_advantage"]
        if a is not None and s is not None:
            delta = a - s

    per_demand = {}
    for method in by_method:
        per_demand[method] = {}
        for demand in ("lookup", "multi-hop"):
            d = {}
            for pt in ("self", "same", "cross_matched"):
                sub = [
                    r for r in predictions
                    if r["level"] != "full" and r["method"] == method and r["pair_type"] == pt
                ]
                k, n = accuracy(sub, demand)
                d[pt] = {"n_correct": k, "n_questions": n, "accuracy": k / n if n else None}
            per_demand[method][demand] = d

    per_sender = {}
    for method in by_method:
        per_sender[method] = {}
        for sender in sorted({r["sender_model"] for r in predictions if r["method"] == method and r["level"] != "full"}):
            vals = {}
            for pt in ("self", "cross_matched"):
                sub = [
                    r for r in predictions
                    if r["level"] != "full" and r["method"] == method
                    and r["pair_type"] == pt and r["sender_model"] == sender
                ]
                k, n = accuracy(sub)
                vals[pt] = k / n if n else None
            if vals["self"] is not None and vals["cross_matched"] is not None:
                per_sender[method][sender] = {
                    "self": vals["self"],
                    "cross_matched": vals["cross_matched"],
                    "self_advantage": vals["self"] - vals["cross_matched"],
                }

    return {
        "n_messages": len(messages),
        "n_predictions": len(predictions),
        "n_docs": len({r["doc_id"] for r in messages}),
        "full": {"n_correct": fk, "n_questions": fn, "accuracy": fk / fn if fn else None},
        "by_method": by_method,
        "method_main_effect": method_main,
        "delta_method": delta,
        "by_demand": per_demand,
        "by_sender": per_sender,
    }


def render_markdown(summary: dict, title: str = "4×4 pairing matrix") -> str:
    lines = [f"# {title}", ""]
    lines.append(
        f"- docs: {summary['n_docs']} · messages: {summary['n_messages']} · "
        f"predictions: {summary['n_predictions']}"
    )
    full = summary["full"]
    lines.append(
        f"- full-document baseline: {full['n_correct']}/{full['n_questions']} = "
        f"{full['accuracy']:.3f}" if full["accuracy"] is not None else "- full: —"
    )
    lines.append("")
    lines.append("| method | self | same | matched cross | unmatched cross | SelfAdv | 95% CI |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method, data in summary["by_method"].items():
        c = data["cells"]
        ci = data["self_advantage_ci"]
        ci_s = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "—"
        lines.append(
            f"| {method} | {c['self']['accuracy']:.3f} | {c['same']['accuracy']:.3f} | "
            f"{c['cross_matched']['accuracy']:.3f} | {c['cross_unmatched']['accuracy']:.3f} | "
            f"{data['self_advantage']:+.3f} | {ci_s} |"
        )
    lines.append("")
    lines.append("## Method main effect (all pair types pooled)")
    lines.append("")
    lines.append("| method | correct | n | accuracy |")
    lines.append("|---|---:|---:|---:|")
    for method, data in summary["method_main_effect"].items():
        lines.append(
            f"| {method} | {data['n_correct']} | {data['n_questions']} | {data['accuracy']:.3f} |"
        )
    lines.append("")
    if summary["delta_method"] is not None:
        lines.append(f"`Δ_method = SelfAdv_abs − SelfAdv_struct = {summary['delta_method']:+.3f}`")
        lines.append("")
    lines.append("## By demand (self / same / matched)")
    lines.append("")
    for method, demands in summary["by_demand"].items():
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| demand | self | same | matched |")
        lines.append("|---|---:|---:|---:|")
        for demand, d in demands.items():
            lines.append(
                f"| {demand} | {d['self']['accuracy']:.3f} | {d['same']['accuracy']:.3f} | "
                f"{d['cross_matched']['accuracy']:.3f} |"
            )
        lines.append("")
    lines.append("## By sender (self − matched)")
    lines.append("")
    lines.append("| method | sender | self | matched | SelfAdv |")
    lines.append("|---|---|---:|---:|---:|")
    for method, senders in summary["by_sender"].items():
        for sender, d in senders.items():
            lines.append(
                f"| {method} | {sender} | {d['self']:.3f} | {d['cross_matched']:.3f} | "
                f"{d['self_advantage']:+.3f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--title", default="4×4 pairing matrix")
    args = ap.parse_args()
    messages = read_jsonl(args.messages)
    predictions = read_jsonl(args.predictions)
    summary = summarize(messages, predictions)
    text = render_markdown(summary, args.title)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
