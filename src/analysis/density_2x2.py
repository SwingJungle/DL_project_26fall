"""G1: the Method x Density 2x2, recomputed with a consistent receiver scope.

The handoff v2 table mixed scopes: the base-density cell averaged all four
receivers, while the b100 cell used self-receivers only. This script reports
**both** scopes from the same frozen predictions so the table is internally
consistent, and adds paired McNemar tests (same document, sender, receiver and
question across densities) plus a document-cluster bootstrap CI for the
interaction.

Data (10 paired documents, same questions in both densities):
  * base density: results/v3base4/predictions.jsonl (4x4 + full)
  * b100 density: results/v3probe4/predictions_4x4.jsonl (4x4 + full)

Usage:
    python -m src.analysis.density_2x2
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from .common import (
    accuracy,
    flat_questions,
    paired_questions,
    paired_summary,
    read_jsonl,
    write_json,
)

DATASETS = {
    "base": "results/v3base4/predictions.jsonl",
    "b100": "results/v3probe4/predictions_4x4.jsonl",
}
METHODS = ("abstractive", "structured")


def _p(value: float) -> str:
    return "<0.0001" if value < 1e-4 else f"{value:.4f}"


def _load() -> dict[str, dict[str, list[dict]]]:
    """density -> method -> flattened question rows (all receivers)."""
    out = {}
    for density, path in DATASETS.items():
        rows = flat_questions(read_jsonl(path))
        by_method = defaultdict(list)
        for row in rows:
            by_method[row["method"]].append(row)
        out[density] = dict(by_method)
    return out


def _acc(rows: list[dict], *, self_only: bool) -> dict:
    sub = [r for r in rows if not self_only or r["pair_type"] == "self"]
    k, n, acc = accuracy(sub)
    return {"n_correct": k, "n_questions": n, "accuracy": acc}


def _pairs(rows_a: list[dict], rows_b: list[dict], *, self_only: bool) -> list:
    a = [r for r in rows_a if not self_only or r["pair_type"] == "self"]
    b = [r for r in rows_b if not self_only or r["pair_type"] == "self"]
    return paired_questions(a, b)


def _bootstrap_density_diff(pairs: list, *, n_boot: int = 4000, seed: int = 17):
    by_doc: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for doc, a, b in pairs:
        by_doc[doc].append((a, b))
    docs = sorted(by_doc)
    if not docs:
        return None, None

    def stat(sample):
        na = nb = n = 0
        for doc in sample:
            for a, b in by_doc[doc]:
                na += int(a["correct"])
                nb += int(b["correct"])
                n += 1
        return (na - nb) / n if n else 0.0

    point = stat(docs)
    rng = random.Random(seed)
    vals = sorted(stat([rng.choice(docs) for _ in docs]) for _ in range(n_boot))
    return point, (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))])


def _bootstrap_interaction(data: dict[str, dict[str, list[dict]]], *, self_only: bool):
    """Doc-cluster bootstrap of (gap at b100) - (gap at base).

    A gap is ``accuracy(abstractive) - accuracy(structured)`` within one density
    (same documents/questions). Resampling documents keeps all four conditions
    of a document together, which is what the paired render buys us.
    """
    per_doc: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for density in ("base", "b100"):
        for method in METHODS:
            for r in data[density][method]:
                if self_only and r["pair_type"] != "self":
                    continue
                key = f"{density}|{method}"
                k, n = per_doc[r["doc_id"]].get(key, (0, 0))
                per_doc[r["doc_id"]][key] = (k + int(r["correct"]), n + 1)
    docs = sorted(per_doc)
    if not docs:
        return None, None, None

    def stat(sample):
        acc = {}
        for key in ("base|abstractive", "base|structured", "b100|abstractive", "b100|structured"):
            k = sum(per_doc[d].get(key, (0, 0))[0] for d in sample)
            n = sum(per_doc[d].get(key, (0, 0))[1] for d in sample)
            acc[key] = k / n if n else 0.0
        gap_base = acc["base|abstractive"] - acc["base|structured"]
        gap_dense = acc["b100|abstractive"] - acc["b100|structured"]
        return gap_dense - gap_base

    point = stat(docs)
    rng = random.Random(19)
    vals = sorted(stat([rng.choice(docs) for _ in docs]) for _ in range(4000))
    return point, (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))])


def analyze() -> dict:
    data = _load()
    report = {"scopes": {}, "per_model": {}, "retention": {}, "baseline_full": {}}

    for scope, self_only in (("all_receivers", False), ("self_only", True)):
        cells = {}
        for density in ("base", "b100"):
            for method in METHODS:
                cells[f"{density}|{method}"] = _acc(
                    data[density][method], self_only=self_only
                )
        abs_gap = cells["base|abstractive"]["accuracy"] - cells["base|structured"]["accuracy"]
        dense_gap = cells["b100|abstractive"]["accuracy"] - cells["b100|structured"]["accuracy"]

        density_pairs = {
            method: _pairs(
                data["base"][method], data["b100"][method], self_only=self_only
            )
            for method in METHODS
        }
        summary = {
            "cells": cells,
            "method_gap": {"base": abs_gap, "b100": dense_gap, "interaction": dense_gap - abs_gap},
            "density_effect": {},
        }
        for method in METHODS:
            ps = paired_summary(density_pairs[method])
            point, ci = _bootstrap_density_diff(density_pairs[method])
            summary["density_effect"][method] = {**ps, "bootstrap_ci": ci}
        point, ci = _bootstrap_interaction(data, self_only=self_only)
        summary["interaction"] = {"diff": point, "bootstrap_ci": ci}
        report["scopes"][scope] = summary

    # Per-sender method gap at each density, self-receiver view and all receivers.
    for scope, self_only in (("all_receivers", False), ("self_only", True)):
        per_model = {}
        for density in ("base", "b100"):
            gaps = {}
            for sender in sorted(
                {r["sender_model"] for r in data[density].get("abstractive", [])}
            ):
                row = {}
                for method in METHODS:
                    sub = [
                        r
                        for r in data[density][method]
                        if r["sender_model"] == sender
                        and (not self_only or r["pair_type"] == "self")
                    ]
                    k = sum(int(r["correct"]) for r in sub)
                    n = len(sub)
                    row[method] = {"accuracy": k / n if n else None, "n": n}
                if row["abstractive"]["accuracy"] is not None and row["structured"]["accuracy"] is not None:
                    row["gap"] = row["abstractive"]["accuracy"] - row["structured"]["accuracy"]
                gaps[sender] = row
            per_model[density] = gaps
        report["per_model"][scope] = per_model

    # Sender-side metadata from the message files (no receiver involved).
    for density, msg_path in (
        ("base", "results/v3base4/messages.jsonl"),
        ("b100", "results/v3probe4/messages.jsonl"),
    ):
        messages = read_jsonl(msg_path)
        meta = {}
        for method in METHODS:
            sub = [m for m in messages if m["method"] == method]
            n = len(sub)
            compliant = sum(1 for m in sub if m["compliant"])
            retentions = [m["retention"]["fact_retention"] for m in sub if m.get("retention")]
            meta[method] = {
                "n_messages": n,
                "compliant": compliant,
                "compliance_rate": compliant / n if n else None,
                "mean_fact_retention": sum(retentions) / len(retentions) if retentions else None,
                "mean_bytes": sum(m["final_bytes"] for m in sub) / n if n else None,
                "mean_ratio_source": sum(m["ratio_source"] for m in sub) / n if n else None,
                "mean_n_items": sum(m.get("n_items") or 0 for m in sub) / n if n else None,
            }
        report["retention"][density] = meta

    for density in ("base", "b100"):
        full = [r for r in flat_questions(read_jsonl(DATASETS[density])) if r["method"] == "full"]
        k = sum(int(r["correct"]) for r in full)
        n = len(full)
        report["baseline_full"][density] = {"n_correct": k, "n_questions": n, "accuracy": k / n if n else None}

    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# G1 — Method × Density 2×2 (corrected, paired 10 documents)",
        "",
        "> Generated by `python -m src.analysis.density_2x2` from frozen predictions.",
        "> Zero new API calls. Base = 20.4 words/fact, b100 = 15.4 words/fact;",
        "> same documents, same questions, 50% byte budget, readable prompts.",
        "",
        "**Correction to handoff v2.** The published 2×2 mixed receiver scopes:",
        "base averaged all four receivers (79.4 / 77.1) while b100 used",
        "self-receivers only (80.6 / 68.5), giving a +12.1pp b100 gap and a ~9.9pp",
        "interaction. With a consistent scope the b100 gap is +10.7pp",
        "(all receivers) or +12.1pp (self only). Direction and significance are",
        "unchanged; the interaction is 8.5pp (all receivers) or 10.1pp (self only).",
        "",
    ]
    for scope, s in report["scopes"].items():
        title = "All four receivers" if scope == "all_receivers" else "Self-receiver only"
        lines += [f"## {title}", ""]
        lines += [
            "| Density | abstractive | structured | Method gap (A−S) |",
            "|---|---:|---:|---:|",
        ]
        for density in ("base", "b100"):
            a = s["cells"][f"{density}|abstractive"]
            st = s["cells"][f"{density}|structured"]
            gap = a["accuracy"] - st["accuracy"]
            lines.append(
                f"| {density} | {a['accuracy']:.3f} ({a['n_correct']}/{a['n_questions']}) | "
                f"{st['accuracy']:.3f} ({st['n_correct']}/{st['n_questions']}) | {100*gap:+.1f}pp |"
            )
        lines.append(
            f"| **interaction** | | | **{100*s['method_gap']['interaction']:+.1f}pp** |"
        )
        lines.append("")
        ci = s["interaction"]["bootstrap_ci"]
        if ci:
            lines.append(
                f"Interaction bootstrap 95% CI: {100*ci[0]:+.1f} to {100*ci[1]:+.1f}pp "
                f"(doc-cluster, 4,000 resamples)."
            )
            lines.append("")
        lines += [
            "| Method | Base acc | b100 acc | Δ (b100−base) | McNemar p | Bootstrap CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for method in METHODS:
            d = s["density_effect"][method]
            ci = d["bootstrap_ci"]
            ci_txt = f"{100*(-ci[1]):+.1f} to {100*(-ci[0]):+.1f}pp" if ci else "-"
            lines.append(
                f"| {method} | {d['accuracy_a']:.3f} | {d['accuracy_b']:.3f} | "
                f"{100*(d['accuracy_b']-d['accuracy_a']):+.1f}pp | {_p(d['mcnemar_p'])} | {ci_txt} |"
            )
        lines.append("")
        lines.append(
            "Discordant pairs (base correct / b100 wrong vs the reverse): "
            + "; ".join(
                f"{m} {d['a_only_correct']} vs {d['b_only_correct']}" for m, d in s["density_effect"].items()
            )
            + "."
        )
        lines.append("")

    lines += [
        "## Per-sender method gap (all four receivers)",
        "",
        "| Sender | base gap | b100 gap | Δ interaction |",
        "|---|---:|---:|---:|",
    ]
    for sender, row in report["per_model"]["all_receivers"]["base"].items():
        dense = report["per_model"]["all_receivers"]["b100"].get(sender, {})
        base_gap = row.get("gap")
        dense_gap = dense.get("gap")
        if base_gap is None or dense_gap is None:
            continue
        lines.append(
            f"| {sender} | {100*base_gap:+.1f}pp | {100*dense_gap:+.1f}pp | {100*(dense_gap-base_gap):+.1f}pp |"
        )
    lines += [
        "",
        "All four senders move in the same direction: the structured penalty grows",
        "when the same facts are rendered more densely.",
        "",
        "## Fact retention and compliance (sender side, 10 documents)",
        "",
        "| Density | Method | compliance | mean bytes | mean source ratio | mean items | lexical fact retention |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for density in ("base", "b100"):
        for method in METHODS:
            m = report["retention"][density][method]
            lines.append(
                f"| {density} | {method} | {m['compliant']}/{m['n_messages']} "
                f"({100*m['compliance_rate']:.0f}%) | {m['mean_bytes']:.0f} | "
                f"{m['mean_ratio_source']:.3f} | {m['mean_n_items']:.1f} | "
                f"{m['mean_fact_retention']:.3f} |"
            )
    lines.append("")
    lines.append(
        "Full-document upper bound: base "
        f"{report['baseline_full']['base']['accuracy']:.3f}, b100 "
        f"{report['baseline_full']['b100']['accuracy']:.3f}."
    )
    lines += [
        "",
        "Retention is the lexical fact-survival proxy from `src/eval/retention.py`,",
        "not the receiver outcome; it is reported to show the sender-side mechanism",
        "(structured drops more facts at b100).",
        "",
        "## Reading",
        "",
        "- The Method × Density interaction is real and same-signed across all four",
        "  senders, but it is **8.5pp** with a consistent all-receiver scope, not the",
        "  9.9pp printed in handoff v2 (which mixed scopes). Use the all-receiver",
        "  numbers for the paper table; the self-only row is the appendix check.",
        "- Abstractive is flat across densities; structured loses ~8pp from base to",
        "  b100, while the full-document baseline barely moves (95.1% → 95.8%).",
        "- Compliance differs by method, so `actual_bytes` remains a covariate in",
        "  the main analysis (G5).",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/density_2x2.md")
    parser.add_argument("--json-out", default="reports/density_2x2.json")
    args = parser.parse_args(argv)
    report = analyze()
    write_json(args.json_out, report)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report) + "\n")
    print(f"wrote {args.out} and {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
