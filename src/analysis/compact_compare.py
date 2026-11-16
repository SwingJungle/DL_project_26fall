"""H3: compact schema vs full JSON at b100 (pre-registered small improvement).

The main experiment runs ``compact`` at b100 with a self receiver only
(``docs/freeze_v3_plan.md`` §8 decision 2). H3 compares it against the **same
documents, senders and questions** in the full-JSON (structured) condition, so
pairing is on (doc, sender, receiver, qid).

Reported:
* accuracy per representation (compact / structured / abstractive) with n;
* compact - structured paired difference, document-cluster bootstrap CI and
  exact McNemar;
* budget compliance and realized bytes per representation (D-5 context).

``--single-file`` mode consumes one predictions file that already holds every
representation (used to validate the script on ``results/compact_probe/``),
otherwise two files are paired.

Usage
-----
    python -m src.analysis.compact_compare --pilot     # validate, expect +4.8pp
    python -m src.analysis.compact_compare             # main experiment
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .common import (
    accuracy,
    bootstrap_paired,
    flat_questions,
    mcnemar_exact,
    paired_questions,
    paired_summary,
    read_jsonl,
    write_json,
)

METHODS = ("abstractive", "structured", "compact")
PILOT = "results/compact_probe/predictions.jsonl"
PILOT_MESSAGES = ["results/compact_probe/messages.jsonl"]
MAIN_COMPACT = "results/main_v3/predictions_b100_compact.jsonl"
MAIN_REFERENCE = "results/main_v3/predictions_b100.jsonl"
MAIN_MESSAGES = ["results/main_v3/messages_b100_compact.jsonl",
                 "results/main_v3/messages_b100.jsonl"]


def _rows(paths: list[str], *, keep_repr: set[str] | None = None) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        for rec in read_jsonl(path):
            if rec.get("method") not in METHODS:
                continue
            if keep_repr and rec["method"] not in keep_repr:
                continue
            rows.extend(flat_questions([rec]))
    return rows


def _self_only(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("pair_type") == "self"]


def _retention(message_paths: list[str]) -> dict:
    """Mean lexical fact retention per representation, from the sender messages."""
    out: dict[str, list[float]] = defaultdict(list)
    for path in message_paths:
        for rec in read_jsonl(path):
            method = rec.get("method")
            if method not in METHODS:
                continue
            value = (rec.get("retention") or {}).get("fact_retention")
            if isinstance(value, (int, float)):
                out[method].append(float(value))
    return {m: (sum(v) / len(v) if v else None) for m, v in out.items()}


def _compliance(message_paths: list[str]) -> dict:
    """Compliance and realized bytes per representation, from *messages*.

    Prediction records do not carry ``compliant``; reading them there would
    silently report 0% for every method (a bug in the first version of this
    script, fixed before the numbers were used).
    """
    out: dict[str, dict] = defaultdict(lambda: {"n": 0, "compliant": 0, "bytes": 0})
    for path in message_paths:
        for rec in read_jsonl(path):
            method = rec.get("method")
            if method not in METHODS:
                continue
            b = out[method]
            b["n"] += 1
            b["compliant"] += int(bool(rec.get("compliant")))
            b["bytes"] += int(rec.get("final_bytes") or rec.get("message_bytes") or 0)
    for method, b in out.items():
        b["compliant_rate"] = b["compliant"] / b["n"] if b["n"] else None
        b["mean_bytes"] = b["bytes"] / b["n"] if b["n"] else None
    return dict(out)


def cell_ci(rows: list[dict], *, n_boot: int = 4000, seed: int = 11) -> dict:
    """Doc-cluster bootstrap CI for each representation's accuracy."""
    import random

    out = {}
    for method in METHODS:
        by_doc: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            if r["method"] == method:
                by_doc[r["doc_id"]].append(int(r["correct"]))
        docs = sorted(by_doc)
        if not docs:
            out[method] = None
            continue
        point = sum(sum(by_doc[d]) for d in docs) / sum(len(by_doc[d]) for d in docs)
        rng = random.Random(seed)
        vals = []
        for _ in range(n_boot):
            sample = [rng.choice(docs) for _ in docs]
            k = sum(sum(by_doc[d]) for d in sample)
            n = sum(len(by_doc[d]) for d in sample)
            vals.append(k / n if n else 0.0)
        vals.sort()
        out[method] = {"point": point,
                       "ci_lo": vals[int(0.025 * len(vals))],
                       "ci_hi": vals[int(0.975 * len(vals))]}
    return out


def analyze(
    rows: list[dict],
    *,
    compliance: dict,
    retention: dict | None = None,
    n_boot: int = 4000,
    seed: int = 11,
) -> dict:
    cells = {}
    for method in METHODS:
        k, n, acc = accuracy([r for r in rows if r["method"] == method])
        cells[method] = {"n_correct": k, "n_questions": n, "accuracy": acc}

    def pair(a: str, b: str) -> dict:
        rows_a = [r for r in rows if r["method"] == a]
        rows_b = [r for r in rows if r["method"] == b]
        pairs = paired_questions(rows_a, rows_b)
        summary = paired_summary(pairs)
        lo, hi = bootstrap_paired(pairs, n_boot=n_boot, seed=seed)
        return {**summary, "ci_lo": lo, "ci_hi": hi}

    comparisons = {
        "compact_vs_structured": pair("compact", "structured"),
        "compact_vs_abstractive": pair("compact", "abstractive"),
        "abstractive_vs_structured": pair("abstractive", "structured"),
    }
    return {
        "cells": cells,
        "cells_ci": cell_ci(rows, n_boot=n_boot, seed=seed),
        "comparisons": comparisons,
        "compliance": compliance,
        "retention": retention or {},
        "n_rows": len(rows),
        "n_docs": len({r["doc_id"] for r in rows}),
        "n_senders": len({r["sender_model"] for r in rows}),
        "bootstrap": {"n_boot": n_boot, "seed": seed},
    }


def _pct(v) -> str:
    return "-" if v is None else f"{100 * v:.2f}%"


def _pp(v) -> str:
    return "-" if v is None else f"{100 * v:+.1f}pp"


def _p(v) -> str:
    if v is None:
        return "-"
    return "<0.0001" if v < 1e-4 else f"{v:.4f}"


def render_markdown(rep: dict) -> str:
    c = rep["cells"]
    cmpact = rep["comparisons"]["compact_vs_structured"]
    lines = [
        "# H3：compact schema vs full JSON（b100，self receiver）",
        "",
        "> 脚本 `src/analysis/compact_compare.py`；判据见 `docs/freeze_v3_plan.md` §7 判据 3",
        "> （compact − full > 0 且 CI 不含 0，且收回 ≥ 50% 的 prose–full 差距）。",
        "",
        f"- 观测：{rep['n_rows']:,} 题级行 · {rep['n_docs']} 篇 · {rep['n_senders']} sender · self receiver",
        "",
        "## 准确率",
        "",
        "| 表征 | 准确率 | n | 平均字节 | ±2% 合规 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        cell = c[method]
        comp = rep["compliance"].get(method, {})
        lines.append(f"| {method} | {_pct(cell['accuracy'])} | {cell['n_questions']:,} | "
                     f"{comp.get('mean_bytes', 0):.0f} | {_pct(comp.get('compliant_rate'))} |")
    lines += [
        "",
        "## 配对检验",
        "",
        "| 对比 | n 对 | 差 | 95% CI（doc cluster） | McNemar p |",
        "|---|---:|---:|---|---:|",
    ]
    for key, label in (("compact_vs_structured", "compact − structured"),
                       ("compact_vs_abstractive", "compact − abstractive"),
                       ("abstractive_vs_structured", "abstractive − structured")):
        m = rep["comparisons"][key]
        lines.append(f"| {label} | {m['n_pairs']:,} | {_pp(m['diff_a_minus_b'])} | "
                     f"[{_pp(m['ci_lo'])}, {_pp(m['ci_hi'])}] | {_p(m['mcnemar_p'])} |")
    prose_gap = rep["comparisons"]["abstractive_vs_structured"]["diff_a_minus_b"]
    compact_gain = cmpact["diff_a_minus_b"]
    recovered = compact_gain / prose_gap if prose_gap else None
    lines += [
        "",
        f"- compact 收回 prose–full 差距：**{_pct(recovered)}**"
        f"（目标 ≥ 50%；pilot 为 40%）。",
        f"- 判据：差 > 0 且 CI 不含 0 → **{'通过' if (compact_gain or 0) > 0 and (cmpact['ci_lo'] or 0) > 0 else '未通过/待数据'}**。",
        "",
        "> 说明：主实验的 compact 只跑 b100 self，与 full 的 self 子集配对；",
        "> cross receiver 的稳健性未测（D-9 limitation）。",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--compact", default=MAIN_COMPACT)
    parser.add_argument("--reference", default=MAIN_REFERENCE,
                        help="predictions holding the full-JSON/abstractive self rows")
    parser.add_argument("--n-boot", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    if args.pilot:
        paths = [PILOT]
        rows = _self_only(_rows(paths))
        message_paths = PILOT_MESSAGES
    else:
        paths = [args.compact, args.reference]
        for p in paths:
            if not Path(p).exists():
                raise SystemExit(f"missing predictions: {p}")
        # the reference file carries structured/abstractive; the compact file
        # carries compact. Self-only on both sides keeps the comparison paired.
        rows = _self_only(_rows(paths))
        message_paths = MAIN_MESSAGES

    compliance = _compliance(message_paths)
    retention = _retention(message_paths)
    rep = analyze(rows, compliance=compliance, retention=retention,
                  n_boot=args.n_boot, seed=args.seed)
    out = args.out or ("reports/compact_compare_pilot.md" if args.pilot
                       else "reports/compact_compare.md")
    json_out = args.json_out or ("reports/compact_compare_pilot.json" if args.pilot
                                 else "reports/compact_compare.json")
    write_json(json_out, rep)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(render_markdown(rep) + "\n", encoding="utf-8")
    print(f"wrote {out} and {json_out}")
    c = rep["comparisons"]["compact_vs_structured"]
    print(f"rows={rep['n_rows']:,} compact={_pct(rep['cells']['compact']['accuracy'])} "
          f"structured={_pct(rep['cells']['structured']['accuracy'])} "
          f"diff={_pp(c['diff_a_minus_b'])} CI=[{_pp(c['ci_lo'])},{_pp(c['ci_hi'])}] "
          f"p={_p(c['mcnemar_p'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
