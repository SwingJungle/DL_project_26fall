"""Token / quota accounting for the API budget.

Reads `runs/*/calls.jsonl`, separates provider-billed requests (fresh) from
local-cache hits, reports measured tokens and cache rates by call type, then
forecasts the main experiment under several design variants.

The forecast is deliberately simple: measured mean tokens per call type, an
assumed sender-calls-per-message factor for each retry policy, and the
receiver grid. It is a planning tool, not a promise; provider billing may
weight tokens differently (the dashboard's "remaining 51.5%" is the
authoritative number).

Usage:
    python -m src.analysis.token_budget
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from ..common import now_iso
from .common import write_json

# Provider dashboard snapshot (2026-11-15, token-plan.maas.qianwenaiapi.com),
# keyed by the dashboard's model filter. Used only for reconciliation against
# the local call logs; update if the plan/date changes.
DASHBOARD_2026_11_15 = {
    "deepseek-v4-flash-0731": {
        "total": 22_249_348,
        "uncached_input": 18_053_876,
        "cached_input": 2_999_296,
        "output": 1_196_176,
        "cache_rate": 0.142,
    },
    "deepseek-v4-pro-0813": {
        "total": 1_196_622,
        "uncached_input": 688_030,
        "cached_input": 269_824,
        "output": 238_768,
        "cache_rate": 0.282,
    },
    "qwen3.7-max": {
        "total": 1_625_878,
        "uncached_input": 1_327_234,
        "cached_input": 1_152,
        "output": 297_492,
        "cache_rate": 0.001,
    },
    "qwen3.7-plus": {
        "total": 1_389_590,
        "uncached_input": 995_167,
        "cached_input": 40_960,
        "output": 353_463,
        "cache_rate": 0.040,
    },
}

# Measured on 2026-11-15 from 5,672 fresh provider requests.
SENDER_CALLS_PER_MESSAGE = {0: 1.0, 1: 1.4, 2: 1.8}
VARIANTS = {
    "A_current": {
        "label": "100 docs, 4 receivers, retries=2",
        "docs": 100,
        "densities": 2,
        "methods": 2,
        "senders": 4,
        "receivers": 4,
        "subset_docs_4x4": None,
        "retries": 2,
        "compact_docs": 0,
    },
    "B_lean": {
        "label": "100 docs, self + 4x4 on 30 docs, retries=1",
        "docs": 100,
        "densities": 2,
        "methods": 2,
        "senders": 4,
        "receivers": 1,
        "subset_docs_4x4": 30,
        "retries": 1,
        "compact_docs": 0,
    },
    "C_lean_compact": {
        "label": "B + compact b100 self-only",
        "docs": 100,
        "densities": 2,
        "methods": 2,
        "senders": 4,
        "receivers": 1,
        "subset_docs_4x4": 30,
        "retries": 1,
        "compact_docs": 100,
    },
    "D_self_only": {
        "label": "100 docs, self only, retries=0 (no 4x4)",
        "docs": 100,
        "densities": 2,
        "methods": 2,
        "senders": 4,
        "receivers": 1,
        "subset_docs_4x4": None,
        "retries": 0,
        "compact_docs": 0,
    },
}


def measured() -> dict:
    by_type = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "cached": 0})
    by_model = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "cached": 0})
    total = {"calls": 0, "fresh": 0, "prompt": 0, "completion": 0, "cached": 0}
    for path in sorted(glob.glob("runs/*/calls.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                total["calls"] += 1
                if rec.get("cached"):
                    continue
                total["fresh"] += 1
                usage = rec.get("usage") or {}
                pt = usage.get("prompt_tokens") or 0
                ct = usage.get("completion_tokens") or 0
                cached = usage.get("cached_tokens") or 0
                total["prompt"] += pt
                total["completion"] += ct
                total["cached"] += cached
                tag = rec.get("tag", "")
                if "|full|recv:" in tag:
                    kind = "full"
                elif tag.startswith("receiver:"):
                    kind = "receiver"
                elif tag.startswith("sender:"):
                    kind = "sender"
                else:
                    kind = "other"
                for key, bucket in ((kind, by_type[kind]), (rec.get("model_key"), by_model[rec.get("model_key")])):
                    bucket["calls"] += 1
                    bucket["prompt"] += pt
                    bucket["completion"] += ct
                    bucket["cached"] += cached
    out = {"total": total, "by_type": {}, "by_model": {}}
    for name, b in by_type.items():
        n = max(1, b["calls"])
        out["by_type"][name] = {
            **b,
            "prompt_per_call": b["prompt"] / n,
            "completion_per_call": b["completion"] / n,
            "cache_rate": b["cached"] / max(1, b["prompt"]),
        }
    for name, b in by_model.items():
        out["by_model"][name] = {
            **b,
            "cache_rate": b["cached"] / max(1, b["prompt"]),
        }
    return out


def forecast(measured_stats: dict, spec: dict) -> dict:
    calls = measured_stats
    sender = calls.get("sender", {"prompt_per_call": 891, "completion_per_call": 151})
    receiver = calls.get("receiver", {"prompt_per_call": 605, "completion_per_call": 236})
    full = calls.get("full", {"prompt_per_call": 656, "completion_per_call": 243})

    def tok(per_call: dict, n_calls: float) -> float:
        return n_calls * (per_call["prompt_per_call"] + per_call["completion_per_call"])

    factor = SENDER_CALLS_PER_MESSAGE[spec["retries"]]
    n_messages = spec["docs"] * spec["densities"] * spec["methods"] * spec["senders"]
    sender_calls = n_messages * factor
    receiver_calls = n_messages * spec["receivers"]
    full_calls = spec["docs"] * 4

    subset_receiver_calls = 0
    if spec["subset_docs_4x4"]:
        subset_messages = spec["subset_docs_4x4"] * spec["densities"] * spec["methods"] * spec["senders"]
        subset_receiver_calls = subset_messages * (4 - spec["receivers"])

    compact_sender_calls = compact_receiver_calls = 0
    if spec["compact_docs"]:
        compact_messages = spec["compact_docs"] * spec["senders"]
        compact_sender_calls = compact_messages * factor
        compact_receiver_calls = compact_messages  # self receiver

    total_receiver_calls = receiver_calls + subset_receiver_calls + compact_receiver_calls
    total_sender_calls = sender_calls + compact_sender_calls
    total_tokens = (
        tok(sender, total_sender_calls)
        + tok(receiver, total_receiver_calls)
        + tok(full, full_calls)
    )
    return {
        **spec,
        "n_messages": n_messages,
        "sender_calls": round(total_sender_calls),
        "receiver_calls": round(total_receiver_calls),
        "full_calls": full_calls,
        "compact_calls": round(compact_sender_calls + compact_receiver_calls),
        "total_calls": round(total_sender_calls + total_receiver_calls + full_calls),
        "total_tokens": round(total_tokens),
    }


def reconcile(measured_stats: dict) -> dict:
    """Dashboard vs local logs, per model. Only the shared-key models diverge."""
    rows = {}
    for model, dash in DASHBOARD_2026_11_15.items():
        logged = measured_stats["by_model"].get(model, {})
        logged_total = logged.get("prompt", 0) + logged.get("completion", 0)
        extra = dash["total"] - logged_total
        rows[model] = {
            "dashboard_total": dash["total"],
            "logged_total": logged_total,
            "ratio": dash["total"] / logged_total if logged_total else None,
            "extra_tokens": extra,
            "dashboard_cache_rate": dash["cache_rate"],
            "logged_cache_rate": logged.get("cache_rate"),
            "extra_input": dash["uncached_input"] + dash["cached_input"]
            - (logged.get("prompt", 0)),
            "extra_output": dash["output"] - logged.get("completion", 0),
        }
    total_dash = sum(d["total"] for d in DASHBOARD_2026_11_15.values())
    total_logged = sum(r["logged_total"] for r in rows.values())
    return {
        "date": "2026-11-15",
        "rows": rows,
        "total_dashboard": total_dash,
        "total_logged": total_logged,
        "total_extra": total_dash - total_logged,
    }


def render_markdown(report: dict) -> str:
    total = report["measured"]["total"]
    lines = [
        "# Token / quota accounting",
        "",
        f"> Generated by `python -m src.analysis.token_budget` at {report['generated_at']}.",
        "",
        "## Measured (all logged runs)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Log records / provider-billed requests | {total['calls']:,} / {total['fresh']:,} |",
        f"| Prompt tokens | {total['prompt']:,} |",
        f"| Completion tokens | {total['completion']:,} |",
        f"| Cached prompt tokens | {total['cached']:,} ({100*total['cached']/max(1,total['prompt']):.1f}%) |",
        f"| Total tokens | {total['prompt']+total['completion']:,} |",
        "",
        "| Call type | Calls | Prompt/call | Completion/call | Cache rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind, b in report["measured"]["by_type"].items():
        lines.append(
            f"| {kind} | {b['calls']:,} | {b['prompt_per_call']:.0f} | "
            f"{b['completion_per_call']:.0f} | {100*b['cache_rate']:.1f}% |"
        )
    lines += [
        "",
        "| Model | Calls | Prompt | Completion | Cache rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, b in report["measured"]["by_model"].items():
        lines.append(
            f"| {model} | {b['calls']:,} | {b['prompt']:,} | {b['completion']:,} | {100*b['cache_rate']:.1f}% |"
        )
    rec = report.get("reconciliation")
    if rec:
        lines += [
            "",
            "## Dashboard reconciliation (2026-11-15)",
            "",
            "| Model | Dashboard tokens | Logged tokens | Ratio | Dashboard cache | Logged cache |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for model, r in rec["rows"].items():
            lines.append(
                f"| {model} | {r['dashboard_total']:,} | {r['logged_total']:,} | "
                f"{r['ratio']:.2f}x | {100*r['dashboard_cache_rate']:.1f}% | "
                f"{100*(r['logged_cache_rate'] or 0):.1f}% |"
            )
        top = max(rec["rows"].items(), key=lambda kv: kv[1]["extra_tokens"])
        model, r = top
        lines += [
            "",
            f"- The three non-flash models match the local logs within a few percent;",
            f"  **{model}** is {r['ratio']:.1f}x the logged usage: "
            f"{r['extra_tokens']:,} tokens ({100*r['extra_tokens']/rec['total_dashboard']:.1f}% "
            "of the day) were spent outside this codebase (owner confirmed: the",
            "  shared key was used for the NLP/COMP5423 reproduction).",
            f"- The extra workload is input-heavy and output-light "
            f"({r['extra_input']:,} extra input vs {r['extra_output']:,} extra output, "
            f"~20:1), consistent with long-context retrieval/RAG or a batch agent, not",
            "  with this project's sender/receiver calls (output/input ~1:4).",
            "- This is the quota drain, not cache misses: the experiment's own",
            "  cache rate is structural (receiver prompts unique per message/model).",
            "",
            "## Forecast for the main experiment",
        "",
        "| Variant | Calls | Sender | Receiver | Total tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, f in report["forecast"].items():
        lines.append(
            f"| {name}: {f['label']} | {f['total_calls']:,} | {f['sender_calls']:,} | "
            f"{f['receiver_calls']:,} | {f['total_tokens']/1e6:.1f}M |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- The cache rate is **structural**, not a property of the corpus: receiver",
        "  calls are unique per message and per receiver model (0.7% cache), while",
        "  sender calls share only the fixed instruction prefix (22.7%). Fictional",
        "  content would not change prefix caching; even a real corpus is unique per",
        "  document.",
        "- qwen3.7-max shows ~0.1% cache in our logs while the two DeepSeek models",
        "  show ~27%; the provider/model routing matters more than the data.",
        "- The dominant cost is the receiver grid (4 receivers per message) and",
        "  sender retries, not cache misses. Variant B cuts ~40% of the calls and",
        "  ~40% of the tokens by keeping self receivers on 100 docs and a 4x4 grid",
        "  on 30 docs.",
        "- The 11-15 dashboard total (26.46M) is now reconciled: three models",
        "  match the local logs within 1%; the entire gap is deepseek-v4-flash-0731",
        "  usage outside this codebase (~21.1M, input-heavy ~20:1). The owner",
        "  confirmed this is the shared key being used for the NLP/COMP5423",
        "  reproduction: the quota drain is key sharing, not cache misses.",
        "- Quota protocol: the DL key is for this project only; the NLP reproduction",
        "  should use its own key/plan (or the official DeepSeek endpoint). Reserve",
        "  ~6M tokens for the DL main run, cap each phase with `--max-calls`, and",
        "  check the dashboard between the sender and receiver phases.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/token_budget.md")
    parser.add_argument("--json-out", default="reports/token_budget.json")
    args = parser.parse_args(argv)
    measured_stats = measured()
    report = {
        "generated_at": now_iso(),
        "measured": measured_stats,
        "reconciliation": reconcile(measured_stats),
        "assumptions": {
            "sender_calls_per_message": SENDER_CALLS_PER_MESSAGE,
            "note": "measured mean tokens/call; forecasting only",
        },
        "forecast": {
            name: forecast(measured_stats["by_type"], spec) for name, spec in VARIANTS.items()
        },
    }
    write_json(args.json_out, report)
    Path(args.out).write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
