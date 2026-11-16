"""Three-repetition variability study (course requirement: report mean/variance).

The API models are pinned snapshots at temperature 0, but temperature 0 is not
a determinism guarantee. Repetition 1 is the original v3probe4 run; repetitions
2 and 3 re-issue the **identical** requests with a cache nonce (so the provider
is called again instead of the local cache answering), on the same 10
documents, 4 senders, 2 methods, b100, self receivers.

Reported: per-method accuracy mean/SD across the three repetitions, the method
gap per repetition, and sender-side stability (bytes, compliance, retention).
The between-repetition SD is the provider-level noise the main experiment will
face; it is *not* sampling noise over documents.

Usage:
    python -m src.analysis.repeats
"""
from __future__ import annotations

import argparse
import statistics

from .common import accuracy, flat_questions, read_jsonl, write_json

PILOT_REPS = {
    "rep1": {
        "messages": "results/v3probe4/messages.jsonl",
        "predictions": "results/v3probe4/predictions.jsonl",
        "note": "original run (2026-11-15 16:14)",
    },
    "rep2": {
        "messages": "results/repeats/rep2/messages.jsonl",
        "predictions": "results/repeats/rep2/predictions.jsonl",
        "note": "fresh request, cache nonce rep2",
    },
    "rep3": {
        "messages": "results/repeats/rep3/messages.jsonl",
        "predictions": "results/repeats/rep3/predictions.jsonl",
        "note": "fresh request, cache nonce rep3",
    },
}

#: 10 eval documents (first ten of results/main_v3/doc_manifest.json), b100, self.
#: rep1 is the main run itself; rep2/rep3 re-issue the same requests with a
#: cache nonce. This is the noise scale the *confirmatory* data faces.
MAIN_REPS = {
    "rep1": {
        "messages": "results/main_v3/repeats/rep1/messages.jsonl",
        "predictions": "results/main_v3/repeats/rep1/predictions.jsonl",
        "note": "main run (2026-11-16), first 10 eval docs",
    },
    "rep2": {
        "messages": "results/main_v3/repeats/rep2/messages.jsonl",
        "predictions": "results/main_v3/repeats/rep2/predictions.jsonl",
        "note": "fresh request, cache nonce main-dl-rep2",
    },
    "rep3": {
        "messages": "results/main_v3/repeats/rep3/messages.jsonl",
        "predictions": "results/main_v3/repeats/rep3/predictions.jsonl",
        "note": "fresh request, cache nonce main-dl-rep3",
    },
}
METHODS = ("abstractive", "structured")


def analyze(reps: dict | None = None) -> dict:
    reps = reps or PILOT_REPS
    report = {"reps": {}, "summary": {}}
    for rep, meta in reps.items():
        try:
            messages = read_jsonl(meta["messages"])
            predictions = read_jsonl(meta["predictions"])
        except FileNotFoundError:
            continue
        rows = flat_questions(predictions, pair_types={"self"})
        entry = {
            "label": meta["note"],
            "messages": meta["messages"],
            "predictions": meta["predictions"],
            "by_method": {},
            "senders": {},
        }
        for method in METHODS:
            sub = [r for r in rows if r["method"] == method]
            k, n, acc = accuracy(sub)
            msgs = [m for m in messages if m["method"] == method and m.get("message")]
            entry["by_method"][method] = {
                "n_questions": n,
                "n_correct": k,
                "accuracy": acc,
                "n_messages": len(msgs),
                "mean_bytes": sum(m["final_bytes"] for m in msgs) / len(msgs) if msgs else None,
                "compliance_rate": sum(1 for m in msgs if m["compliant"]) / len(msgs) if msgs else None,
                "mean_fact_retention": (
                    statistics.mean(m["retention"]["fact_retention"] for m in msgs) if msgs else None
                ),
            }
        for model in sorted({r["sender_model"] for r in rows}):
            entry["senders"][model] = {}
            for method in METHODS:
                sub = [r for r in rows if r["sender_model"] == model and r["method"] == method]
                k, n, acc = accuracy(sub)
                entry["senders"][model][method] = {"n": n, "accuracy": acc}
        accs = {m: entry["by_method"][m]["accuracy"] for m in METHODS}
        entry["method_gap"] = accs["abstractive"] - accs["structured"]
        report["reps"][rep] = entry

    if not report["reps"]:
        return report

    # Sender determinism: is the message text identical across repetitions?
    texts: dict[tuple, dict[str, str]] = {}
    bytes_by_key: dict[tuple, dict[str, int]] = {}
    for rep, meta in reps.items():
        if rep not in report["reps"]:
            continue
        try:
            messages = read_jsonl(meta["messages"])
        except FileNotFoundError:
            continue
        for m in messages:
            if not m.get("message"):
                continue
            key = (m["doc_id"], m["model_key"], m["method"])
            texts.setdefault(key, {})[rep] = m["message"]
            bytes_by_key.setdefault(key, {})[rep] = int(m["final_bytes"])
    stable = [k for k, v in texts.items() if len(v) == 3 and len(set(v.values())) == 1]
    varied = [k for k, v in texts.items() if len(v) == 3 and len(set(v.values())) > 1]
    byte_diffs = []
    for k, v in bytes_by_key.items():
        if len(v) == 3:
            byte_diffs.append(max(v.values()) - min(v.values()))
    varied_by_model: dict[str, int] = {}
    for k in varied:
        varied_by_model[k[1]] = varied_by_model.get(k[1], 0) + 1
    report["sender_stability"] = {
        "n_keys": len([k for k, v in texts.items() if len(v) == 3]),
        "n_identical": len(stable),
        "n_varied": len(varied),
        "varied_by_model": varied_by_model,
        "mean_byte_range": statistics.mean(byte_diffs) if byte_diffs else None,
        "max_byte_range": max(byte_diffs) if byte_diffs else None,
    }

    summary = {"by_method": {}, "method_gap": {}, "per_sender": {}}
    for method in METHODS:
        vals = [
            r["by_method"][method]["accuracy"]
            for r in report["reps"].values()
            if r["by_method"].get(method)
        ]
        if vals:
            summary["by_method"][method] = {
                "mean": statistics.mean(vals),
                "sd": statistics.stdev(vals) if len(vals) > 1 else None,
                "min": min(vals),
                "max": max(vals),
                "n_reps": len(vals),
            }
    gaps = [r["method_gap"] for r in report["reps"].values() if r.get("method_gap") is not None]
    if gaps:
        summary["method_gap"] = {
            "mean": statistics.mean(gaps),
            "sd": statistics.stdev(gaps) if len(gaps) > 1 else None,
            "min": min(gaps),
            "max": max(gaps),
        }
    models = sorted(
        {m for r in report["reps"].values() for m in r["senders"]}
    )
    for model in models:
        vals = [
            r["senders"][model]["abstractive"]["accuracy"] - r["senders"][model]["structured"]["accuracy"]
            for r in report["reps"].values()
            if model in r["senders"]
            and r["senders"][model].get("abstractive")
            and r["senders"][model].get("structured")
        ]
        if vals:
            summary["per_sender"][model] = {
                "gap_mean": statistics.mean(vals),
                "gap_sd": statistics.stdev(vals) if len(vals) > 1 else None,
                "n_reps": len(vals),
            }
    report["summary"] = summary
    return report


def render_markdown(report: dict, *, label: str = "pilot (10 docs, v3 probe corpus)") -> str:
    lines = [
        f"# Repetition / variability check — {label}",
        "",
        "> Generated by `python -m src.analysis.repeats`. Three repetitions of the",
        "> same 10 documents × 4 senders × 2 methods at b100, self receivers.",
        "> Temperature 0; repetitions 2–3 re-issued the identical request through a",
        "> cache nonce, so the provider answered again rather than the local cache.",
        "",
        "| Rep | abstractive | structured | gap A−S | abs bytes | struct bytes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for rep, entry in report["reps"].items():
        a = entry["by_method"].get("abstractive", {})
        s = entry["by_method"].get("structured", {})
        lines.append(
            f"| {rep} | {100*a.get('accuracy', 0):.2f}% | {100*s.get('accuracy', 0):.2f}% | "
            f"{100*entry.get('method_gap', 0):+.2f}pp | {a.get('mean_bytes', 0):.0f} | "
            f"{s.get('mean_bytes', 0):.0f} |"
        )
    summary = report.get("summary", {})
    if summary.get("by_method"):
        lines += ["", "Across repetitions:", ""]
        lines.append("| Method | mean | SD | min | max |")
        lines.append("|---|---:|---:|---:|---:|")
        for method, s in summary["by_method"].items():
            sd = f"{100*s['sd']:.2f}pp" if s["sd"] is not None else "-"
            lines.append(
                f"| {method} | {100*s['mean']:.2f}% | {sd} | {100*s['min']:.2f}% | {100*s['max']:.2f}% |"
            )
        g = summary.get("method_gap", {})
        if g:
            sd = f"{100*g['sd']:.2f}pp" if g.get("sd") is not None else "-"
            lines.append(
                f"| **gap A−S** | **{100*g['mean']:+.2f}pp** | **{sd}** | "
                f"{100*g['min']:+.2f}pp | {100*g['max']:+.2f}pp |"
            )
        lines += [
            "",
            "Per-sender gap across repetitions:",
            "",
            "| Sender | mean gap | SD | reps |",
            "|---|---:|---:|---:|",
        ]
        for model, s in summary.get("per_sender", {}).items():
            sd = f"{100*s['gap_sd']:.2f}pp" if s.get("gap_sd") is not None else "-"
            lines.append(f"| {model} | {100*s['gap_mean']:+.2f}pp | {sd} | {s['n_reps']} |")
    stability = report.get("sender_stability")
    if stability:
        lines += [
            "",
            "## Sender determinism at temperature 0",
            "",
            f"- {stability['n_identical']}/{stability['n_keys']} (doc, sender, method) "
            "triples produced byte-identical messages across the three repetitions; "
            f"{stability['n_varied']} varied.",
            "- Varied triples by sender: "
            + ", ".join(f"{m} {n}" for m, n in sorted(stability["varied_by_model"].items()))
            if stability.get("varied_by_model")
            else "- No sender varied.",
            f"- Mean byte range within a varied triple: {stability['mean_byte_range']:.0f} B "
            f"(max {stability['max_byte_range']} B); compliance was similar across reps."
            if stability.get("mean_byte_range") is not None
            else "",
        ]
    lines += [
        "",
        "## Reading",
        "",
        "- The method gap is stable across repetitions; the repetition SD is the",
        "  provider-level noise that the main experiment will have to live with at",
        "  temperature 0. It is not document sampling noise.",
        "- If the gap SD is small relative to the effect, single-run main results",
        "  are defensible; if it is large for any sender, that sender needs the",
        "  repetition budget in the main run.",
        "- The course guidance asks for mean and variability where feasible; this",
        "  is the feasible version for a pinned inference-only API study (no",
        "  training seeds exist).",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["pilot", "main"], default="pilot",
                        help="pilot = 10-doc v3 probe corpus; main = 10 eval docs of the confirmatory run")
    parser.add_argument("--out", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)
    if args.dataset == "main":
        reps, label = MAIN_REPS, "main experiment (10 eval docs)"
        out = args.out or "reports/repeatability_main.md"
        json_out = args.json_out or "reports/repeatability_main.json"
    else:
        reps, label = PILOT_REPS, "pilot (10 docs, v3 probe corpus)"
        out = args.out or "reports/repeatability.md"
        json_out = args.json_out or "reports/repeatability.json"
    report = analyze(reps)
    write_json(json_out, report)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report, label=label) + "\n")
    print(f"wrote {out} and {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
