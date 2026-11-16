"""Figures for the report (matplotlib; run with the analysis venv).

Reads the frozen JSONL/JSON artefacts and writes PNG + PDF into
``reports/figures/``. Every figure is deterministic (fixed seeds where a
bootstrap is used).

    /tmp/dlvenv/bin/python -m src.analysis.figures

The rest of ``src.analysis`` is standard-library only; matplotlib is needed
only here, which is why this module is optional.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .common import accuracy, flat_questions, read_jsonl  # noqa: E402

BLUE, RED, GREEN, GREY = "#2b6cb0", "#c53030", "#2f855a", "#718096"
METHOD_COLOR = {"abstractive": BLUE, "structured": RED, "compact": GREEN}
METHOD_LABEL = {"abstractive": "abstractive (prose)", "structured": "structured (full JSON)", "compact": "structured (compact)"}
OUT = Path("reports/figures")


def _style():
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


def _cluster_ci(rows, *, n_boot=2000, seed=23):
    by_doc = defaultdict(lambda: [0, 0])
    for r in rows:
        by_doc[r["doc_id"]][0] += int(r["correct"])
        by_doc[r["doc_id"]][1] += 1
    docs = sorted(by_doc)
    if not docs:
        return (None, None)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        k = n = 0
        for _ in docs:
            d = rng.choice(docs)
            k += by_doc[d][0]
            n += by_doc[d][1]
        vals.append(k / n if n else 0.0)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def fig_method_density_2x2():
    data = {
        "base": [
            r
            for r in flat_questions(read_jsonl("results/v3base4/predictions.jsonl"))
            if r["method"] != "full"
        ],
        "b100": [
            r
            for r in flat_questions(read_jsonl("results/v3probe4/predictions_4x4.jsonl"))
            if r["method"] != "full"
        ],
    }
    densities = ["base", "b100"]
    # measured on this corpus; the old 20.4/15.4 labels were bpf/6.3, not word counts
    labels = ["base\n(21.2 words/fact)", "b100\n(16.1 words/fact)"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    for method in ("abstractive", "structured"):
        xs, ys, errs = [], [], []
        for i, density in enumerate(densities):
            rows = [r for r in data[density] if r["method"] == method]
            k, n, acc = accuracy(rows)
            lo, hi = _cluster_ci(rows)
            xs.append(i)
            ys.append(100 * acc)
            errs.append((100 * (acc - lo), 100 * (hi - acc)))
        ax.errorbar(
            xs,
            ys,
            yerr=np.array(errs).T,
            marker="o",
            ms=5,
            lw=1.4,
            capsize=2.5,
            color=METHOD_COLOR[method],
            label=METHOD_LABEL[method],
        )
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6.5, color=METHOD_COLOR[method])
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels)
    ax.set_ylabel("receiver accuracy (%)")
    ax.set_ylim(64, 86)
    ax.set_title("(a) method x source density, all four receivers")
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1]
    models = sorted({r["sender_model"] for r in data["base"]})
    width = 0.36
    for j, density in enumerate(densities):
        gaps = []
        for model in models:
            a = [r for r in data[density] if r["sender_model"] == model and r["method"] == "abstractive"]
            s = [r for r in data[density] if r["sender_model"] == model and r["method"] == "structured"]
            ka, na, acc_a = accuracy(a)
            ks, ns, acc_s = accuracy(s)
            gaps.append(100 * (acc_a - acc_s))
        ax.bar(np.arange(len(models)) + (j - 0.5) * width, gaps, width=width, color=GREY if j == 0 else RED, label=f"{density}", alpha=0.85)
    short = [m.replace("deepseek-v4-", "ds-").replace("qwen3.7-", "qwen-") for m in models]
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short, rotation=15, ha="right")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("method gap A−S (pp)")
    ax.set_title("(b) gap grows for every sender")
    ax.legend(frameon=False, title=None)
    fig.tight_layout()
    _save(fig, "fig1_method_density_2x2")


def _heatmap(ax, matrix, models, title, vmin, vmax):
    im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
    short = [m.replace("deepseek-v4-", "ds-").replace("qwen3.7-", "qwen-") for m in models]
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(short, rotation=35, ha="right")
    ax.set_yticklabels(short)
    ax.set_xlabel("receiver")
    ax.set_ylabel("sender")
    ax.set_title(title)
    for i in range(len(models)):
        for j in range(len(models)):
            val = matrix[i, j]
            ax.text(j, i, f"{100*val:.0f}", ha="center", va="center", fontsize=6.5, color="white" if val < (vmin + vmax) / 2 else "black")
    return im


def fig_sender_receiver_heatmap():
    predictions = read_jsonl("results/pilot_trans/predictions.jsonl")
    models = sorted({r["sender_model"] for r in predictions if r["method"] != "full"})
    rows = flat_questions(predictions)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    vmin, vmax = 0.66, 0.93
    for ax, method in zip(axes, ("abstractive", "structured")):
        matrix = np.zeros((4, 4))
        for i, s in enumerate(models):
            for j, r in enumerate(models):
                k, n, acc = accuracy(
                    [x for x in rows if x["method"] == method and x["sender_model"] == s and x["receiver_model"] == r]
                )
                matrix[i, j] = acc
        im = _heatmap(ax, matrix, models, f"{METHOD_LABEL[method]}, transmission pilot", vmin, vmax)
        ax.set_ylabel("")
    axes[0].set_ylabel("sender")
    fig.subplots_adjust(left=0.11, right=0.88, wspace=0.28, bottom=0.22)
    cax = fig.add_axes([0.90, 0.22, 0.015, 0.62])
    fig.colorbar(im, cax=cax, label="accuracy")
    _save(fig, "fig2_sender_receiver_heatmap")


def fig_ratio_distribution():
    corpora = [
        ("pilot_trans", "results/pilot_trans/messages.jsonl", "transmission (25 docs)"),
        ("b100", "results/v3probe4/messages.jsonl", "readable b100 (10 docs)"),
        ("base", "results/v3base4/messages.jsonl", "readable base (10 docs)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for ax, (name, path, title) in zip(axes, corpora):
        messages = read_jsonl(path)
        for i, method in enumerate(("abstractive", "structured")):
            vals = [100 * m["ratio_source"] for m in messages if m["method"] == method]
            jitter = np.random.default_rng(4 + i).normal(0, 0.05, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=5,
                alpha=0.45,
                color=METHOD_COLOR[method],
                linewidths=0,
            )
            ax.hlines(np.median(vals), i - 0.25, i + 0.25, color="black", lw=1.2)
        ax.axhspan(48, 52, color=GREEN, alpha=0.12, lw=0)
        ax.axhline(50, color="black", lw=0.7, ls="--")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["abs", "struct"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(title)
        ax.set_ylim(28, 105)
    axes[0].set_ylabel("message bytes / source bytes (%)")
    fig.tight_layout()
    _save(fig, "fig3_compliance_distribution")


def fig_compact_probe():
    report = json.loads(Path("reports/compact_probe.json").read_text())
    predictions = read_jsonl("results/compact_probe/predictions.jsonl")
    rows = flat_questions(predictions, pair_types={"self"})
    methods = ["abstractive", "structured", "compact"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    vals, errs, labels, colors = [], [], [], []
    for method in methods:
        sub = [r for r in rows if r["method"] == method]
        k, n, acc = accuracy(sub)
        lo, hi = _cluster_ci(sub)
        vals.append(100 * acc)
        errs.append((100 * (acc - lo), 100 * (hi - acc)))
        labels.append({"abstractive": "prose", "structured": "full JSON", "compact": "compact"}[method])
        colors.append(METHOD_COLOR[method])
    # Same-facts control: identical facts, compact encoding only.
    rs_rows = None
    try:
        rs_rows = [
            r
            for r in flat_questions(
                read_jsonl("results/compact_reserialize/predictions.jsonl"), pair_types={"self"}
            )
        ]
    except FileNotFoundError:
        rs_rows = None
    if rs_rows:
        k, n, acc = accuracy(rs_rows)
        lo, hi = _cluster_ci(rs_rows)
        vals.append(100 * acc)
        errs.append((100 * (acc - lo), 100 * (hi - acc)))
        labels.append("compact\n(same facts)")
        colors.append("#9ae6b4")
    ax.bar(range(len(vals)), vals, color=colors, width=0.62)
    ax.errorbar(range(len(vals)), vals, yerr=np.array(errs).T, fmt="none", ecolor="black", capsize=2.5, lw=1)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1f}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("self-receiver accuracy (%)")
    ax.set_ylim(60, 82)
    ax.set_title("(a) compact recovers part of the structured loss")

    ax = axes[1]
    sender = report["sender"]
    stmt = [sender["abstractive"]["mean_statement_bytes"], sender["structured"]["mean_statement_bytes"], sender["compact"]["mean_statement_bytes"]]
    syn = [sender["abstractive"]["mean_syntax_bytes"], sender["structured"]["mean_syntax_bytes"], sender["compact"]["mean_syntax_bytes"]]
    ax.bar(range(3), stmt, color=[METHOD_COLOR[m] for m in methods], width=0.62, label="statement bytes")
    ax.bar(range(3), syn, bottom=stmt, color="#cbd5e0", width=0.62, label="syntax bytes")
    for i, (s, y) in enumerate(zip(stmt, syn)):
        text = f"{s:.0f}+{y:.0f}" if y else f"{s:.0f}"
        ax.annotate(text, (i, s + y), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=6.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["prose", "full JSON", "compact"])
    ax.set_ylabel("mean bytes per message")
    ax.set_ylim(0, 540)
    ax.set_title("(b) where the 50% budget goes")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout()
    _save(fig, "fig4_compact_probe")


def fig_accuracy_vs_bytes():
    messages = {m["key"]: m for m in read_jsonl("results/pilot_trans/messages.jsonl")}
    predictions = read_jsonl("results/pilot_trans/predictions.jsonl")
    rows = []
    for rec in predictions:
        if rec["method"] == "full":
            continue
        msg = messages.get(rec["key"].rsplit("|recv:", 1)[0])
        if not msg:
            continue
        for q in rec["per_question"]:
            rows.append(
                {
                    "method": rec["method"],
                    "ratio": msg["ratio_source"],
                    "correct": q["correct"],
                }
            )
    bins = [(0.40, 0.44), (0.44, 0.48), (0.48, 0.52), (0.52, 0.56), (0.56, 0.60), (0.60, 0.70), (0.70, 1.05)]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for method in ("abstractive", "structured"):
        xs, ys, ns = [], [], []
        for lo, hi in bins:
            sub = [r for r in rows if r["method"] == method and lo <= r["ratio"] < hi]
            if len(sub) < 50:
                continue
            xs.append(100 * (lo + hi) / 2)
            ys.append(100 * sum(r["correct"] for r in sub) / len(sub))
            ns.append(len(sub))
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.3, color=METHOD_COLOR[method], label=METHOD_LABEL[method])
    ax.axvspan(48, 52, color=GREEN, alpha=0.12, lw=0)
    ax.set_xlabel("actual message bytes / source bytes (%)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("transmission pilot: accuracy rises with realized bytes")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig5_accuracy_vs_realized_bytes")


def fig_fact_retention():
    corpora = [
        ("base", "results/v3base4/messages.jsonl", "base"),
        ("b100", "results/v3probe4/messages.jsonl", "b100"),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for method in ("abstractive", "structured"):
        xs, ys = [], []
        for i, (_, path, label) in enumerate(corpora):
            messages = read_jsonl(path)
            vals = [m["retention"]["fact_retention"] for m in messages if m["method"] == method]
            xs.append(i)
            ys.append(100 * sum(vals) / len(vals))
        ax.plot(xs, ys, marker="o", ms=4, lw=1.3, color=METHOD_COLOR[method], label=METHOD_LABEL[method])
    ax.set_xticks(range(len(corpora)))
    ax.set_xticklabels([c[2] for c in corpora])
    ax.set_ylabel("lexical fact retention (%)")
    ax.set_title("sender-side retention collapses for structured at b100")
    ax.set_ylim(60, 90)
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    _save(fig, "fig6_fact_retention")


def fig_repeatability():
    report = json.loads(Path("reports/repeatability.json").read_text())
    reps = list(report["reps"])
    if not reps:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ax = axes[0]
    for method in ("abstractive", "structured"):
        ys = [100 * report["reps"][r]["by_method"][method]["accuracy"] for r in reps]
        ax.plot(range(len(reps)), ys, marker="o", ms=4, lw=1.2, color=METHOD_COLOR[method], label=METHOD_LABEL[method])
        for i, y in enumerate(ys):
            ax.annotate(f"{y:.1f}", (i, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6, color=METHOD_COLOR[method])
    ax.set_xticks(range(len(reps)))
    ax.set_xticklabels(reps)
    ax.set_ylabel("self-receiver accuracy (%)")
    ax.set_ylim(64, 83)
    ax.set_title("(a) same prompts, three fresh requests")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.4, columnspacing=1.0)

    ax = axes[1]
    gaps = [100 * report["reps"][r]["method_gap"] for r in reps]
    mean_gap = 100 * report["summary"]["method_gap"]["mean"]
    sd_gap = 100 * report["summary"]["method_gap"]["sd"]
    ax.bar(range(len(reps)), gaps, color=RED, width=0.55)
    ax.axhline(mean_gap, color="black", lw=1.0, ls="--", label=f"mean {mean_gap:.1f}pp")
    ax.fill_between([-0.5, len(reps) - 0.5], mean_gap - sd_gap, mean_gap + sd_gap, color=GREY, alpha=0.18, lw=0, label=f"±1 SD ({sd_gap:.1f}pp)")
    for i, g in enumerate(gaps):
        ax.annotate(f"{g:+.1f}", (i, g), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.set_xticks(range(len(reps)))
    ax.set_xticklabels(reps)
    ax.set_ylabel("method gap A−S (pp)")
    ax.set_ylim(0, 15)
    ax.set_title("(b) gap is positive but not noise-free")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.4, columnspacing=1.0)
    fig.tight_layout()
    _save(fig, "fig7_repeatability")


def fig_main_2x2():
    """Main-experiment 2x2 (reads reports/main_results.json; skips if absent)."""
    import json

    path = "reports/main_results.json"
    if not os.path.exists(path):
        print(f"  skip fig8: {path} not found")
        return
    rep = json.load(open(path))
    cells = rep["cells_all"]
    ci = rep.get("cells_all_ci") or {}
    densities = ["base", "b100"]
    labels = ["base", "b100"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    for method in ("abstractive", "structured"):
        xs, ys, errs = [], [], []
        for i, density in enumerate(densities):
            cell = cells[f"{density}|{method}"]
            acc = cell["accuracy"]
            xs.append(i)
            ys.append(100 * acc)
            b = ci.get(f"{density}|{method}")
            if b:
                errs.append((100 * (acc - b["ci_lo"]), 100 * (b["ci_hi"] - acc)))
            else:
                errs.append((0.0, 0.0))
        ax.errorbar(xs, ys, yerr=np.array(errs).T, marker="o", ms=5, lw=1.4,
                    capsize=2.5, color=METHOD_COLOR[method], label=METHOD_LABEL[method])
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=6.5, color=METHOD_COLOR[method])
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels)
    ax.set_ylabel("receiver accuracy (%)")
    ax.set_title("(a) main experiment, all receivers")
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1]
    per_sender = rep.get("per_sender") or {}
    models = list(per_sender)
    width = 0.36
    for j, density in enumerate(densities):
        gaps = [100 * (per_sender[m]["table"][f"{density}|gap"]["gap"] or 0.0) for m in models]
        ax.bar(np.arange(len(models)) + (j - 0.5) * width, gaps, width=width,
               color=GREY if j == 0 else RED, label=density, alpha=0.85)
    short = [m.replace("deepseek-v4-", "ds-").replace("qwen3.7-", "qwen-") for m in models]
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short, rotation=15, ha="right")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("method gap A−S (pp)")
    ax.set_title("(b) per sender")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig8_main_2x2")


def fig_compact_main():
    """Main-experiment compact ablation (reads reports/compact_compare.json)."""
    import json

    path = "reports/compact_compare.json"
    if not os.path.exists(path):
        print(f"  skip fig9: {path} not found")
        return
    rep = json.load(open(path))
    methods = ["abstractive", "structured", "compact"]
    cells = rep["cells"]
    ci = rep.get("cells_ci") or {}
    retention = rep.get("retention") or {}
    compliance = rep.get("compliance") or {}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    xs = range(len(methods))
    vals = [100 * cells[m]["accuracy"] for m in methods]
    errs = []
    for m in methods:
        b = ci.get(m)
        if b:
            errs.append((100 * (cells[m]["accuracy"] - b["ci_lo"]),
                         100 * (b["ci_hi"] - cells[m]["accuracy"])))
        else:
            errs.append((0.0, 0.0))
    ax.bar(xs, vals, width=0.62, color=[METHOD_COLOR[m] for m in methods])
    ax.errorbar(xs, vals, yerr=np.array(errs).T, fmt="none", ecolor="black",
                capsize=2.5, lw=1.0)
    for x, y in zip(xs, vals):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=6.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["prose", "full JSON", "compact"], rotation=0)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("(a) high density, self receiver")

    ax = axes[1]
    rets = [100 * (retention.get(m) or 0) for m in methods]
    ax.bar(xs, rets, width=0.62, color=[METHOD_COLOR[m] for m in methods])
    for x, (y, m) in enumerate(zip(rets, methods)):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=6.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["prose", "full JSON", "compact"], rotation=0)
    ax.set_ylabel("fact retention (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(b) how much of the source survives")
    fig.tight_layout()
    _save(fig, "fig9_compact_main")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="comma-separated figure names")
    args = parser.parse_args(argv)
    _style()
    builders = {
        "method_density": fig_method_density_2x2,
        "heatmap": fig_sender_receiver_heatmap,
        "compliance": fig_ratio_distribution,
        "compact": fig_compact_probe,
        "bytes": fig_accuracy_vs_bytes,
        "retention": fig_fact_retention,
        "repeatability": fig_repeatability,
        "main_2x2": fig_main_2x2,
        "compact_main": fig_compact_main,
    }
    selected = args.only.split(",") if args.only else list(builders)
    for name in selected:
        builders[name]()
        print(f"figure {name}: ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
