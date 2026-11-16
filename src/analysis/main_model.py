"""Pre-registered main analysis: Method x Density with the actual-bytes covariate.

Implements ``docs/freeze_v3_plan.md`` §4 (signed 2026-11-16 20:47):

    correct ~ method * density + actual_ratio + demand + (1|doc) + (1|doc:sender)
              + (1|doc:receiver) + (1|sender:receiver)

``actual_ratio`` enters because the 50% ±2% band is not met by every message
(D-5); messages are never dropped. Random intercepts are approximated two ways
that must agree before the number is reported:

* **GEE** -- binomial, exchangeable working correlation, clusters = documents
  (the resampling unit of the paired render). This is the primary estimate when
  ``statsmodels`` is available.
* **LPM** -- linear probability model with document-clustered robust SEs,
  implemented in numpy so the script still runs without statsmodels.

The interaction is also reported as a raw difference-in-differences with a
document-cluster bootstrap CI, which is the number the pilot table quotes
(+8.5pp all receivers).

Usage
-----
    python -m src.analysis.main_model --pilot          # validate on pilot data
    python -m src.analysis.main_model                  # main experiment
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import (
    accuracy,
    bootstrap_interaction,
    flat_questions,
    mcnemar_exact,
    paired_questions,
    paired_summary,
    read_jsonl,
    write_json,
)

METHODS = ("abstractive", "structured")

MAIN = {
    "base": {
        "predictions": "results/main_v3/predictions_base.jsonl",
        "predictions_cross": "results/main_v3/predictions_base_cross.jsonl",
        "data_dir": "data/_v3_paired/base",
    },
    "b100": {
        "predictions": "results/main_v3/predictions_b100.jsonl",
        "predictions_cross": "results/main_v3/predictions_b100_cross.jsonl",
        "data_dir": "data/_v3_paired/b100",
    },
}

PILOT = {
    "base": {
        "predictions": "results/v3base4/predictions.jsonl",
        "data_dir": "data/_density_probe_paired/base",
    },
    "b100": {
        "predictions": "results/v3probe4/predictions_4x4.jsonl",
        "data_dir": "data/_density_probe_paired/b100",
    },
}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _doc_bytes(data_dir: str) -> dict[str, int]:
    docs = read_jsonl(Path(data_dir) / "docs.jsonl")
    return {d["doc_id"]: int(d.get("bytes") or len(d["text"].encode("utf-8")))
            for d in docs}


def load_rows(spec: dict) -> list[dict]:
    """Flatten predictions and attach density, source bytes and actual ratio.

    The 4x4 files repeat the self-receiver records of the 30-document subset
    (they were cache hits there); records are de-duplicated on their ``key`` so
    that subset is not counted twice.
    """
    rows: list[dict] = []
    for density, paths in spec.items():
        doc_bytes = _doc_bytes(paths["data_dir"])
        files = [paths["predictions"]]
        if paths.get("predictions_cross") and Path(paths["predictions_cross"]).exists():
            files.append(paths["predictions_cross"])
        seen: set[str] = set()
        for path in files:
            records = []
            for rec in read_jsonl(path):
                key = rec.get("key")
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                records.append(rec)
            for row in flat_questions(records):
                source = doc_bytes.get(row["doc_id"])
                if not source:
                    continue
                row = dict(row)
                row["density"] = density
                row["source_bytes"] = source
                row["actual_ratio"] = (row["message_bytes"] / source
                                       if row["message_bytes"] else None)
                rows.append(row)
    if not rows:
        raise SystemExit("no prediction rows found for the given spec")
    return rows


def _add_covariates(rows: list[dict]) -> list[dict]:
    """Add centered actual_ratio and the fixed-effect indicator columns."""
    keep = [
        r for r in rows
        if r["method"] in METHODS
        and r.get("actual_ratio") is not None
        and not (r.get("pair_type") == "baseline_full")
    ]
    for r in keep:
        r["struct"] = 1.0 if r["method"] == "structured" else 0.0
        r["dense"] = 1.0 if r["density"] == "b100" else 0.0
        r["inter"] = r["struct"] * r["dense"]
        r["ratio_c"] = r["actual_ratio"] - 0.5
        r["multihop"] = 1.0 if r.get("demand") == "multi-hop" else 0.0
        r["self"] = 1.0 if r.get("pair_type") == "self" else 0.0
        r["y"] = 1.0 if r["correct"] else 0.0
    return keep


# ---------------------------------------------------------------------------
# descriptive tables
# ---------------------------------------------------------------------------

def cell_table(rows: list[dict], *, self_only: bool = False) -> dict:
    out: dict[str, dict] = {}
    for density in ("base", "b100"):
        for method in METHODS:
            sub = [r for r in rows if r["density"] == density and r["method"] == method]
            if self_only:
                sub = [r for r in sub if r["pair_type"] == "self"]
            k, n, acc = accuracy(sub)
            out[f"{density}|{method}"] = {"n_correct": k, "n_questions": n, "accuracy": acc}
        a = out[f"{density}|abstractive"]["accuracy"]
        s = out[f"{density}|structured"]["accuracy"]
        out[f"{density}|gap"] = {"gap": (a - s) if a is not None and s is not None else None}
    base_gap = out["base|gap"]["gap"]
    dense_gap = out["b100|gap"]["gap"]
    out["interaction"] = {
        "interaction": (dense_gap - base_gap)
        if base_gap is not None and dense_gap is not None else None
    }
    return out


def cell_ci(rows: list[dict], *, n_boot: int = 4000, seed: int = 11,
            self_only: bool = False) -> dict:
    """Doc-cluster bootstrap CI for each of the four cell means."""
    import random

    out: dict[str, dict] = {}
    for density in ("base", "b100"):
        for method in METHODS:
            sub = [r for r in rows
                   if r["density"] == density and r["method"] == method]
            if self_only:
                sub = [r for r in sub if r["pair_type"] == "self"]
            by_doc: dict[str, list[int]] = defaultdict(list)
            for r in sub:
                by_doc[r["doc_id"]].append(int(r["correct"]))
            docs = sorted(by_doc)
            if not docs:
                out[f"{density}|{method}"] = None
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
            out[f"{density}|{method}"] = {
                "point": point,
                "ci_lo": vals[int(0.025 * len(vals))],
                "ci_hi": vals[int(0.975 * len(vals))],
            }
    return out


def interaction_bootstrap(rows: list[dict], *, self_only: bool = False) -> dict:
    """Doc-cluster bootstrap of (gap at b100) - (gap at base)."""
    by_density: dict[str, dict[str, list[dict]]] = {}
    for density in ("base", "b100"):
        sub = [r for r in rows if r["density"] == density]
        if self_only:
            sub = [r for r in sub if r["pair_type"] == "self"]
        by_density[density] = {
            m: [r for r in sub if r["method"] == m] for m in METHODS
        }
    pairs_abs = paired_questions(by_density["base"]["abstractive"],
                                 by_density["b100"]["abstractive"])
    pairs_struct = paired_questions(by_density["base"]["structured"],
                                    by_density["b100"]["structured"])
    point, lo, hi = bootstrap_interaction(pairs_abs, pairs_struct)
    return {"point": point, "ci_lo": lo, "ci_hi": hi,
            "n_pairs_abs": len(pairs_abs), "n_pairs_struct": len(pairs_struct)}


def mcnemar_tables(rows: list[dict]) -> dict:
    """Paired base-vs-b100 tests within method (same doc/sender/receiver/question)."""
    out = {}
    for method in METHODS:
        sub = [r for r in rows if r["method"] == method]
        base = [r for r in sub if r["density"] == "base"]
        dense = [r for r in sub if r["density"] == "b100"]
        pairs = paired_questions(base, dense)
        out[method] = paired_summary(pairs)
    return out


def baseline_full(rows: list[dict]) -> dict:
    """Full-document upper bound, per density (not part of the 2x2)."""
    out = {}
    for density in ("base", "b100"):
        sub = [r for r in rows if r["density"] == density and r["method"] == "full"]
        k, n, acc = accuracy(sub)
        out[density] = {"n_correct": k, "n_questions": n, "accuracy": acc}
    return out


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def design_matrix(rows: list[dict], *, with_self: bool = True) -> tuple[np.ndarray, list[str]]:
    """Columns: intercept, struct, dense, struct*dense, ratio_c, multihop, self."""
    cols = [np.ones(len(rows)), np.array([r["struct"] for r in rows]),
            np.array([r["dense"] for r in rows]), np.array([r["inter"] for r in rows]),
            np.array([r["ratio_c"] for r in rows]),
            np.array([r["multihop"] for r in rows])]
    names = ["intercept", "struct", "dense", "struct_x_dense", "ratio_c", "multihop"]
    if with_self:
        cols.append(np.array([r["self"] for r in rows]))
        names.append("self")
    return np.column_stack(cols), names


def fit_lpm_cluster(rows: list[dict]) -> dict:
    """Linear probability model, document-clustered robust covariance."""
    X, names = design_matrix(rows)
    y = np.array([r["y"] for r in rows])
    clusters = [r["doc_id"] for r in rows]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        XtX_inv = np.linalg.pinv(X.T @ X)
    if not np.isfinite(beta).all():
        raise RuntimeError("LPM produced non-finite coefficients")

    groups: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for idx in groups.values():
        Xg = X[idx]
        ug = resid[idx]
        score = Xg.T @ ug
        meat += np.outer(score, score)

    n, k, g = len(rows), X.shape[1], len(groups)
    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    cov = XtX_inv @ meat @ XtX_inv * correction
    # guard: a degenerate cluster can make a diagonal entry marginally
    # negative through floating-point cancellation
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    coefs = {}
    for name, b, s in zip(names, beta, se):
        z = b / s if s else None
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))) if z is not None else None
        coefs[name] = {"coef": float(b), "se": float(s), "z": float(z) if z else None,
                       "p": float(p) if p is not None else None}
    # H2 is about the *gap* (abstractive - structured) widening with density.
    # With struct=1 and dense=1 indicators, the struct_x_dense coefficient is
    # the negative of that widening: a negative coefficient means structured
    # loses more at b100. Both readings are reported explicitly.
    coef_struct_x_dense = coefs["struct_x_dense"]["coef"]
    return {
        "model": "LPM (doc-clustered robust SE)", "n": n, "n_clusters": g,
        "coefs": coefs,
        "struct_extra_loss_pp": 100 * coef_struct_x_dense,
        "h2_gap_interaction_pp": -100 * coef_struct_x_dense,
    }


def fit_gee(rows: list[dict]) -> dict | None:
    """Binomial GEE, exchangeable working correlation, clusters = documents."""
    try:
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.generalized_estimating_equations import GEE
    except Exception:  # noqa: BLE001 - optional dependency
        return None

    X, names = design_matrix(rows)
    y = np.array([r["y"] for r in rows])
    codes = {d: i for i, d in enumerate(sorted({r["doc_id"] for r in rows}))}
    groups = np.array([codes[r["doc_id"]] for r in rows])
    try:
        result = GEE(y, X, groups=groups, family=Binomial(),
                     cov_struct=Exchangeable()).fit()
    except Exception as exc:  # noqa: BLE001 - report, do not crash
        return {"model": "GEE", "error": f"{type(exc).__name__}: {exc}"}

    params = np.asarray(result.params)
    # statsmodels names an ndarray exog x1..xk; keep the readable names instead
    coefs = {name: {"coef": float(p), "se": float(s), "z": float(z), "p": float(pv)}
             for name, p, s, z, pv in zip(names, params,
                                          np.asarray(result.bse),
                                          np.asarray(result.tvalues),
                                          np.asarray(result.pvalues))}

    # marginal (g-computation) cell means: average predicted probability with
    # the design toggled to each cell, keeping every other covariate observed.
    Xc = X.copy()
    cells = {}
    for dense in (0.0, 1.0):
        for struct in (0.0, 1.0):
            Xc[:, 1] = struct
            Xc[:, 2] = dense
            Xc[:, 3] = struct * dense
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                eta = Xc @ params
            if not np.isfinite(eta).all():
                raise RuntimeError("GEE linear predictor is not finite")
            p = 1.0 / (1.0 + np.exp(-eta))  # logit inverse
            cells[f"{'b100' if dense else 'base'}|"
                  f"{'structured' if struct else 'abstractive'}"] = float(np.mean(p))
    gap_base = cells["base|abstractive"] - cells["base|structured"]
    gap_dense = cells["b100|abstractive"] - cells["b100|structured"]
    return {
        "model": "GEE binomial (exchangeable, doc clusters)",
        "n": int(len(rows)),
        "n_clusters": int(len(codes)),
        "coefs": coefs,
        "marginal_cells": cells,
        "h2_gap_interaction_pp": 100 * (gap_dense - gap_base),
        "struct_extra_loss_pp": -100 * (gap_dense - gap_base),
    }


# ---------------------------------------------------------------------------
# D-11 sensitivity
# ---------------------------------------------------------------------------

def premise_flagged_qids(data_dirs: dict[str, str]) -> dict[str, list[str]]:
    """D-11: qids whose premise the source contradicts, per density."""
    from ..data.validate_paired import premise_flags

    out = {}
    for density, data_dir in data_dirs.items():
        corpus = {
            "docs": read_jsonl(Path(data_dir) / "docs.jsonl"),
            "by_doc": _questions_by_doc(Path(data_dir) / "questions.jsonl"),
        }
        out[density] = sorted(f["qid"] for f in premise_flags(corpus))
    return out


def _questions_by_doc(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for q in read_jsonl(path):
        out[q["doc_id"]].append(q)
    return out


def drop_flagged(rows: list[dict], flagged: dict[str, list[str]]) -> list[dict]:
    drop = {(density, qid) for density, qids in flagged.items() for qid in qids}
    return [r for r in rows if (r["density"], r["qid"]) not in drop]


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _pct(value) -> str:
    return "-" if value is None else f"{100 * value:.2f}%"


def _pp(value) -> str:
    """Format a *fraction* difference as percentage points."""
    return "-" if value is None else f"{100 * value:+.1f}pp"


def _pp_value(value) -> str:
    """Format a number that is already in percentage points."""
    return "-" if value is None else f"{value:+.2f}pp"


def _p(value) -> str:
    if value is None:
        return "-"
    return "<0.0001" if value < 1e-4 else f"{value:.4f}"


def analyze(spec: dict, *, n_boot: int = 4000, seed: int = 11) -> dict:
    rows_all = load_rows(spec)
    rows = _add_covariates(rows_all)
    rows = [r for r in rows if r.get("pair_type") != "baseline_full"]

    table_all = cell_table(rows, self_only=False)
    table_self = cell_table(rows, self_only=True)
    ci_all = cell_ci(rows, n_boot=n_boot, seed=seed, self_only=False)
    inter_all = interaction_bootstrap(rows, self_only=False)
    inter_self = interaction_bootstrap(rows, self_only=True)
    mcnemar = mcnemar_tables(rows)
    full = baseline_full(rows_all)

    lpm = fit_lpm_cluster(rows)
    gee = fit_gee(rows)

    flagged = premise_flagged_qids({d: p["data_dir"] for d, p in spec.items()})
    rows_sens = drop_flagged(rows, flagged)
    sens = {
        "n_flagged": sum(len(v) for v in flagged.values()),
        "flagged": flagged,
        "n_rows_after": len(rows_sens),
        "table": cell_table(rows_sens),
        "interaction": interaction_bootstrap(rows_sens),
        "lpm": fit_lpm_cluster(rows_sens),
    }

    # exploratory: per sender and per demand
    per_sender = {}
    for sender in sorted({r["sender_model"] for r in rows}):
        sub = [r for r in rows if r["sender_model"] == sender]
        per_sender[sender] = {
            "table": cell_table(sub),
            "interaction": interaction_bootstrap(sub),
        }
    per_demand = {}
    for demand in ("lookup", "multi-hop"):
        sub = [r for r in rows if r.get("demand") == demand]
        if not sub:
            continue
        per_demand[demand] = {
            "table": cell_table(sub),
            "interaction": interaction_bootstrap(sub),
        }

    return {
        "spec": {d: {k: str(v) for k, v in p.items()} for d, p in spec.items()},
        "n_rows": len(rows),
        "n_docs": len({r["doc_id"] for r in rows}),
        "n_senders": len({r["sender_model"] for r in rows}),
        "n_receivers": len({r["receiver_model"] for r in rows}),
        "cells_all": table_all,
        "cells_all_ci": ci_all,
        "cells_self": table_self,
        "interaction_all": inter_all,
        "interaction_self": inter_self,
        "mcnemar": mcnemar,
        "baseline_full": full,
        "lpm": lpm,
        "gee": gee,
        "d11_sensitivity": sens,
        "per_sender": per_sender,
        "per_demand": per_demand,
        "bootstrap": {"n_boot": n_boot, "seed": seed},
    }


def render_markdown(rep: dict) -> str:
    c = rep["cells_all"]
    s = rep["cells_self"]
    lines = [
        "# 主分析：Method × Density（预注册）",
        "",
        "> 脚本 `src/analysis/main_model.py`；模型见 `docs/freeze_v3_plan.md` §4（签字版 v1.0）。",
        "> `actual_ratio` 始终进模型；不合规消息不删分母（D-5）；D-11 做对称剔除敏感性。",
        "",
        f"- 观测：{rep['n_rows']:,} 题级行 · {rep['n_docs']} 篇 · "
        f"{rep['n_senders']} sender × {rep['n_receivers']} receiver",
        f"- 语料：`{rep['spec']['base']['data_dir']}` / `{rep['spec']['b100']['data_dir']}`",
        "",
        "## 1. 2×2 表（all receivers）",
        "",
        "| 密度 | abstractive | structured | 方法差 A−S | n/格 |",
        "|---|---:|---:|---:|---:|",
    ]
    ci = rep.get("cells_all_ci") or {}
    for density, label in (("base", "base"), ("b100", "b100")):
        a = c[f"{density}|abstractive"]
        st = c[f"{density}|structured"]
        def _with_ci(key, cell):
            b = ci.get(key)
            if not b:
                return _pct(cell["accuracy"])
            return (f"{_pct(cell['accuracy'])} "
                    f"[{_pct(b['ci_lo'])}, {_pct(b['ci_hi'])}]")
        lines.append(f"| {label} | {_with_ci(f'{density}|abstractive', a)} | "
                     f"{_with_ci(f'{density}|structured', st)} | "
                     f"{_pp(c[f'{density}|gap']['gap'])} | "
                     f"{a['n_questions']} / {st['n_questions']} |")
    lines += [
        "",
        f"**交互（raw DiD）**：{_pp(c['interaction']['interaction'])}；"
        f"doc-cluster bootstrap 95% CI "
        f"[{_pp(rep['interaction_all']['ci_lo'])}, {_pp(rep['interaction_all']['ci_hi'])}]"
        f"（{rep['interaction_all']['n_pairs_abs']:,} 对 abstractive / "
        f"{rep['interaction_all']['n_pairs_struct']:,} 对 structured）。",
        "",
        "### self-only（附录）",
        "",
        "| 密度 | abstractive | structured | 方法差 |",
        "|---|---:|---:|---:|",
    ]
    for density in ("base", "b100"):
        lines.append(f"| {density} | {_pct(s[f'{density}|abstractive']['accuracy'])} | "
                     f"{_pct(s[f'{density}|structured']['accuracy'])} | "
                     f"{_pp(s[f'{density}|gap']['gap'])} |")
    lines += [
        "",
        f"self-only 交互 {_pp(s['interaction']['interaction'])}"
        f"（CI [{_pp(rep['interaction_self']['ci_lo'])}, {_pp(rep['interaction_self']['ci_hi'])}]）。",
        "",
        "## 2. 预注册模型",
        "",
        f"### LPM（doc-clustered robust SE, {rep['lpm']['n_clusters']} clusters）",
        "",
        "| 项 | 系数 (pp) | SE | z | p |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, coef in rep["lpm"]["coefs"].items():
        lines.append(f"| `{name}` | {100 * coef['coef']:+.2f} | {100 * coef['se']:.2f} | "
                     f"{coef['z']:.2f} | {_p(coef['p'])} |")
    gee = rep.get("gee") or {}
    lines += [
        "",
        "> `ratio_c = actual_ratio − 0.5`（actual_ratio = message_bytes / source_bytes）；",
        "> 系数单位是“actual_ratio 每增加 1.0（即源字节的 100%）”的百分点，换算成常见的",
        "> “预算带内 +10pp（0.1）”要除以 10。",
        "",
        f"`struct_x_dense` 是 **structured 额外损失**：{_pp_value(rep['lpm']['struct_extra_loss_pp'])}；"
        f"换成 H2 的“方法差扩大”口径是 **{_pp_value(rep['lpm']['h2_gap_interaction_pp'])}**。",
    ]
    if gee.get("coefs"):
        lines += [
            "",
            f"### GEE binomial（exchangeable, {gee['n_clusters']} doc clusters）",
            "",
            "| 项 | 系数 (logit) | SE | z | p |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, coef in gee["coefs"].items():
            lines.append(f"| `{name}` | {coef['coef']:+.4f} | {coef['se']:.4f} | "
                         f"{coef['z']:.2f} | {_p(coef['p'])} |")
        if gee.get("marginal_cells"):
            lines += ["", "GEE 边际（平均预测概率）：", "",
                      "| 格 | 概率 |", "|---|---:|"]
            for key in ("base|abstractive", "base|structured", "b100|abstractive", "b100|structured"):
                lines.append(f"| {key} | {_pct(gee['marginal_cells'].get(key))} |")
            lines.append("")
            lines.append(
                f"GEE 边际交互（平均预测概率的差之差，H2 口径）："
                f"**{_pp_value(gee['h2_gap_interaction_pp'])}**；"
                f"structured 额外损失 {_pp_value(gee['struct_extra_loss_pp'])}。"
            )
    elif gee.get("error"):
        lines += ["", f"> GEE 未拟合：`{gee['error']}`（以 LPM 与 bootstrap 为准）。"]

    lines += [
        "",
        "## 3. 配对 McNemar（base vs b100，同题）",
        "",
        "| 方法 | n 对 | base 对→b100 错 | base 错→b100 对 | 差 | p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, m in rep["mcnemar"].items():
        lines.append(f"| {method} | {m['n_pairs']:,} | {m['a_only_correct']:,} | "
                     f"{m['b_only_correct']:,} | {_pp(m['diff_a_minus_b'])} | {_p(m['mcnemar_p'])} |")

    full = rep["baseline_full"]
    lines += [
        "",
        "## 4. 全文档基线（上界，不进 2×2）",
        "",
        "| 密度 | 准确率 | n |",
        "|---|---:|---:|",
    ]
    for density in ("base", "b100"):
        f = full[density]
        lines.append(f"| {density} | {_pct(f['accuracy'])} | {f['n_questions']:,} |")

    d11 = rep["d11_sensitivity"]
    lines += [
        "",
        "## 5. D-11 敏感性（对称剔除前提冲突题）",
        "",
        f"- 剔除：base {len(d11['flagged']['base'])} 题 / b100 "
        f"{len(d11['flagged']['b100'])} 题；剩余 {d11['n_rows_after']:,} 行。",
        "",
        "| 口径 | 交互 | 95% CI | LPM 交互 (pp) |",
        "|---|---:|---|---:|",
        f"| 主分析（全部题） | {_pp(c['interaction']['interaction'])} | "
        f"[{_pp(rep['interaction_all']['ci_lo'])}, {_pp(rep['interaction_all']['ci_hi'])}] | "
        f"{_pp_value(rep['lpm']['h2_gap_interaction_pp'])} |",
        f"| 剔除 D-11 | {_pp(d11['table']['interaction']['interaction'])} | "
        f"[{_pp(d11['interaction']['ci_lo'])}, {_pp(d11['interaction']['ci_hi'])}] | "
        f"{_pp_value(d11['lpm']['h2_gap_interaction_pp'])} |",
        "",
        "## 6. 探索：分 sender / 分 demand",
        "",
        "| Sender | base 方法差 | b100 方法差 | 交互 |",
        "|---|---:|---:|---:|",
    ]
    for sender, block in rep["per_sender"].items():
        t = block["table"]
        lines.append(f"| {sender} | {_pp(t['base|gap']['gap'])} | {_pp(t['b100|gap']['gap'])} | "
                     f"{_pp(t['interaction']['interaction'])} |")
    if rep["per_demand"]:
        lines += ["", "| Demand | base 方法差 | b100 方法差 | 交互 |", "|---|---:|---:|---:|"]
        for demand, block in rep["per_demand"].items():
            t = block["table"]
            lines.append(f"| {demand} | {_pp(t['base|gap']['gap'])} | {_pp(t['b100|gap']['gap'])} | "
                         f"{_pp(t['interaction']['interaction'])} |")
    lines += [
        "",
        "> 分 sender / 分 demand 只作探索：per-sender 的重复 SD 在 pilot 上最大 4.72pp，",
        "> 主实验单次跑不足以对这些子组做强断言（见 `reports/repeatability.md`）。",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", action="store_true",
                        help="validate the pipeline on the pilot prediction files")
    parser.add_argument("--n-boot", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    spec = PILOT if args.pilot else MAIN
    out = args.out or ("reports/main_model_pilot.md" if args.pilot else "reports/main_results.md")
    json_out = args.json_out or (
        "reports/main_model_pilot.json" if args.pilot else "reports/main_results.json")
    for density, paths in spec.items():
        if not Path(paths["predictions"]).exists():
            raise SystemExit(f"missing predictions for {density}: {paths['predictions']}")

    rep = analyze(spec, n_boot=args.n_boot, seed=args.seed)
    write_json(json_out, rep)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(render_markdown(rep) + "\n", encoding="utf-8")
    print(f"wrote {out} and {json_out}")
    gee = rep.get("gee") or {}
    print(f"rows={rep['n_rows']:,} docs={rep['n_docs']} "
          f"raw DiD={_pp(rep['cells_all']['interaction']['interaction'])} "
          f"LPM={_pp_value(rep['lpm']['h2_gap_interaction_pp'])} "
          f"GEE={_pp_value(gee.get('h2_gap_interaction_pp'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
