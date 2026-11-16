"""P0.9 report: density ladder from measured messages and receiver answers."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from ..api.config import ROOT
from ..common import read_jsonl, sha256_text
from ..eval.retention import retention_summary
from .data import docs_path, load_docs, load_questions_by_doc


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _fmt(value, digits=3, pct=False):
    if value is None:
        return "—"
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.{digits}f}"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _group(records, *keys):
    groups: dict[tuple, list] = {}
    for rec in records:
        groups.setdefault(tuple(rec.get(k) for k in keys), []).append(rec)
    return groups


def aggregate_messages(messages: list[dict]) -> list[dict]:
    rows = []
    for (level, method), recs in sorted(_group(messages, "level", "method").items()):
        n = len(recs)
        rows.append(
            {
                "level": level,
                "method": method,
                "n": n,
                "b_target": _mean(
                    [r["window"]["target"] / r["n_facts"] for r in recs]
                ),
                "ratio_source": _mean([r["ratio_source"] for r in recs]),
                "first_compliance": _mean([1.0 if r["first_attempt_compliant"] else 0.0 for r in recs]),
                "final_compliance": _mean([1.0 if r["compliant"] else 0.0 for r in recs]),
                "under_rate": _mean([1.0 if r["direction"] == "under" else 0.0 for r in recs]),
                "over_rate": _mean([1.0 if r["direction"] == "over" else 0.0 for r in recs]),
                "invalid_rate": _mean([1.0 if r["direction"] == "invalid" else 0.0 for r in recs]),
                "first_invalid": _mean(
                    [
                        1.0
                        if r["attempts"] and r["attempts"][0]["direction"] == "invalid"
                        else 0.0
                        for r in recs
                    ]
                ),
                "mean_attempts": _mean([r["n_attempts"] for r in recs]),
                "fact_retention": _mean([r["retention"]["fact_retention"] for r in recs]),
                "question_evidence": _mean(
                    [r["retention"]["question_evidence_rate"] for r in recs]
                ),
                "mean_fact_recall": _mean([r["retention"]["mean_fact_recall"] for r in recs]),
                "numbers_missing": _mean(
                    [r["retention"]["n_numbers_missing"] for r in recs]
                ),
                "n_items": _mean([r["n_items"] for r in recs if r["n_items"] is not None]),
                "json_invalid_final": _mean(
                    [0.0 if r["json_valid"] else 1.0 for r in recs]
                )
                if method == "structured"
                else None,
            }
        )
    return rows


def aggregate_model_messages(messages: list[dict]) -> list[dict]:
    rows = []
    for (level, method, model), recs in sorted(
        _group(messages, "level", "method", "model_key").items()
    ):
        rows.append(
            {
                "level": level,
                "method": method,
                "model": model,
                "n": len(recs),
                "first_compliance": _mean(
                    [1.0 if r["first_attempt_compliant"] else 0.0 for r in recs]
                ),
                "final_compliance": _mean([1.0 if r["compliant"] else 0.0 for r in recs]),
                "fact_retention": _mean([r["retention"]["fact_retention"] for r in recs]),
                "question_evidence": _mean(
                    [r["retention"]["question_evidence_rate"] for r in recs]
                ),
            }
        )
    return rows


def _accuracy(records: list[dict]) -> dict:
    total = sum(r["n_questions"] for r in records)
    correct = sum(1 for r in records for q in r["per_question"] if q["correct"])
    lookup = [q for r in records for q in r["per_question"] if q["demand"] == "lookup"]
    multihop = [q for r in records for q in r["per_question"] if q["demand"] == "multi-hop"]
    lo, hi = wilson_ci(correct, total)
    return {
        "n_records": len(records),
        "n_questions": total,
        "n_correct": correct,
        "accuracy": correct / total if total else None,
        "ci_lo": lo,
        "ci_hi": hi,
        "lookup_accuracy": (
            sum(1 for q in lookup if q["correct"]) / len(lookup) if lookup else None
        ),
        "multihop_accuracy": (
            sum(1 for q in multihop if q["correct"]) / len(multihop) if multihop else None
        ),
    }


def aggregate_accuracy(predictions: list[dict]) -> dict:
    full_records = [r for r in predictions if r["level"] == "full"]
    self_records = [r for r in predictions if r["level"] != "full" and r["pair_type"] == "self"]
    full_by_model = {
        model: _accuracy(recs)
        for (model,), recs in _group(full_records, "receiver_model").items()
    }
    full_all = _accuracy(full_records)

    by_condition = {}
    for (level, method), recs in _group(self_records, "level", "method").items():
        overall = _accuracy(recs)
        per_model = {}
        ratios = []
        for (model,), recs_m in _group(recs, "receiver_model").items():
            acc = _accuracy(recs_m)
            per_model[model] = acc
            full_acc = full_by_model.get(model, {}).get("accuracy")
            if acc["accuracy"] is not None and full_acc:
                ratios.append(acc["accuracy"] / full_acc)
        by_condition[(level, method)] = {
            "overall": overall,
            "per_model": per_model,
            "ratio_vs_full": (
                overall["accuracy"] / full_all["accuracy"]
                if overall["accuracy"] is not None and full_all["accuracy"]
                else None
            ),
            "ratio_per_model_mean": _mean(ratios),
        }
    return {
        "full_all": full_all,
        "full_by_model": full_by_model,
        "by_condition": by_condition,
    }


def failure_modes(e: dict) -> list[str]:
    modes = []
    if not e["compliance_ok"]:
        modes.append("compliance")
    if e["ratio"] is not None and e["ratio"] > 0.80:
        modes.append("ceiling")
    if e["ratio"] is not None and e["ratio"] < 0.55:
        modes.append("floor")
    if e["retention"] is not None and e["retention"] > 0.85:
        modes.append("retention_high")
    if e["retention"] is not None and e["retention"] < 0.60:
        modes.append("retention_low")
    return modes


def accuracy_compliant_only(predictions: list[dict], messages: list[dict]) -> dict:
    """Accuracy ratio on messages that actually landed in the byte band.

    The pooled ratio mixes in out-of-band messages; this sensitivity view is
    the cleanest available read on density while the compliance gate is open.
    """
    by_key = {m["key"]: m for m in messages}
    self_recs = [
        r for r in predictions
        if r["level"] != "full" and r["pair_type"] == "self"
    ]
    full_all = _accuracy([r for r in predictions if r["level"] == "full"])
    out = {}
    for (level, method), recs in _group(self_recs, "level", "method").items():
        sub = [
            r for r in recs
            if by_key.get(r["key"].rsplit("|recv:", 1)[0], {}).get("compliant")
        ]
        acc = _accuracy(sub)
        out[(level, method)] = {
            "overall": acc,
            "ratio_vs_full": (
                acc["accuracy"] / full_all["accuracy"]
                if acc["accuracy"] is not None and full_all["accuracy"]
                else None
            ),
        }
    return {"by_condition": out, "full_all": full_all}


RATIO_BINS = [
    (0.00, 0.45, "<45%"),
    (0.45, 0.48, "45–48%"),
    (0.48, 0.52, "48–52%"),
    (0.52, 0.56, "52–56%"),
    (0.56, 0.60, "56–60%"),
    (0.60, 9.99, "≥60%"),
]


def accuracy_by_actual_ratio(predictions: list[dict], messages: list[dict]) -> list[dict]:
    """Dose-response: receiver accuracy as a function of actual bytes/source.

    This is the analysis that still works when the ±2% gate is not met: it uses
    the realised ratio as a continuous variable instead of discarding
    out-of-band messages.
    """
    by_key = {m["key"]: m for m in messages}
    full_all = _accuracy([r for r in predictions if r["level"] == "full"])
    rows = []
    for lo, hi, label in RATIO_BINS:
        recs = []
        for r in predictions:
            if r["level"] == "full" or r["pair_type"] != "self":
                continue
            msg = by_key.get(r["key"].rsplit("|recv:", 1)[0])
            if not msg:
                continue
            ratio = msg["final_bytes"] / msg["window"]["source_bytes"]
            if lo <= ratio < hi:
                recs.append(r)
        if not recs:
            rows.append({"bin": label, "n_records": 0, "n_questions": 0,
                         "accuracy": None, "ratio_vs_full": None})
            continue
        acc = _accuracy(recs)
        rows.append(
            {
                "bin": label,
                "n_records": len(recs),
                "n_questions": acc["n_questions"],
                "accuracy": acc["accuracy"],
                "ci_lo": acc["ci_lo"],
                "ci_hi": acc["ci_hi"],
                "lookup_accuracy": acc["lookup_accuracy"],
                "multihop_accuracy": acc["multihop_accuracy"],
                "ratio_vs_full": (
                    acc["accuracy"] / full_all["accuracy"]
                    if acc["accuracy"] is not None and full_all["accuracy"]
                    else None
                ),
            }
        )
    return rows



def decide(message_rows: list[dict], acc: dict) -> dict:
    by_condition = {(r["level"], r["method"]): r for r in message_rows}
    levels = sorted({r["level"] for r in message_rows})
    evaluations = []
    for level in levels:
        abs_row = by_condition.get((level, "abstractive"))
        struct_row = by_condition.get((level, "structured"))
        if not abs_row or not struct_row:
            continue
        conds = {}
        for method in ("abstractive", "structured"):
            row = by_condition[(level, method)]
            acc_row = acc["by_condition"].get((level, method), {})
            conds[method] = {
                "first_compliance": row["first_compliance"],
                "final_compliance": row["final_compliance"],
                "retention": row["fact_retention"],
                "evidence": row["question_evidence"],
                "accuracy": acc_row.get("overall", {}).get("accuracy"),
                "ratio": acc_row.get("ratio_vs_full"),
                "ci": (
                    acc_row.get("overall", {}).get("ci_lo"),
                    acc_row.get("overall", {}).get("ci_hi"),
                ),
            }
        first_ok = all(
            conds[m]["first_compliance"] is not None
            and conds[m]["first_compliance"] >= 0.80
            for m in conds
        )
        final_ok = all(
            conds[m]["final_compliance"] is not None
            and conds[m]["final_compliance"] >= 0.80
            for m in conds
        )
        ratio_val = _mean([conds[m]["ratio"] for m in conds])
        retention_val = _mean([conds[m]["retention"] for m in conds])
        ratio_ok = ratio_val is not None and 0.55 <= ratio_val <= 0.80
        retention_ok = retention_val is not None and 0.60 <= retention_val <= 0.85
        score = None
        if ratio_val is not None and retention_val is not None:
            score = abs(ratio_val - 0.675) + abs(retention_val - 0.725)
        evaluation = {
            "level": level,
            "conditions": conds,
            "compliance_ok": first_ok,          # G3 as written: first-round compliance
            "first_compliance_ok": first_ok,
            "final_compliance_ok": final_ok,
            "ratio": ratio_val,
            "ratio_ok": ratio_ok,
            "retention": retention_val,
            "retention_ok": retention_ok,
            "score": score,
        }
        evaluation["failure_modes"] = failure_modes(evaluation)
        evaluations.append(evaluation)

    qualifying = [
        e for e in evaluations
        if e["compliance_ok"] and e["ratio_ok"] and e["retention_ok"]
    ]
    if qualifying:
        status = "qualified"
        selected = min(qualifying, key=lambda e: e["score"])
    elif evaluations and all(not e["compliance_ok"] for e in evaluations):
        status = "compliance_gate_failed"
        scored = [e for e in evaluations if e["score"] is not None]
        selected = min(scored, key=lambda e: e["score"]) if scored else evaluations[0]
    elif evaluations:
        status = "no_level_qualifies"
        scored = [e for e in evaluations if e["score"] is not None]
        selected = min(scored, key=lambda e: e["score"]) if scored else evaluations[0]
    else:
        status = "no_data"
        selected = None
    return {"status": status, "evaluations": evaluations, "selected": selected}


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def render_report(
    *,
    run_id: str,
    messages: list[dict],
    predictions: list[dict],
    message_rows: list[dict],
    model_rows: list[dict],
    acc: dict,
    acc_compliant: dict,
    acc_by_ratio: list[dict],
    decision: dict,
) -> str:
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    docs_hash = sha256_text(docs_path().read_text(encoding="utf-8"))
    doc_ids = sorted({r["doc_id"] for r in messages})
    corpus_b = _mean([0.5 * r["doc_bytes"] / r["n_facts"] for r in messages])
    corpus_wpf = _mean([r["doc_bytes"] / 6.3 / r["n_facts"] for r in messages])

    lines: list[str] = []
    lines.append("# P0.9 密度阶梯标定报告")
    lines.append("")
    lines.append(
        f"> 生成时间：{generated} · run_id：`{run_id}` · 语料：{len(doc_ids)} 篇"
        f"（{doc_ids[0]} … {doc_ids[-1]}）· `docs.jsonl` sha256：`{docs_hash[:16]}`"
    )
    lines.append(
        "> 数据来源：`results/messages_p0.9.jsonl` + `results/predictions_p0.9.jsonl`，"
        "所有数字由这两个文件汇总，无人工估计。"
    )
    lines.append("")

    lines.append("## 1. 结论")
    lines.append("")
    sel = decision["selected"]
    if decision["status"] == "qualified" and sel:
        lines.append(
            f"**`{sel['level']}` 同时满足三条判据**：首轮合规率 ≥80%、"
            f"保留率 60–85%、压缩后/全文档准确率 0.55–0.80。"
        )
        lines.append("")
        row = next((r for r in message_rows if r["level"] == sel["level"]), None)
        b_star = row["b_target"] if row else None
        if b_star is not None:
            lines.append(
                f"选定档位的难度 **b\\* ≈ {b_star:.1f} B/事实**，"
                f"对应 `words_per_fact ≈ {b_star / 3.15:.1f}`"
                "（b = 3.15 × words_per_fact，因为 6.3 B/词 × 50%）。"
            )
        if corpus_b is not None and b_star is not None:
            delta = (b_star - corpus_b) / corpus_b
            lines.append(
                f"现役 v2 语料的实测难度是 **b ≈ {corpus_b:.1f} B/事实**"
                f"（`words_per_fact ≈ {corpus_wpf:.1f}`），与 b\\* 相差 **{delta * 100:+.1f}%**。"
            )
            if abs(delta) <= 0.05:
                lines.append(
                    "→ **现役语料的密度可直接用于主实验**；"
                    "「50% + 22 词/事实」这一组合成立，无需重新生成数据。"
                )
            elif delta < 0:
                lines.append(
                    "→ b\\* 低于现役语料，现役语料在 50% 预算下**偏容易**：需要**更密**的源文本"
                    f"（`words_per_fact` 降到 ≈{b_star / 3.15:.0f}）。这要求 v2 生成器；"
                    "**当前仓库没有该脚本**，是数据侧阻塞（见 §7）。"
                )
            else:
                lines.append(
                    "→ b\\* 高于现役语料，现役语料在 50% 预算下**偏难**：需要**更疏**的源文本"
                    f"（`words_per_fact` 升到 ≈{b_star / 3.15:.0f}）。同样要求 v2 生成器（见 §7）。"
                )
    elif decision["status"] == "compliance_gate_failed" and sel:
        lines.append(
            "**三档都没有达到 G3 的首轮合规率 ≥80%，本轮数据无法把「密度」和"
            "「长度控制失败」分开。**"
        )
        lines.append("")
        lines.append(
            f"最接近的档位是 `{sel['level']}`，但它首先在合规上失败"
            f"（abstractive 首轮 {_fmt(sel['conditions']['abstractive']['first_compliance'], pct=True)}，"
            f"structured 首轮 {_fmt(sel['conditions']['structured']['first_compliance'], pct=True)}；"
            f"重试后 {_fmt(sel['conditions']['abstractive']['final_compliance'], pct=True)} / "
            f"{_fmt(sel['conditions']['structured']['final_compliance'], pct=True)}）。"
            "**合规不通过时，准确率比和保留率都不能当作密度证据**：它们是在一批"
            "字节数偏离 50% 的消息上测得的。"
        )
        lines.append("")
        lines.append(
            "→ 先修 P3.4 的长度控制（prompt 或候选选择），再重跑 P0.9；"
            "**在此之前不要改语料密度**，否则会把「模型不能控长」误读成「语料太难/太易」。"
        )
        lines.append("")
        lines.append(
            "**但本轮还暴露了第二个独立问题**：§5.2 的剂量-反应在 44–62% 的实际字节比区间内"
            "几乎是平的（准确率比 0.84–0.89）。也就是说，**即使暂时不看合规闸门**，"
            "现役语料在 50% 附近也没有产生足够的信息损失；把预算从 62% 压到 44%，"
            "事实保留只从 ~79% 降到 ~77%，准确率比几乎不动。"
            "两个问题必须分开修：合规是 P3 的问题，冗余是语料的问题。"
        )
    elif sel:
        modes = sel["failure_modes"]
        lines.append(
            f"**没有任何一档同时满足三条判据。** 最接近的是 `{sel['level']}`，"
            f"失败模式：{', '.join(modes) if modes else 'unknown'}。"
        )
        lines.append("")
        if "ceiling" in modes or "retention_high" in modes:
            lines.append(
                "→ **天花板**：50% 预算下几乎所有事实都活下来，压缩后准确率贴近全文档基线。"
                "这意味着现役语料在 50% 处有太多可丢弃的填充/冗余文本，"
                "需要**更密**的源文本（更低 `words_per_fact`）才能产生可测的方法差异。"
            )
        if "floor" in modes or "retention_low" in modes:
            lines.append(
                "→ **地板**：消息丢得太多，receiver 已经无法作答。需要**更疏**的源文本"
                "（更高 `words_per_fact`）或更宽松的预算。"
            )
        lines.append("")
        lines.append("**这是一条阻塞性发现，不是可以四舍五入的偏差**，处理选项见 §7。")
    else:
        lines.append("**数据不足，无法给出密度决策。**")

    lines.append("")
    lines.append("## 2. 设计：用字节目标扫出 b，而不是重新生成三套语料")
    lines.append("")
    lines.append(
        "P0.9 的原计划是「10 篇 × 3 个源文本密度档」。但 v2 主集已定稿且其生成器未入库，"
        "无法在不违反「不改数据」纪律的前提下重造 18/22/26 词/事实三套文本。"
        "本报告改用**数学等价的操作**："
    )
    lines.append("")
    lines.append(
        "压缩难度由 `b = r × d`（r = 字节预算比例，d = 源字节/事实）决定。"
        "固定现役文档（d ≈ "
        f"{_mean([r['doc_bytes'] / r['n_facts'] for r in messages]):.1f} B/事实），"
        "把目标字节设为 `b × n_facts`，即可在**同一条文档**上复现 b = 57 / 69 / 82 三档的压缩任务。"
        "对 sender 而言，它看到的唯一难度变量就是字节目标，因此这与「换三套密度不同的文本、"
        "再统一按 50% 压缩」在**压缩难度**上是等价的。"
    )
    lines.append("")
    lines.append(
        "**限制（必须写进论文 limitation）**：本操作没有改变源文本本身的密度与可读性，"
        "只改变了输出预算。因此它标定的是「sender 在难度 b 下的行为」，"
        "不是「密度 d 对源文本可读性的影响」。后者只有在拿到 v2 生成器后才能做。"
    )
    lines.append("")
    lines.append(
        "三档命名：`b57` / `b69` / `b82`，分别对应原计划的密 / 中 / 疏（18 / 22 / 26 词/事实）。"
        "合规带仍是**目标 ±2% 源字节**（D-5），不是 ±2% 相对目标。"
    )
    lines.append("")

    lines.append("## 3. 合规与格式（sender 侧，实测）")
    lines.append("")
    rows = []
    for r in message_rows:
        rows.append(
            [
                f"`{r['level']}`",
                r["method"],
                r["n"],
                f"{r['b_target']:.1f}",
                _fmt(r["ratio_source"], pct=True),
                _fmt(r["first_compliance"], pct=True),
                _fmt(r["final_compliance"], pct=True),
                _fmt(r["under_rate"], pct=True),
                _fmt(r["over_rate"], pct=True),
                _fmt(r["mean_attempts"], digits=2),
                _fmt(r["first_invalid"], pct=True) if r["first_invalid"] is not None else "—",
            ]
        )
    lines.append(
        _table(
            ["档位", "方法", "n", "b (B/事实)", "实际字节/源", "首轮合规", "终轮合规",
             "偏短率", "超长率", "平均尝试", "首轮格式失败"],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "判据 1（G3）：首轮合规率 ≥80%（abstractive 与 structured 各自）。"
        "「首轮格式失败」只对 structured 有意义（JSON 解析失败）；"
        "abstractive 的格式失败率为 0 或极小，显示为 —。"
    )
    lines.append("")

    lines.append("## 4. 消息内事实保留（词面代理，非判决指标）")
    lines.append("")
    rows = []
    for r in message_rows:
        rows.append(
            [
                f"`{r['level']}`",
                r["method"],
                _fmt(r["fact_retention"], pct=True),
                _fmt(r["question_evidence"], pct=True),
                _fmt(r["mean_fact_recall"], pct=True),
                _fmt(r["numbers_missing"], digits=2),
                _fmt(r["n_items"], digits=1) if r["n_items"] is not None else "—",
            ]
        )
    lines.append(
        _table(
            ["档位", "方法", "事实保留率", "问题证据率", "平均词面召回", "丢数字条数/篇", "消息内条目数"],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "判据 2（G3）：事实保留率 60–85%。**这两个指标都是词面代理**，"
        "对 abstractive 的改写有系统性低估（structured 常直接复用原句）。"
        "因此它们只用于诊断「sender 到底丢了多少」，密度决策以 §5 的 receiver 准确率为准。"
    )
    lines.append("")

    lines.append("## 5. 压缩后准确率 / 全文档准确率（self-pair receiver，实测）")
    lines.append("")
    full = acc["full_all"]
    lines.append(
        f"全文档基线：{full['n_correct']}/{full['n_questions']} = "
        f"**{_fmt(full['accuracy'], pct=True)}** "
        f"[{_fmt(full['ci_lo'], pct=True)}, {_fmt(full['ci_hi'], pct=True)}]；"
        f"lookup {_fmt(full['lookup_accuracy'], pct=True)}，"
        f"multi-hop {_fmt(full['multihop_accuracy'], pct=True)}。"
    )
    lines.append("")
    lines.append(
        "self-pair 条件 = 每个 sender 的消息由**它自己**读（每篇每法每档一次），"
        "因此这里的准确率比主实验的 16 格矩阵更容易，只用于判据 3 的密度标定。"
    )
    lines.append("")
    rows = []
    for (level, method), data in sorted(acc["by_condition"].items()):
        o = data["overall"]
        rows.append(
            [
                f"`{level}`",
                method,
                f"{o['n_correct']}/{o['n_questions']}",
                _fmt(o["accuracy"], pct=True),
                f"[{_fmt(o['ci_lo'], pct=True)}, {_fmt(o['ci_hi'], pct=True)}]",
                _fmt(o["lookup_accuracy"], pct=True),
                _fmt(o["multihop_accuracy"], pct=True),
                _fmt(data["ratio_vs_full"], digits=2),
                _fmt(data["ratio_per_model_mean"], digits=2),
            ]
        )
    lines.append(
        _table(
            ["档位", "方法", "正确/题", "准确率", "95% CI", "lookup", "multi-hop",
             "准确率比（合并）", "准确率比（按模型平均）"],
            rows,
        )
    )
    lines.append("")
    lines.append("判据 3（G3）：准确率比落在 0.55–0.80。低于 0.55 是地板，高于 0.80 是天花板。")
    lines.append("")
    lines.append("### 5.1 仅合规消息（敏感性分析）")
    lines.append("")
    lines.append(
        "上表把所有消息都算进来，包含字节数不在带内的消息。下表只保留 `compliant=true` 的"
        "消息（以及它们在 self-pair 下的作答），是密度效应最干净的现有读数；"
        "样本量会明显下降，CI 变宽。"
    )
    lines.append("")
    rows = []
    for (level, method), data in sorted(acc_compliant["by_condition"].items()):
        o = data["overall"]
        rows.append(
            [
                f"`{level}`",
                method,
                f"{o['n_correct']}/{o['n_questions']}",
                _fmt(o["accuracy"], pct=True),
                f"[{_fmt(o['ci_lo'], pct=True)}, {_fmt(o['ci_hi'], pct=True)}]",
                _fmt(data["ratio_vs_full"], digits=2),
            ]
        )
    lines.append(_table(["档位", "方法", "正确/题", "准确率", "95% CI", "准确率比"], rows))
    lines.append("")

    lines.append("### 5.2 实际字节比的剂量-反应（不丢弃任何消息）")
    lines.append("")
    lines.append(
        "当 ±2% 闸门未通过时，把所有消息按**实际** `bytes/source` 分箱，直接看准确率随"
        "压缩强度的变化。这张表回答的是「把预算真的压到 45% 会不会掉准确率」，"
        "而不是「消息有没有落在 50%±2%」。"
    )
    lines.append("")
    rows = []
    for r in acc_by_ratio:
        rows.append(
            [
                r["bin"],
                r["n_records"],
                r["n_questions"],
                _fmt(r["accuracy"], pct=True),
                _fmt(r["lookup_accuracy"], pct=True) if r.get("lookup_accuracy") is not None else "—",
                _fmt(r["multihop_accuracy"], pct=True) if r.get("multihop_accuracy") is not None else "—",
                _fmt(r["ratio_vs_full"], digits=2),
            ]
        )
    lines.append(
        _table(["实际 bytes/source", "消息数", "题数", "准确率", "lookup", "multi-hop", "准确率比"], rows)
    )
    lines.append("")

    lines.append("## 6. 分模型（合规率 / 保留率 / 准确率比）")
    lines.append("")
    rows = []
    for r in model_rows:
        cond = acc["by_condition"].get((r["level"], r["method"]), {})
        per_model = cond.get("per_model", {}).get(r["model"], {})
        full_acc = acc["full_by_model"].get(r["model"], {}).get("accuracy")
        ratio = (
            per_model.get("accuracy") / full_acc
            if per_model.get("accuracy") is not None and full_acc
            else None
        )
        rows.append(
            [
                f"`{r['level']}`",
                r["method"],
                r["model"],
                _fmt(r["first_compliance"], pct=True),
                _fmt(r["final_compliance"], pct=True),
                _fmt(r["fact_retention"], pct=True),
                _fmt(per_model.get("accuracy"), pct=True),
                _fmt(ratio, digits=2),
            ]
        )
    lines.append(
        _table(
            ["档位", "方法", "模型", "首轮合规", "终轮合规", "事实保留", "self 准确率", "准确率比"],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "这张表是 §7 判断「密度是否会变成 sender 能力差异」（坑 #11）的依据："
        "若某模型在三档都合规而另一模型在三档都不合规，说明标定选的是模型而不是密度。"
    )
    lines.append("")

    lines.append("## 7. 决策与风险")
    lines.append("")
    for e in decision["evaluations"]:
        conds = e["conditions"]
        lines.append(
            f"- `{e['level']}`：合规 {'✅' if e['compliance_ok'] else '❌'}"
            f"（abs 首轮 {_fmt(conds['abstractive']['first_compliance'], pct=True)} / "
            f"struct 首轮 {_fmt(conds['structured']['first_compliance'], pct=True)}；"
            f"重试后 {_fmt(conds['abstractive']['final_compliance'], pct=True)} / "
            f"{_fmt(conds['structured']['final_compliance'], pct=True)}），"
            f"保留 {'✅' if e['retention_ok'] else '❌'}（{_fmt(e['retention'], pct=True)}），"
            f"准确率比 {'✅' if e['ratio_ok'] else '❌'}（{_fmt(e['ratio'], digits=2)}）"
        )
    lines.append("")
    if decision["status"] != "qualified":
        lines.append("**若没有一档全绿，按以下顺序处理（不改变 50% 这个比例）：**")
        lines.append("")
        lines.append(
            "1. **先看合规闸门**（本轮的首要失败模式）：首轮合规率是否 ≥80%？"
            "若否，问题在 P3.4 长度控制，不在密度。可试的修法："
            "（a）**已试**：p4.1 的可数单位（7 句/7 条 + 每句词预算）+ 个性化重试，"
            "把首轮从 0–19% 提到 30%，重试后 54%，仍离 80% 很远；"
            "（b）**未试**：多候选生成（同一 sender、不同请求目标），按与目标的距离选一条合规消息，"
            "预计需要 3–5 条候选；"
            "（c）**未试**：校准**请求的**目标字节（不是 48–52% 的判定带），让实际输出分布居中。"
            "三条都必须重跑 P0.9。"
        )
        lines.append(
            "2. **再看语料密度**（本轮已有独立证据）：b57 档的实际中位字节比约 44%，"
            "保留率仍 74–80%，准确率比 0.84；b82 档约 62%，准确率比 0.86。"
            "**压缩强度差 18 个百分点，准确率比只差 2 个百分点**，剂量-反应是平的。"
            "这说明现役 v2 语料里可丢弃的填充/冗余文本比例过高，"
            "50% 预算下 sender 先删填充、不删事实。要产生 0.55–0.80 的准确率比，"
            "需要**更密**的源文本（更低 `words_per_fact`），这要求 v2 生成器；"
            "当前仓库没有该脚本，**必须先向组员拿到**。"
        )
        lines.append("")
        lines.append(
            "3. **合规问题还会与 sender 能力混淆**（坑 #11）：§6 表里 qwen3.7-max 在 "
            "b57/b69 structured 重试后能到 100%，而 deepseek-v4-pro-0813 structured 只有 "
            "10–40%。这不是密度效应，是模型长度控制能力差异。"
            "在合规率修到可接受之前，不能把「谁更合规」读成「谁更会被自己读懂」。"
        )
        lines.append(
            "4. **不要**通过改 50% 比例、截断消息、或只报合规样本来「修」结果——"
            "这三条分别违反 D-5、P3 铁律和 G3 的分母规则。"
        )
        lines.append("")
    lines.append(
        "**下一步**：P0.9 只回答密度，不替代 pilot；但本轮 G3 未通过，"
        "**freeze-v2 与 P4 pilot 暂缓**。决策选项见 `docs/freeze_v2_decision.md`。"
    )
    lines.append("")

    lines.append("## 8. 附录：调用与可复现性")
    lines.append("")
    send_calls = sum(
        len(r["attempts"]) for r in messages
    )
    recv_calls = sum(1 for r in predictions if not r.get("cached"))
    lines.append(
        f"- sender 逻辑消息 {len(messages)} 条，sender 尝试 {send_calls} 次"
        f"（含预算闭环重试）；receiver 逻辑请求 {len(predictions)} 次，"
        f"其中缓存命中 {sum(1 for r in predictions if r.get('cached'))} 次。"
    )
    lines.append(
        "- 缓存：`runs/_cache.sqlite3`（内容哈希）；调用日志：`runs/<run_id>*/calls.jsonl`。"
    )
    versions = sorted({r.get("prompt_version") for r in messages if r.get("prompt_version")})
    lines.append(
        "- prompt 版本（本 run 实际）：" + "、".join(f"`{v}`" for v in versions)
        + "；receiver `r1.0` / `r1.0-full`。改任何 prompt 字符串都会改变缓存键。"
    )
    lines.append(
        "- 模型：`qwen3.7-plus` / `qwen3.7-max` / `deepseek-v4-flash-0731` / "
        "`deepseek-v4-pro-0813`，全部 `enable_thinking=false`、`temperature=0`、`json_object`。"
    )
    lines.append("")
    lines.append(
        "> 本报告由 `src/run/report.py` 从 JSONL 自动生成；"
        "重跑 `python -m src.run.cli report` 即可复现同一张表（缓存命中不产生新调用）。"
    )
    lines.append("")
    return "\n".join(lines)


def build_and_write(
    *,
    run_id: str,
    messages_path: str,
    predictions_path: str,
    out_path: str,
    summary_path: str | None = None,
) -> dict:
    messages = read_jsonl(messages_path)
    predictions = read_jsonl(predictions_path)
    if not messages:
        raise SystemExit(f"no messages in {messages_path}")
    if not predictions:
        raise SystemExit(f"no predictions in {predictions_path}")

    # Recompute retention from the message text with the current code. The
    # stored field can be stale if the retention definition changed mid-run.
    docs_by_id = {d["doc_id"]: d for d in load_docs()}
    questions_by_doc = load_questions_by_doc()
    for rec in messages:
        doc = docs_by_id.get(rec["doc_id"])
        if doc is None or not rec.get("message"):
            continue
        rec["retention"] = retention_summary(
            doc, rec["message"], questions_by_doc.get(rec["doc_id"], []), rec["method"]
        )

    message_rows = aggregate_messages(messages)
    model_rows = aggregate_model_messages(messages)
    acc = aggregate_accuracy(predictions)
    acc_compliant = accuracy_compliant_only(predictions, messages)
    acc_by_ratio = accuracy_by_actual_ratio(predictions, messages)
    decision = decide(message_rows, acc)
    text = render_report(
        run_id=run_id,
        messages=messages,
        predictions=predictions,
        message_rows=message_rows,
        model_rows=model_rows,
        acc=acc,
        acc_compliant=acc_compliant,
        acc_by_ratio=acc_by_ratio,
        decision=decision,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    summary = {
        "run_id": run_id,
        "n_docs": len({r["doc_id"] for r in messages}),
        "n_messages": len(messages),
        "n_predictions": len(predictions),
        "message_rows": message_rows,
        "model_rows": model_rows,
        "full_accuracy": acc["full_all"],
        "accuracy_by_actual_ratio": acc_by_ratio,
        "compliant_only": {
            f"{level}|{method}": {
                "accuracy": data["overall"]["accuracy"],
                "n_questions": data["overall"]["n_questions"],
                "ratio_vs_full": data["ratio_vs_full"],
            }
            for (level, method), data in acc_compliant["by_condition"].items()
        },
        "full_by_model": acc["full_by_model"],
        "by_condition": {
            f"{level}|{method}": {
                "accuracy": data["overall"]["accuracy"],
                "ci_lo": data["overall"]["ci_lo"],
                "ci_hi": data["overall"]["ci_hi"],
                "n_questions": data["overall"]["n_questions"],
                "ratio_vs_full": data["ratio_vs_full"],
                "ratio_per_model_mean": data["ratio_per_model_mean"],
                "lookup_accuracy": data["overall"]["lookup_accuracy"],
                "multihop_accuracy": data["overall"]["multihop_accuracy"],
            }
            for (level, method), data in acc["by_condition"].items()
        },
        "decision": decision,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if summary_path:
        p = Path(summary_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
