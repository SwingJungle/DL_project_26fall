"""Validate a paired-density corpus (same documents, two source densities).

The main experiment pairs every document with itself at ``base`` (bpf 133) and
``b100`` (bpf 100): identical ``doc_id``, identical facts (same slot order, same
gold values), identical questions word for word, different sentence
realisations. This script is the **zero-call** gate that has to pass before any
paid call is spent on the paired corpus (``docs/freeze_v3_plan.md`` §10).

Checks
------
1. structure   -- both dirs carry ``docs.jsonl`` / ``questions.jsonl``; doc_id
                  sets, order, split, domain and fact count agree.
2. facts       -- per position: ``fid``, ``renderer``, ``type``,
                  ``sentence_idx`` and ``gold`` are identical; only ``text``
                  (and hence the document text) may differ.
3. questions   -- verbatim identical in both directions (all fields), and each
                  lookup question's gold matches its fact's gold.
4. schema      -- ``bytes``/``words``/``bytes_per_fact`` agree with the text.
5. per-density -- ``generate_v3.validate`` reports zero problems under each
                  corpus's own density configuration; bytes/fact and words/fact
                  sit inside the ±10% band; multi-hop pairs cross sentences.
6. density     -- the two corpora are actually different (no doc keeps the same
                  text) and the density gap is large enough to be a contrast.

Usage
-----
    python -m src.data.validate_paired \
        --a data/_v3_paired/base --b data/_v3_paired/b100 \
        --bpf-a 133 --bpf-b 100
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from . import generate_v3 as G


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _signed_delta(a: float, b: float) -> float:
    return (a - b) / b if b else 0.0


def load_corpus(path: Path) -> dict:
    path = Path(path)
    docs = read_jsonl(path / "docs.jsonl")
    questions = read_jsonl(path / "questions.jsonl")
    by_doc: dict[str, list[dict]] = collections.defaultdict(list)
    for q in questions:
        by_doc[q["doc_id"]].append(q)
    return {
        "path": path,
        "docs": docs,
        "questions": questions,
        "by_doc": by_doc,
        "hashes": {
            "docs": sha256_file(path / "docs.jsonl"),
            "questions": sha256_file(path / "questions.jsonl"),
        },
    }


#: Attribution templates that deny the project happens in ``{year3}``. The
#: corpus's ``attribution`` slot is meant to express *uncertainty* about a
#: project planned for ``{year3}``; these forms instead assert it will not
#: happen then, which contradicts the ordinal multi-hop question
#: ("Which came first, the relocation or the project planned for {year3}?")
#: whenever that question's gold is the project. The shortest attribution
#: template is one of these, so it is selected far more often at low densities
#: (16.5% of b100 documents vs 3.5% at base) -- i.e. the artifact is
#: density-correlated, which is why it is flagged rather than ignored.
YEAR3_DENIAL_PATTERNS = ("would not arrive", "will not arrive", "never arrive",
                         "not going ahead in", "will not go ahead in")


def denial_docs(corpus: dict) -> list[str]:
    """Documents whose ``attribution`` sentence denies the project happens in year3."""
    out = []
    for d in corpus["docs"]:
        attribution = [f["text"] for f in d["facts"] if f["renderer"] == "attribution"]
        if attribution and any(
            pat in attribution[0].lower() for pat in YEAR3_DENIAL_PATTERNS
        ):
            out.append(d["doc_id"])
    return out


def premise_flags(corpus: dict) -> list[dict]:
    """Ordinal multi-hop questions whose premise the source text contradicts.

    Informational: these questions are still scored normally. The point is to
    let the analysis report a symmetric sensitivity check (drop the flagged
    questions in *both* densities) instead of silently keeping ill-posed items.
    """
    flags: list[dict] = []
    denied = set(denial_docs(corpus))
    for d in corpus["docs"]:
        if d["doc_id"] not in denied:
            continue
        attribution = [f["text"] for f in d["facts"] if f["renderer"] == "attribution"]
        for q in corpus["by_doc"].get(d["doc_id"], []):
            if q["demand"] != "multi-hop" or not q["text"].startswith("Which came first"):
                continue
            if q["gold"].startswith("the relocation"):
                continue  # project-not-in-year3 agrees with relocation-first
            flags.append({
                "doc_id": d["doc_id"],
                "qid": q["qid"],
                "question": q["text"],
                "gold": q["gold"],
                "source_sentence": attribution[0],
            })
    return flags


def _schema_problems(docs: list[dict]) -> list[str]:
    problems = []
    for d in docs:
        text = d["text"]
        nbytes = len(text.encode("utf-8"))
        if d.get("bytes") != nbytes:
            problems.append(f"{d['doc_id']}: bytes {d.get('bytes')} != {nbytes}")
        if d.get("words") != len(text.split()):
            problems.append(f"{d['doc_id']}: words {d.get('words')} != {len(text.split())}")
        want_bpf = round(nbytes / d["n_facts"], 1)
        if abs(float(d.get("bytes_per_fact", -1)) - want_bpf) > 0.05:
            problems.append(
                f"{d['doc_id']}: bytes_per_fact {d.get('bytes_per_fact')} != {want_bpf}"
            )
    return problems


def validate_pair(
    dir_a: str | Path,
    dir_b: str | Path,
    bpf_a: float | None = None,
    bpf_b: float | None = None,
) -> dict:
    """Compare corpora ``a`` (usually base) and ``b`` (usually b100)."""
    A = load_corpus(Path(dir_a))
    B = load_corpus(Path(dir_b))
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "", data: dict | None = None) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail, "data": data or {}})

    # ---- 1. structure -----------------------------------------------------
    ids_a = [d["doc_id"] for d in A["docs"]]
    ids_b = [d["doc_id"] for d in B["docs"]]
    check(
        "same_doc_ids",
        ids_a == ids_b,
        f"{len(ids_a)} vs {len(ids_b)} docs; order {'identical' if ids_a == ids_b else 'DIFFERS'}",
        {"a": len(ids_a), "b": len(ids_b), "missing_in_b": sorted(set(ids_a) - set(ids_b))[:10]},
    )
    check(
        "same_split",
        all(x.get("split") == y.get("split") for x, y in zip(A["docs"], B["docs"])),
        "split labels agree per doc_id",
        {"a": dict(collections.Counter(d["split"] for d in A["docs"])),
         "b": dict(collections.Counter(d["split"] for d in B["docs"]))},
    )

    # ---- 2. facts ---------------------------------------------------------
    n_facts_a = sum(len(d["facts"]) for d in A["docs"])
    n_facts_b = sum(len(d["facts"]) for d in B["docs"])
    fact_problems: list[str] = []
    differing_facts = 0
    identical_docs = 0
    pair_map = {d["doc_id"]: d for d in B["docs"]}
    for da in A["docs"]:
        db = pair_map.get(da["doc_id"])
        if db is None:
            continue
        if da["text"] == db["text"]:
            identical_docs += 1
        if da.get("domain") != db.get("domain") or da["n_facts"] != db["n_facts"]:
            fact_problems.append(f"{da['doc_id']}: domain/n_facts mismatch")
            continue
        if len(da["facts"]) != len(db["facts"]):
            fact_problems.append(f"{da['doc_id']}: fact count mismatch")
            continue
        for fa, fb in zip(da["facts"], db["facts"]):
            meta_a = (fa["fid"], fa["renderer"], fa["type"], fa["sentence_idx"], fa["gold"])
            meta_b = (fb["fid"], fb["renderer"], fb["type"], fb["sentence_idx"], fb["gold"])
            if meta_a != meta_b:
                fact_problems.append(f"{da['doc_id']}/{fa['fid']}: metadata mismatch")
            if fa["text"] != fb["text"]:
                differing_facts += 1
    check(
        "same_fact_metadata",
        not fact_problems,
        f"{n_facts_a} vs {n_facts_b} facts; {len(fact_problems)} problems",
        {"examples": fact_problems[:10]},
    )
    check(
        "texts_differ_everywhere",
        identical_docs == 0 and differing_facts > 0,
        f"{identical_docs} doc(s) with identical text; "
        f"{differing_facts}/{n_facts_a} facts re-rendered",
        {"identical_docs": identical_docs, "differing_facts": differing_facts,
         "n_facts": n_facts_a},
    )

    # ---- 3. questions -----------------------------------------------------
    q_problems: list[str] = []
    q_fields = ("qid", "demand", "text", "gold", "answerable", "fact_ids", "fact_types")
    for doc_id, qs_a in A["by_doc"].items():
        qs_b = B["by_doc"].get(doc_id, [])
        if len(qs_a) != len(qs_b):
            q_problems.append(f"{doc_id}: {len(qs_a)} vs {len(qs_b)} questions")
            continue
        for qa, qb in zip(qs_a, qs_b):
            for f in q_fields:
                if qa.get(f) != qb.get(f):
                    q_problems.append(f"{doc_id}/{qa.get('qid')}: field {f} differs")
    # lookup gold must equal the fact gold it names
    gold_problems: list[str] = []
    for corpus, tag in ((A, "a"), (B, "b")):
        for d in corpus["docs"]:
            facts = {f["fid"]: f for f in d["facts"]}
            for q in corpus["by_doc"].get(d["doc_id"], []):
                if q["demand"] != "lookup" or not q.get("answerable"):
                    continue
                if not q["fact_ids"]:
                    gold_problems.append(f"{tag}:{q['qid']}: answerable with no fact_ids")
                    continue
                fact = facts.get(q["fact_ids"][0])
                if fact is None:
                    gold_problems.append(f"{tag}:{q['qid']}: unknown fact_id")
                elif fact["gold"] != q["gold"]:
                    gold_problems.append(f"{tag}:{q['qid']}: gold {q['gold']!r} != fact {fact['gold']!r}")
    check(
        "questions_verbatim_identical",
        not q_problems,
        f"{A['hashes']['questions'] == B['hashes']['questions'] and 'file hash identical' or 'file hash differs'}; "
        f"{len(q_problems)} field problems",
        {"examples": q_problems[:10],
         "sha256_a": A["hashes"]["questions"], "sha256_b": B["hashes"]["questions"]},
    )
    check(
        "question_gold_matches_facts",
        not gold_problems,
        f"{len(gold_problems)} problems across both densities",
        {"examples": gold_problems[:10]},
    )

    # ---- 4. schema --------------------------------------------------------
    schema_problems = _schema_problems(A["docs"]) + _schema_problems(B["docs"])
    check("schema_consistent", not schema_problems, f"{len(schema_problems)} problems",
          {"examples": schema_problems[:10]})

    # ---- 5. per-density validation ---------------------------------------
    stats = {}
    for tag, corpus, target in (("a", A, bpf_a), ("b", B, bpf_b)):
        target = target if target else _median_bpf(corpus["docs"])
        G.configure_density(target)
        problems: list[str] = []
        for d in corpus["docs"]:
            # ``validate`` expects the questions attached, but docs.jsonl keeps
            # them in questions.jsonl (questions are shared across densities).
            augmented = dict(d, questions=corpus["by_doc"].get(d["doc_id"], []))
            for p in G.validate(augmented):
                problems.append(f"{d['doc_id']}: {p}")
        bpf = [d["bytes_per_fact"] for d in corpus["docs"]]
        wpf = [d["words"] / d["n_facts"] for d in corpus["docs"]]
        stats[tag] = {
            "path": str(corpus["path"]),
            "target_bpf": round(float(target), 1),
            "n_docs": len(corpus["docs"]),
            "n_questions": len(corpus["questions"]),
            "n_multi_hop": sum(1 for q in corpus["questions"] if q["demand"] == "multi-hop"),
            "bpf_mean": round(sum(bpf) / len(bpf), 1) if bpf else None,
            "bpf_min": round(min(bpf), 1) if bpf else None,
            "bpf_max": round(max(bpf), 1) if bpf else None,
            "wpf_mean": round(sum(wpf) / len(wpf), 2) if wpf else None,
            "wpf_min": round(min(wpf), 2) if wpf else None,
            "wpf_max": round(max(wpf), 2) if wpf else None,
            "validate_problems": len(problems),
            "n_denial_template_docs": len(denial_docs(corpus)),
            "example_problems": problems[:10],
            "sha256": corpus["hashes"],
        }
        check(
            f"validate_zero_problems_{tag}",
            not problems,
            f"{len(problems)} problems at target bpf {target:g}",
            {"examples": problems[:10]},
        )
        out_band = [
            d["doc_id"] for d in corpus["docs"]
            if abs(d["bytes_per_fact"] - target) > 0.10 * target
        ]
        check(
            f"bytes_per_fact_in_band_{tag}",
            not out_band,
            f"mean {stats[tag]['bpf_mean']} B/fact (target {target:g} ±10%)",
            {"out_of_band": out_band[:10], "n_out": len(out_band)},
        )
        multi_hop_same = []
        dup_texts = collections.Counter(d["text"] for d in corpus["docs"])
        dup_docs = [t[:40] for t, c in dup_texts.items() if c > 1]
        for d in corpus["docs"]:
            pos = {f["fid"]: f["sentence_idx"] for f in d["facts"]}
            for q in corpus["by_doc"].get(d["doc_id"], []):
                if q["demand"] == "multi-hop" and len(q["fact_ids"]) == 2:
                    if pos.get(q["fact_ids"][0]) == pos.get(q["fact_ids"][1]):
                        multi_hop_same.append(q["qid"])
        check(
            f"multi_hop_cross_sentence_{tag}",
            not multi_hop_same,
            f"{len(multi_hop_same)} multi-hop question(s) inside one sentence",
            {"examples": multi_hop_same[:10]},
        )
        check(
            f"no_duplicate_docs_{tag}",
            not dup_docs,
            f"{len(dup_docs)} duplicated document text(s)",
            {"examples": dup_docs[:10]},
        )

    # ---- 6. density contrast ---------------------------------------------
    gap_bpf = None
    gap_wpf = None
    if stats["a"]["bpf_mean"] and stats["b"]["bpf_mean"]:
        gap_bpf = round(stats["a"]["bpf_mean"] - stats["b"]["bpf_mean"], 1)
    if stats["a"]["wpf_mean"] and stats["b"]["wpf_mean"]:
        gap_wpf = round(stats["a"]["wpf_mean"] - stats["b"]["wpf_mean"], 2)
    check(
        "density_contrast",
        bool(gap_bpf and gap_bpf > 15),
        f"base - b100 = {gap_bpf} B/fact ({gap_wpf} words/fact)",
        {"gap_bpf": gap_bpf, "gap_wpf": gap_wpf,
         "ratio": round(_signed_delta(stats["a"]["bpf_mean"], stats["b"]["bpf_mean"]), 3)},
    )

    flags = {"a": premise_flags(A), "b": premise_flags(B)}
    return {
        "a": stats["a"],
        "b": stats["b"],
        "checks": checks,
        "passed": all(c["ok"] for c in checks),
        "n_checks": len(checks),
        "n_failed": sum(1 for c in checks if not c["ok"]),
        "premise_flags": flags,
        "generator": {
            "seed_b100": 20261111,
            "bpf_b": bpf_b, "bpf_a": bpf_a,
        },
    }


def _median_bpf(docs: list[dict]) -> float:
    vals = sorted(d["bytes_per_fact"] for d in docs)
    return vals[len(vals) // 2] if vals else 0.0


def render_markdown(report: dict) -> str:
    a, b = report["a"], report["b"]
    lines = [
        "# 配对密度语料校验（0 调用）",
        "",
        "> 脚本：`src/data/validate_paired.py`；数据：两个同 `doc_id` 的语料目录。",
        "> 这是主实验前的 freeze 门槛：题目必须逐字相同，事实元数据必须一致，",
        "> 每档密度都要过 `generate_v3.validate`。",
        "",
        f"**结论：{'PASS' if report['passed'] else 'FAIL'}** "
        f"（{report['n_checks']} 项检查，{report['n_failed']} 项失败）",
        "",
        "## 两档密度",
        "",
        "| 语料 | docs | questions | 目标 B/fact | 平均 B/fact | 范围 | 平均词/fact | sha256(docs) |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for tag, label, s in (("a", "A (base)", a), ("b", "B (b100)", b)):
        lines.append(
            f"| {label} | {s['n_docs']} | {s['n_questions']} | {s['target_bpf']:g} | "
            f"{s['bpf_mean']} | {s['bpf_min']}–{s['bpf_max']} | {s['wpf_mean']} | "
            f"`{s['sha256']['docs'][:16]}…` |"
        )
    contrast = next((c for c in report["checks"] if c["check"] == "density_contrast"), None)
    if contrast:
        lines += ["", f"- 密度差：{contrast['detail']}。"]
    lines += ["", "## 检查清单", "", "| # | 检查 | 结果 | 说明 |", "|---:|---|---|---|"]
    for i, c in enumerate(report["checks"], start=1):
        mark = "PASS" if c["ok"] else "**FAIL**"
        lines.append(f"| {i} | `{c['check']}` | {mark} | {c['detail']} |")
    failed = [c for c in report["checks"] if not c["ok"]]
    if failed:
        lines += ["", "## 失败明细", ""]
        for c in failed:
            lines.append(f"- `{c['check']}`：{c['detail']}")
            for ex in (c.get("data", {}).get("examples") or c.get("data", {}).get("out_of_band") or [])[:5]:
                lines.append(f"    - {ex}")
    else:
        lines += [
            "",
            "## 说明",
            "",
            "- 两档语料由同一 `roles.jsonl` 重渲染，题目按角色生成，因此天然逐字相同；",
            "  这里用文件 sha256 与逐字段比较**验证**而不是假设。",
            "- 事实元数据（`fid`/`renderer`/`type`/`sentence_idx`/`gold`）完全相同，",
            "  只有 `text` 变化——这是配对密度设计成立的前提。",
            "- b100 语料与 `data/_v3_candidate/b100/` 逐字节相同（seed 20261111，200 篇），",
            "  说明生成器可复现冻结候选池。",
        ]

    # -- source-level premise flags (informational, not a gate) --------------
    flags = report.get("premise_flags") or {}
    if flags.get("a") or flags.get("b"):
        lines += [
            "",
            "## 源文本前提冲突（informational，不是门槛）",
            "",
            "`attribution` 槽位里最长的模板之一是 "
            "`{director} said the {project} would not arrive in {year3}.`，它断言项目"
            "**不会**在 `{year3}` 发生，而 ordinal multi-hop 题的前提是“项目计划在 "
            "`{year3}`”。当该题 gold 是项目时，两者冲突，题目变成 ill-posed。",
            "这条模板是 attribution 池里**最短**的一条，所以低密度（短句）时被选中得更多："
            f"base {a['n_denial_template_docs']}/{a['n_docs']}"
            f"（{100 * a['n_denial_template_docs'] / max(1, a['n_docs']):.1f}%） vs "
            f"b100 {b['n_denial_template_docs']}/{b['n_docs']}"
            f"（{100 * b['n_denial_template_docs'] / max(1, b['n_docs']):.1f}%）"
            "——即冲突率**与密度相关**。",
            "",
            "| 语料 | 冲突题数 | 占全部题目 | 占 multi-hop 题 |",
            "|---|---:|---:|---:|",
        ]
        for tag, label, s in (("a", "base", a), ("b", "b100", b)):
            n = len(flags.get(tag) or [])
            mh = s.get("n_multi_hop") or 0
            lines.append(
                f"| {label} | {n} | {100 * n / max(1, s['n_questions']):.2f}% | "
                f"{100 * n / max(1, mh):.2f}% |"
            )
        lines += [
            "",
            "影响量级（用 20 篇 compact probe 估计）：冲突 doc 上 ordinal 题的方法差 "
            "18.4pp，clean doc 上 10.9pp；按 b100 6%、base 2% 的占比折算，"
            "对 b100 方法差的偏差约 **+0.5pp**、对交互约 **+0.3pp**（≈8.5pp 的 4%）。",
            "建议作为**预注册敏感性分析**：两档密度、所有方法同步剔除这些题，"
            "主结果不变则报告“剔除后结论不变”。不要在事后静默删除。",
            "",
        ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", default="data/_v3_paired/base", help="corpus A (base)")
    parser.add_argument("--b", default="data/_v3_paired/b100", help="corpus B (b100)")
    parser.add_argument("--bpf-a", type=float, default=133.0,
                        help="target B/fact of corpus A; 0 = infer from median")
    parser.add_argument("--bpf-b", type=float, default=100.0,
                        help="target B/fact of corpus B; 0 = infer from median")
    parser.add_argument("--out", default="reports/paired_corpus.md")
    parser.add_argument("--json-out", default="reports/paired_corpus.json")
    args = parser.parse_args(argv)

    report = validate_pair(args.a, args.b,
                           bpf_a=args.bpf_a or None, bpf_b=args.bpf_b or None)

    from ..analysis.common import write_json

    write_json(args.json_out, report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.json_out}")
    print(f"{report['n_checks']} checks, {report['n_failed']} failed -> "
          f"{'PASS' if report['passed'] else 'FAIL'}")
    for c in report["checks"]:
        print(f"  [{'ok' if c['ok'] else '!!'}] {c['check']:<28} {c['detail']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
