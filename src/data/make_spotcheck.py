"""Build the human spot-check sheet for the paired v3 corpus (zero calls).

``docs/freeze_v3_plan.md`` §10 still has "A 人工抽检 20 篇 v3 短模板" open. The
sheet makes that cheap: the same document at both densities, side by side, with
the questions and golds the sender will actually be judged on, plus any
premise flag the validator raised.

Usage
-----
    python -m src.data.make_spotcheck --n 20 --out reports/spotcheck_v3.md
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .validate_paired import premise_flags, read_jsonl


def load(path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    docs = read_jsonl(path / "docs.jsonl")
    by_doc: dict[str, list[dict]] = collections.defaultdict(list)
    for q in read_jsonl(path / "questions.jsonl"):
        by_doc[q["doc_id"]].append(q)
    return docs, by_doc


def sample_ids(docs: list[dict], by_doc: dict, n: int, flags: list[dict]) -> list[str]:
    """Evenly spaced sample, plus every document with a flagged question."""
    ids = [d["doc_id"] for d in docs]
    if n >= len(ids):
        chosen = list(ids)
    else:
        step = len(ids) / n
        chosen = [ids[min(len(ids) - 1, int(i * step))] for i in range(n)]
    flagged = {f["doc_id"] for f in flags}
    for doc_id in ids:
        if doc_id in flagged and doc_id not in chosen:
            chosen.append(doc_id)
    return chosen


def render_doc(doc: dict, base: dict, questions: list[dict], flags: list[dict]) -> list[str]:
    qflag = {f["qid"] for f in flags}
    lines = [
        f"### `{doc['doc_id']}` · {doc['domain']} · {doc['split']} · {doc['n_facts']} facts",
        "",
        f"- b100：{doc['words']} 词 / {doc['bytes']} B / **{doc['bytes_per_fact']} B/事实**",
        f"- base：{base['words']} 词 / {base['bytes']} B / **{base['bytes_per_fact']} B/事实**",
        "",
        "**b100 源文本**",
        "",
        "> " + doc["text"],
        "",
        "**base 源文本**",
        "",
        "> " + base["text"],
        "",
        "**题目（两档密度共用，逐字相同）**",
        "",
    ]
    for q in questions:
        mark = " ⚠ 前提冲突" if q["qid"] in qflag else ""
        lines.append(f"- `{q['qid']}` [{q['demand']}] {q['text']} → **{q['gold']}**{mark}")
    lines.append("")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="data/_v3_paired/base")
    parser.add_argument("--dense", default="data/_v3_paired/b100")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", default="reports/spotcheck_v3.md")
    args = parser.parse_args(argv)

    base_docs, base_qs = load(Path(args.base))
    dense_docs, dense_qs = load(Path(args.dense))
    dense_by_id = {d["doc_id"]: d for d in dense_docs}
    base_by_id = {d["doc_id"]: d for d in base_docs}

    flags = premise_flags({"docs": dense_docs, "by_doc": dense_qs})
    chosen = sample_ids(dense_docs, dense_qs, args.n, flags)

    lines = [
        "# v3 配对语料人工抽检表（A 抽检用，0 调用生成）",
        "",
        "> 脚本：`src/data/make_spotcheck.py`。抽样：语料内在等距抽取 "
        f"{args.n} 篇，另加所有被标记为“前提冲突”的文档。",
        "> 检查重点：**b100 的短模板是否仍然语法通顺、语义等于 base 的同一条事实**；",
        "> 题目与 gold 是否仍然成立。逐条对照 base 与 b100 即可。",
        "",
        "## 速查：语料规格",
        "",
        "| 项 | base | b100 |",
        "|---|---:|---:|",
    ]
    for label, key in (("文档数", "n"), ("B/事实（均值）", "bpf"), ("词/事实（均值）", "wpf")):
        vals = []
        for docs in (base_docs, dense_docs):
            if key == "n":
                vals.append(str(len(docs)))
            elif key == "bpf":
                vals.append(f"{sum(d['bytes_per_fact'] for d in docs) / len(docs):.1f}")
            else:
                vals.append(f"{sum(d['words'] / d['n_facts'] for d in docs) / len(docs):.2f}")
        lines.append(f"| {label} | {vals[0]} | {vals[1]} |")
    lines += [
        "",
        f"注明：主实验冻结口径里的 “20.4 / 15.4 词/事实” 是 `B/事实 ÷ 6.3` 的标称值，"
        f"实测词数约为 base 21.0 / b100 16.1，撰写报告时应在设计表里写实测值或注明换算。",
        "",
        f"本篇抽检共 {len(chosen)} 篇；标 ⚠ 的题目是源文本与题干前提不一致的题目"
        f"（应作为预注册敏感性分析剔除，不要静默删）。",
        "",
        "---",
        "",
    ]
    for doc_id in chosen:
        dense = dense_by_id[doc_id]
        base = base_by_id[doc_id]
        doc_flags = [f for f in flags if f["doc_id"] == doc_id]
        lines += render_doc(dense, base, dense_qs[doc_id], doc_flags)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(chosen)} docs, {len(flags)} flagged questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
