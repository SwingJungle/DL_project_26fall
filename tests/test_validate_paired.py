"""Unit tests for the paired-density corpus validator (``src.data.validate_paired``)."""
import json
import tempfile
import unittest
from pathlib import Path

from src.data.validate_paired import premise_flags, validate_pair


def _fact(fid, renderer, typ, idx, text, gold):
    return {"fid": fid, "renderer": renderer, "type": typ,
            "sentence_idx": idx, "text": text, "gold": gold}


def _doc(doc_id, text, facts, *, split="eval", domain="institute"):
    n = len(facts)
    nbytes = len(text.encode("utf-8"))
    return {
        "doc_id": doc_id, "split": split, "domain": domain, "text": text,
        "words": len(text.split()), "bytes": nbytes, "n_facts": n,
        "bytes_per_fact": round(nbytes / n, 1), "facts": facts,
    }


def _write_corpus(root: Path, docs: list[dict], questions: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "docs.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8"
    )
    (root / "questions.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8"
    )
    return root


def _toy(attribution_text: str, ordinal_gold: str):
    """One document with a long and a short realisation of the same facts."""
    facts_long = [
        _fact("f1", "entity", "entity", 0,
              "The long form names the Kelvor Institute as the body responsible for signal verification.",
              "Kelvor Institute"),
        _fact("f2", "temporal", "temporal", 1,
              "The Kelvor Institute relocated from Ostra to Veym in 2047, after a full year of preparation.",
              "2047"),
        _fact("f3", "attribution", "attribution_uncertainty", 2, attribution_text,
              "Vashen Ruvel"),
    ]
    facts_short = [
        _fact("f1", "entity", "entity", 0,
              "The Kelvor Institute handles signal verification.", "Kelvor Institute"),
        _fact("f2", "temporal", "temporal", 1,
              "The Kelvor Institute moved from Ostra to Veym in 2047.", "2047"),
        _fact("f3", "attribution", "attribution_uncertainty", 2, attribution_text,
              "Vashen Ruvel"),
    ]
    questions = [
        {"qid": "p-0000-q1", "doc_id": "p-0000", "demand": "multi-hop", "answerable": True,
         "text": "Which came first, the Kelvor Institute relocation to Veym or the second phase planned for 2049?",
         "gold": ordinal_gold, "fact_ids": ["f2", "f3"], "fact_types": ["temporal", "attribution_uncertainty"]},
        {"qid": "p-0000-q2", "doc_id": "p-0000", "demand": "multi-hop", "answerable": True,
         "text": "How many years after its founding did the Kelvor Institute relocate to Veym?",
         "gold": "2", "fact_ids": ["f1", "f2"], "fact_types": ["entity", "temporal"]},
        {"qid": "p-0000-q3", "doc_id": "p-0000", "demand": "lookup", "answerable": True,
         "text": "Who runs the sensor unit?", "gold": "Vashen Ruvel",
         "fact_ids": ["f3"], "fact_types": ["attribution_uncertainty"]},
    ]
    long_text = " ".join(f["text"] for f in facts_long)
    short_text = " ".join(f["text"] for f in facts_short)
    return ([_doc("p-0000", long_text, facts_long)], [_doc("p-0000", short_text, facts_short)],
            questions)


class TestPremiseFlags(unittest.TestCase):
    def test_flags_denied_year_when_project_is_gold(self):
        corpus = {"docs": [_doc("p-0000", "", [
            _fact("f3", "attribution", "attribution_uncertainty", 0,
                  "Vashen Ruvel said the second phase would not arrive in 2049.", "Vashen Ruvel"),
        ])], "by_doc": {"p-0000": [
            {"qid": "p-0000-q1", "demand": "multi-hop",
             "text": "Which came first, the relocation to Veym or the second phase planned for 2049?",
             "gold": "the second phase"},
        ]}}
        flags = premise_flags(corpus)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["qid"], "p-0000-q1")

    def test_ignores_consistent_ordering(self):
        # gold is the relocation -> "project does not happen in year3" agrees
        corpus = {"docs": [_doc("p-0000", "", [
            _fact("f3", "attribution", "attribution_uncertainty", 0,
                  "Vashen Ruvel said the second phase would not arrive in 2049.", "Vashen Ruvel"),
        ])], "by_doc": {"p-0000": [
            {"qid": "p-0000-q1", "demand": "multi-hop",
             "text": "Which came first, the relocation to Veym or the second phase planned for 2049?",
             "gold": "the relocation"},
        ]}}
        self.assertEqual(premise_flags(corpus), [])

    def test_ignores_plain_uncertainty(self):
        corpus = {"docs": [_doc("p-0000", "", [
            _fact("f3", "attribution", "attribution_uncertainty", 0,
                  "Vashen Ruvel called the second phase, due in 2049, unsettled.", "Vashen Ruvel"),
        ])], "by_doc": {"p-0000": [
            {"qid": "p-0000-q1", "demand": "multi-hop",
             "text": "Which came first, the relocation to Veym or the second phase planned for 2049?",
             "gold": "the second phase"},
        ]}}
        self.assertEqual(premise_flags(corpus), [])


class TestValidatePair(unittest.TestCase):
    def test_matching_pair_passes_structural_checks(self):
        long_docs, short_docs, questions = _toy(
            "Vashen Ruvel called the second phase, due in 2049, unsettled.",
            "the second phase",
        )
        with tempfile.TemporaryDirectory() as tmp:
            a = _write_corpus(Path(tmp) / "a", long_docs, questions)
            b = _write_corpus(Path(tmp) / "b", short_docs, questions)
            report = validate_pair(a, b, bpf_a=133, bpf_b=100)
        by_name = {c["check"]: c for c in report["checks"]}
        for name in ("same_doc_ids", "same_fact_metadata", "questions_verbatim_identical",
                     "question_gold_matches_facts", "schema_consistent",
                     "multi_hop_cross_sentence_a", "multi_hop_cross_sentence_b",
                     "no_duplicate_docs_a", "no_duplicate_docs_b"):
            self.assertTrue(by_name[name]["ok"], f"{name}: {by_name[name]['detail']}")
        self.assertTrue(by_name["texts_differ_everywhere"]["ok"])
        self.assertIn("premise_flags", report)

    def test_question_edit_is_caught(self):
        long_docs, short_docs, questions = _toy(
            "Vashen Ruvel called the second phase, due in 2049, unsettled.",
            "the second phase",
        )
        edited = [dict(q) for q in questions]
        edited[0]["gold"] = "the relocation"  # a single-word corruption
        with tempfile.TemporaryDirectory() as tmp:
            a = _write_corpus(Path(tmp) / "a", long_docs, questions)
            b = _write_corpus(Path(tmp) / "b", short_docs, edited)
            report = validate_pair(a, b, bpf_a=133, bpf_b=100)
        by_name = {c["check"]: c for c in report["checks"]}
        self.assertFalse(by_name["questions_verbatim_identical"]["ok"])
        self.assertFalse(report["passed"])

    def test_fact_metadata_drift_is_caught(self):
        long_docs, short_docs, questions = _toy(
            "Vashen Ruvel called the second phase, due in 2049, unsettled.",
            "the second phase",
        )
        short_docs[0]["facts"][1]["gold"] = "2048"  # gold drift across densities
        with tempfile.TemporaryDirectory() as tmp:
            a = _write_corpus(Path(tmp) / "a", long_docs, questions)
            b = _write_corpus(Path(tmp) / "b", short_docs, questions)
            report = validate_pair(a, b, bpf_a=133, bpf_b=100)
        by_name = {c["check"]: c for c in report["checks"]}
        self.assertFalse(by_name["same_fact_metadata"]["ok"])
        self.assertFalse(report["passed"])

    def test_identical_texts_are_caught(self):
        long_docs, _short_docs, questions = _toy(
            "Vashen Ruvel called the second phase, due in 2049, unsettled.",
            "the second phase",
        )
        with tempfile.TemporaryDirectory() as tmp:
            a = _write_corpus(Path(tmp) / "a", long_docs, questions)
            b = _write_corpus(Path(tmp) / "b", [dict(d) for d in long_docs], questions)
            report = validate_pair(a, b, bpf_a=133, bpf_b=100)
        by_name = {c["check"]: c for c in report["checks"]}
        self.assertFalse(by_name["texts_differ_everywhere"]["ok"])


if __name__ == "__main__":
    unittest.main()
