"""Tests for the density tag on record keys (paired-density corpora).

The two densities of the paired design share ``doc_id``/``method``/``level``, so
without a corpus tag a resumed run or a merged file would silently treat the
b100 records as already-done copies of the base records.
"""
import unittest

from src.run.receiver import plan_receiver_tasks
from src.run.sender import Level, message_key, plan_messages

DOC = {
    "doc_id": "p-0000",
    "split": "eval",
    "domain": "institute",
    "text": "The Kelvor Institute handles signal verification.",
    "n_facts": 8,
}


class TestMessageKey(unittest.TestCase):
    def test_default_is_unchanged(self):
        self.assertEqual(message_key("p-0000", "qwen3.7-plus", "structured", "r50"),
                         "p-0000|qwen3.7-plus|structured|r50")

    def test_corpus_tag_prefixes_the_key(self):
        self.assertEqual(
            message_key("p-0000", "qwen3.7-plus", "structured", "r50", "b100"),
            "b100|p-0000|qwen3.7-plus|structured|r50",
        )

    def test_densities_do_not_collide(self):
        base = message_key("p-0000", "qwen3.7-plus", "structured", "r50", "base")
        dense = message_key("p-0000", "qwen3.7-plus", "structured", "r50", "b100")
        self.assertNotEqual(base, dense)
        # analyses recover the message key from a prediction key this way
        pred = f"{dense}|recv:qwen3.7-max"
        self.assertEqual(pred.rsplit("|recv:", 1)[0], dense)


class TestPlanKeys(unittest.TestCase):
    def test_message_plan_keys_are_disjoint_across_densities(self):
        docs = [DOC]
        levels = [Level("r50", "ratio", 0.5)]
        base = plan_messages(docs, ["qwen3.7-plus"], ["abstractive"], levels, "base")
        dense = plan_messages(docs, ["qwen3.7-plus"], ["abstractive"], levels, "b100")
        self.assertEqual(len({t["key"] for t in base} & {t["key"] for t in dense}), 0)

    def test_full_baseline_keys_carry_the_tag(self):
        messages = [{
            "key": "b100|p-0000|qwen3.7-plus|abstractive|r50",
            "doc_id": "p-0000", "model_key": "qwen3.7-plus",
            "method": "abstractive", "level": "r50",
            "message": "short message", "final_bytes": 13, "status": "ok",
        }]
        docs = {"p-0000": DOC}
        tasks = plan_receiver_tasks(
            messages, docs, ["qwen3.7-plus"], mode="self",
            include_full=True, corpus_id="b100",
        )
        keys = {t["key"] for t in tasks}
        self.assertIn("b100|p-0000|full|recv:qwen3.7-plus", keys)
        self.assertIn("b100|p-0000|qwen3.7-plus|abstractive|r50|recv:qwen3.7-plus", keys)

    def test_no_tag_keeps_the_old_forms(self):
        messages = [{
            "key": "p-0000|qwen3.7-plus|abstractive|r50",
            "doc_id": "p-0000", "model_key": "qwen3.7-plus",
            "method": "abstractive", "level": "r50",
            "message": "short message", "final_bytes": 13, "status": "ok",
        }]
        tasks = plan_receiver_tasks(messages, {"p-0000": DOC}, ["qwen3.7-plus"],
                                    mode="self", include_full=True)
        keys = {t["key"] for t in tasks}
        self.assertEqual(keys, {
            "p-0000|qwen3.7-plus|abstractive|r50|recv:qwen3.7-plus",
            "p-0000|full|recv:qwen3.7-plus",
        })


if __name__ == "__main__":
    unittest.main()
