"""Retention proxies and receiver JSON parsing."""
import unittest

from src.eval.retention import retention_summary
from src.run.receiver import parse_answers

DOC = {
    "doc_id": "p-0000",
    "facts": [
        {
            "fid": "f1",
            "type": "number_date",
            "text": "The Kulgel Foundation was founded in 2045 by Mira Sol.",
            "gold": "2045",
        },
        {
            "fid": "f2",
            "type": "condition_exception",
            "text": "The light unit operates only below 8 degrees Celsius.",
            "gold": "only below 8 degrees Celsius",
        },
    ],
}
QUESTIONS = [
    {"qid": "q1", "demand": "lookup", "answerable": True, "gold": "2045", "fact_ids": ["f1"]},
    {
        "qid": "q2",
        "demand": "multi-hop",
        "answerable": True,
        "gold": "8",
        "fact_ids": ["f1", "f2"],
    },
    {"qid": "q3", "demand": "lookup", "answerable": False, "gold": "unknown", "fact_ids": []},
]


class TestRetention(unittest.TestCase):
    def test_full_message_retains_everything(self):
        msg = "Kulgel Foundation founded 2045 by Mira Sol. Light unit operates only below 8 degrees Celsius."
        out = retention_summary(DOC, msg, QUESTIONS, "abstractive")
        self.assertEqual(out["n_facts"], 2)
        self.assertEqual(out["n_facts_survived"], 2)
        self.assertEqual(out["question_evidence_rate"], 1.0)

    def test_missing_number_fails(self):
        msg = "Kulgel Foundation founded by Mira Sol. Light unit operates only below degrees Celsius."
        out = retention_summary(DOC, msg, QUESTIONS, "abstractive")
        self.assertEqual(out["n_numbers_missing"], 2)
        self.assertLess(out["fact_retention"], 1.0)

    def test_structured_segments(self):
        msg = '{"v":1,"f":[{"k":"D","x":"Kulgel Foundation was founded in 2045."}]}'
        out = retention_summary(DOC, msg, QUESTIONS, "structured")
        self.assertEqual(out["n_facts_survived"], 1)


class TestReceiverParse(unittest.TestCase):
    def test_valid(self):
        answers, err = parse_answers('{"answers":[{"qid":"q1","a":"2045"}]}')
        self.assertIsNone(err)
        self.assertEqual(answers, {"q1": "2045"})

    def test_fenced(self):
        answers, err = parse_answers('```json\n{"answers":[{"qid":"q1","a":"x"}]}\n```')
        self.assertIsNone(err)
        self.assertEqual(answers["q1"], "x")

    def test_bare_list(self):
        answers, err = parse_answers('[{"qid":"q1","a":"x"}]')
        self.assertIsNone(err)
        self.assertEqual(answers["q1"], "x")

    def test_missing_answers_key(self):
        _, err = parse_answers('{"foo": 1}')
        self.assertEqual(err, "no_answers_key")

    def test_invalid(self):
        _, err = parse_answers("not json at all")
        self.assertEqual(err, "invalid_json")


if __name__ == "__main__":
    unittest.main()
