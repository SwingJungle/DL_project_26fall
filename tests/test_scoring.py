"""Unit tests for answer normalization and scoring (P6.1-P6.3)."""
import unittest

from src.eval.normalize import is_unknown, normalize, numeric_value
from src.eval.score import match, score_answers, token_f1


class TestNormalize(unittest.TestCase):
    def test_articles_and_punctuation(self):
        self.assertEqual(normalize("The Kulgel Foundation."), "kulgel foundation")

    def test_number_words(self):
        self.assertEqual(normalize("four years"), "4 years")

    def test_thousands(self):
        self.assertEqual(normalize("1,200 modules"), "1200 modules")

    def test_numeric_value(self):
        self.assertEqual(numeric_value("in 2045"), 2045.0)
        self.assertEqual(numeric_value("1,200"), 1200.0)

    def test_unknown(self):
        self.assertTrue(is_unknown("unknown"))
        self.assertTrue(is_unknown("The message does not mention this."))
        self.assertFalse(is_unknown("Kulgel Foundation"))


class TestScoring(unittest.TestCase):
    def test_exact(self):
        r = match("Kulgel Foundation", "Kulgel Foundation", True)
        self.assertTrue(r["correct"])
        self.assertEqual(r["match_type"], "exact")

    def test_numeric(self):
        r = match("4 years", "4", True)
        self.assertTrue(r["correct"])
        self.assertEqual(r["match_type"], "numeric")

    def test_gold_in_pred(self):
        r = match("the relocation to Momdon", "the relocation", True)
        self.assertTrue(r["correct"])

    def test_pred_in_gold(self):
        r = match("below 8 degrees", "only below 8 degrees Celsius", True)
        self.assertTrue(r["correct"])

    def test_mismatch(self):
        r = match("Tesk", "Varn", True)
        self.assertFalse(r["correct"])

    def test_unknown_unanswerable(self):
        self.assertTrue(match("unknown", "unknown", False)["correct"])
        self.assertFalse(match("Kulgel", "unknown", False)["correct"])

    def test_unknown_for_answerable_is_wrong(self):
        r = match("unknown", "2045", True)
        self.assertFalse(r["correct"])
        self.assertEqual(r["match_type"], "predicted_unknown")

    def test_token_f1(self):
        self.assertAlmostEqual(token_f1("the Kulgel Foundation", "Kulgel Foundation"), 1.0)

    def test_score_answers(self):
        questions = [
            {"qid": "q1", "demand": "lookup", "gold": "2045", "answerable": True},
            {"qid": "q2", "demand": "multi-hop", "gold": "4", "answerable": True},
            {"qid": "q3", "demand": "lookup", "gold": "unknown", "answerable": False},
        ]
        scored = score_answers(questions, {"q1": "2045", "q2": "four", "q3": "unknown"})
        self.assertEqual(scored["accuracy"], 1.0)
        self.assertEqual(scored["n"], 3)


if __name__ == "__main__":
    unittest.main()
