"""Unit tests for the byte-budget validator (P1.6)."""
import math
import unittest

from src.compress.budget import (
    budget_report,
    count_bytes,
    count_chars,
    count_tokens_est,
    count_words,
    measure,
    window_from_b,
    window_from_ratio,
)


class TestBudget(unittest.TestCase):
    def test_ratios_window(self):
        w = window_from_ratio(1000, 0.50, 0.02)
        self.assertEqual((w.target, w.lo, w.hi), (500, 480, 520))
        self.assertTrue(w.compliant(480))
        self.assertTrue(w.compliant(520))
        self.assertFalse(w.compliant(479))
        self.assertFalse(w.compliant(521))

    def test_direction(self):
        w = window_from_ratio(1000, 0.50, 0.02)
        self.assertEqual(w.direction(500), "ok")
        self.assertEqual(w.direction(600), "over")
        self.assertEqual(w.direction(100), "under")

    def test_b_window(self):
        # 1198 bytes, 9 facts, b=69 -> target 621; band is +/-2% of source (23.96)
        w = window_from_b(1198, 9, 69.0, 0.02)
        self.assertEqual(w.target, 621)
        self.assertEqual((w.lo, w.hi), (598, 644))
        self.assertEqual(w.direction(650), "over")
        self.assertEqual(w.direction(590), "under")

    def test_four_counts(self):
        text = "Café facts: 8 modules.\nSecond line."
        self.assertEqual(count_bytes(text), len(text.encode("utf-8")))
        self.assertEqual(count_chars(text), len(text))
        self.assertEqual(count_words(text), 6)
        self.assertEqual(count_tokens_est(text), math.ceil(count_bytes(text) / 4))
        m = measure(text)
        self.assertEqual(m["bytes"], count_bytes(text))
        self.assertIn("tokens_est", m)

    def test_budget_report(self):
        w = window_from_ratio(1000, 0.50, 0.02)
        rep = budget_report("x" * 500, w)
        self.assertTrue(rep["compliant"])
        self.assertEqual(rep["direction"], "ok")
        self.assertEqual(rep["delta_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
