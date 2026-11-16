"""Unit tests for the pre-registered main analysis (``src.analysis.main_model``)."""
import unittest

import numpy as np

from src.analysis.main_model import (
    _add_covariates,
    cell_table,
    design_matrix,
    drop_flagged,
    fit_lpm_cluster,
)


def _row(doc, sender, receiver, method, density, qid, correct, demand="lookup",
         ratio=0.5, pair_type="self"):
    return {
        "doc_id": doc, "sender_model": sender, "receiver_model": receiver,
        "method": method, "density": density, "qid": qid,
        "demand": demand, "correct": correct,
        "message_bytes": int(round(ratio * 1000)), "actual_ratio": ratio,
        "pair_type": pair_type,
    }


def _synthetic(effects, *, n_docs=4, n_q=5):
    """Balanced rows; ``effects`` maps (density, method) -> accuracy."""
    rows = []
    for d in range(n_docs):
        for q in range(n_q):
            for density in ("base", "b100"):
                for method in ("abstractive", "structured"):
                    acc = effects[(density, method)]
                    correct = q < round(acc * n_q)
                    rows.append(_row(f"p-{d:04d}", "m1", "m1", method, density,
                                     f"p-{d:04d}-q{q}", correct))
    return rows


class TestCells(unittest.TestCase):
    def test_cell_table_recovers_known_accuracies(self):
        rows = _add_covariates(_synthetic({
            ("base", "abstractive"): 0.8, ("base", "structured"): 0.8,
            ("b100", "abstractive"): 0.8, ("b100", "structured"): 0.4,
        }))
        t = cell_table(rows)
        self.assertAlmostEqual(t["base|abstractive"]["accuracy"], 0.8, places=6)
        self.assertAlmostEqual(t["b100|structured"]["accuracy"], 0.4, places=6)
        self.assertAlmostEqual(t["interaction"]["interaction"], 0.4, places=6)

    def test_self_only_scope_is_respected(self):
        rows = _synthetic({("base", "abstractive"): 1.0, ("base", "structured"): 1.0,
                           ("b100", "abstractive"): 1.0, ("b100", "structured"): 0.0})
        for r in rows:
            r["pair_type"] = "cross"
        with_self = _add_covariates(rows)
        t = cell_table(with_self, self_only=True)
        self.assertEqual(t["base|abstractive"]["n_questions"], 0)


class TestDesignMatrix(unittest.TestCase):
    def test_columns_are_centered_and_indicators(self):
        rows = _add_covariates(_synthetic({k: 0.5 for k in (
            ("base", "abstractive"), ("base", "structured"),
            ("b100", "abstractive"), ("b100", "structured"))}))
        X, names = design_matrix(rows)
        self.assertEqual(names, ["intercept", "struct", "dense", "struct_x_dense",
                                 "ratio_c", "multihop", "self"])
        self.assertTrue(np.allclose(X[:, 0], 1.0))
        self.assertTrue(np.allclose(X[:, 3], X[:, 1] * X[:, 2]))
        self.assertTrue(np.allclose(X[:, 4], 0.0))  # ratio 0.5 -> centered 0


class TestLPM(unittest.TestCase):
    def test_lpm_recovers_the_interaction(self):
        # structured loses 30pp at b100 and nothing at base; effects are exact
        # so the balanced fit should recover it up to floating point.
        effects = {
            ("base", "abstractive"): 0.9, ("base", "structured"): 0.9,
            ("b100", "abstractive"): 0.9, ("b100", "structured"): 0.6,
        }
        rows = _add_covariates(_synthetic(effects, n_docs=6, n_q=10))
        fit = fit_lpm_cluster(rows)
        self.assertAlmostEqual(fit["h2_gap_interaction_pp"], 30.0, places=6)
        self.assertAlmostEqual(fit["struct_extra_loss_pp"], -30.0, places=6)
        self.assertGreater(fit["coefs"]["struct_x_dense"]["se"], 0.0)

    def test_clusters_are_documents(self):
        effects = {("base", "abstractive"): 1.0, ("base", "structured"): 0.0,
                   ("b100", "abstractive"): 1.0, ("b100", "structured"): 0.0}
        rows = _add_covariates(_synthetic(effects, n_docs=3, n_q=4))
        fit = fit_lpm_cluster(rows)
        self.assertEqual(fit["n_clusters"], 3)


class TestD11(unittest.TestCase):
    def test_drop_flagged_is_density_specific(self):
        rows = [_row("p-0000", "m1", "m1", "abstractive", "base", "q1", True),
                _row("p-0000", "m1", "m1", "abstractive", "b100", "q1", True),
                _row("p-0000", "m1", "m1", "abstractive", "b100", "q2", True)]
        kept = drop_flagged(rows, {"base": ["q1"], "b100": ["q1"]})
        self.assertEqual([(r["density"], r["qid"]) for r in kept], [("b100", "q2")])


if __name__ == "__main__":
    unittest.main()
