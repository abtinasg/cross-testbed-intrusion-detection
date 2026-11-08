#!/usr/bin/env python3
"""Unit tests for CV spec v2 Amendment A1: temporal observability and threshold behaviour.
SYNTHETIC DATA ONLY. No dataset access, no model.

Run: python3 -I test_cv_threshold_reference.py   (from any directory)
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_threshold_reference as R  # noqa: E402

RNG = np.random.default_rng(7)
H = R.HOUR_US


class TestTb(unittest.TestCase):
    def test_finite_sample_index_formula(self):
        # tau == x_sorted[ceil((n-1)(1-alpha))] for many n, alpha
        for n in (1, 2, 3, 7, 10, 101, 1000):
            x = RNG.normal(size=n)
            xs = np.sort(x)
            for alpha in (0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 0.999, 1.0):
                idx = math.ceil((n - 1) * (1 - alpha) - 1e-12)
                self.assertEqual(R.tb_threshold(x, alpha), float(xs[idx]), (n, alpha))

    def test_calibration_fpr_never_exceeds_alpha(self):
        for _ in range(500):
            n = int(RNG.integers(1, 400))
            x = np.round(RNG.normal(size=n), int(RNG.integers(0, 3)))  # induce ties
            alpha = float(RNG.uniform(0, 1))
            tau = R.tb_threshold(x, alpha)
            self.assertLessEqual(float((x > tau).mean()), alpha + 1e-12)

    def test_ties_at_tau_not_alerted_conservative(self):
        x = np.array([0.5] * 100)
        tau = R.tb_threshold(x, 0.05)
        self.assertEqual(tau, 0.5)
        self.assertEqual(int(R.tb_alerts(x, tau).sum()), 0)  # achieved FPR 0 << nominal 0.05

    def test_alpha_edges_and_empty(self):
        x = np.array([1.0, 2.0, 3.0])
        self.assertEqual(R.tb_threshold(x, 0.0), 3.0)
        self.assertEqual(int(R.tb_alerts(x, 3.0).sum()), 0)
        self.assertEqual(R.tb_threshold(x, 1.0), 1.0)
        self.assertEqual(int(R.tb_alerts(x, 1.0).sum()), 2)
        with self.assertRaises(ValueError):
            R.tb_threshold([], 0.1)

    def test_achieved_vs_nominal_reported_separately(self):
        cal = RNG.normal(0, 1, 5000)
        tau = R.tb_threshold(cal, 0.01)
        held_benign = RNG.normal(0.3, 1, 5000)  # score-scale shift between inner and outer models
        achieved = R.fpr(R.tb_alerts(held_benign, tau), np.zeros(5000, bool))
        self.assertGreater(achieved, 0.01)  # shift inflates achieved FPR; must be reported, not corrected


class TestTieKey(unittest.TestCase):
    def setUp(self):
        self.n = 200
        self.f = np.round(RNG.normal(size=(self.n, 4)), 1)
        self.a = RNG.integers(0, H, self.n)
        self.score = np.round(RNG.uniform(size=self.n), 1)  # many ties
        self.token = np.arange(self.n)

    def test_permutation_invariance(self):
        k = R.tie_key(self.f, self.a)
        al, _ = R.ta_select(self.score, k, 0.1)
        sel = set(self.token[al])
        for _ in range(20):
            p = RNG.permutation(self.n)
            kp = R.tie_key(self.f[p], self.a[p])
            alp, _ = R.ta_select(self.score[p], kp, 0.1)
            self.assertEqual(set(self.token[p][alp]), sel)

    def test_independent_of_labels_ids_and_file_order(self):
        k1 = R.tie_key(self.f, self.a)
        # labels / ids are not inputs; recomputing with different 'file order' gives the same per-row key
        p = RNG.permutation(self.n)
        k2 = R.tie_key(self.f[p], self.a[p])
        self.assertTrue(np.array_equal(k1[p], k2))
        import inspect
        params = set(inspect.signature(R.tie_key).parameters)
        self.assertEqual(params, {"features", "availability_us", "seed"})

    def test_uses_only_decision_time_information(self):
        # availability time and features are known at batch decision time; key changes if a changes
        k1 = R.tie_key(self.f[:1], self.a[:1])
        k2 = R.tie_key(self.f[:1], self.a[:1] + 1)
        self.assertNotEqual(int(k1[0]), int(k2[0]))

    def test_deterministic_and_seed_dependent(self):
        self.assertTrue(np.array_equal(R.tie_key(self.f, self.a), R.tie_key(self.f, self.a)))
        self.assertFalse(np.array_equal(R.tie_key(self.f, self.a, 1), R.tie_key(self.f, self.a, 2)))

    def test_indistinguishable_rows_alerted_as_group(self):
        f = np.zeros((5, 3)); a = np.zeros(5, np.int64); s = np.ones(5)
        k = R.tie_key(f, a)
        al, info = R.ta_select(s, k, 0.4)  # quota 2, but 5 identical rows
        self.assertEqual(info["quota"], 2)
        self.assertEqual(int(al.sum()), 5)
        self.assertEqual(info["overshoot"], 3)


class TestTaQuota(unittest.TestCase):
    def test_integer_quota_floor(self):
        s = RNG.uniform(size=1999); k = R.tie_key(RNG.normal(size=(1999, 2)), np.zeros(1999, np.int64))
        al, info = R.ta_select(s, k, 0.01)
        self.assertEqual(info["quota"], 19)
        self.assertEqual(int(al.sum()), 19)
        self.assertTrue(np.all(s[al] >= s[~al].max()))

    def test_small_window_and_empty(self):
        al, info = R.ta_select(np.array([0.9] * 50), R.tie_key(np.arange(50.0)[:, None], np.zeros(50, np.int64)), 0.01)
        self.assertEqual(info["quota"], 0); self.assertEqual(int(al.sum()), 0)
        al, info = R.ta_select(np.array([]), np.array([], np.uint64), 0.5)
        self.assertTrue(info["empty"]); self.assertEqual(al.size, 0)

    def test_tie_at_cut_resolved_by_key_not_order(self):
        s = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
        f = np.arange(5.0)[:, None]
        k = R.tie_key(f, np.zeros(5, np.int64))
        al, info = R.ta_select(s, k, 0.4)  # quota 2: row 0 + one of rows 1-3 (smallest key)
        self.assertTrue(al[0])
        winner = 1 + int(np.argmin(k[1:4]))
        self.assertTrue(al[winner]); self.assertEqual(int(al.sum()), 2)
        self.assertEqual(info["ties_at_cut_score"], 3)


class TestTemporalObservability(unittest.TestCase):
    def setUp(self):
        n = 5000
        self.s = RNG.integers(0, 6 * H, n).astype(np.int64)
        self.d = RNG.integers(0, R.TIMEOUT_US + 1, n).astype(np.int64)  # durations <= 120 s

    def test_time_ordering_start_completion_availability(self):
        for rule in R.AVAILABILITY_RULES:
            t = R.timing_table(self.s, self.d, rule)
            self.assertTrue(np.all(t["start"] <= t["completion"]))
            self.assertTrue(np.all(t["completion"] <= t["availability"]), rule)

    def test_no_flow_decided_before_its_score_is_available(self):
        for rule in R.AVAILABILITY_RULES:
            for delay in (0, 5 * R.US, 30 * 60 * R.US):
                t = R.timing_table(self.s, self.d, rule, score_delay_us=delay)
                self.assertTrue(np.all(t["availability"] < t["closure"]))
                self.assertTrue(np.all(t["decision"] >= t["score_available"]), (rule, delay))
                self.assertTrue(np.all(t["decision"] >= t["closure"]))

    def test_batching_by_start_time_would_violate_observability(self):
        # counterexample motivating the rule: with batches by START time, flows completing after closure exist
        closure_by_start = (self.s // H + 1) * H
        completion = self.s + self.d
        self.assertGreater(int((completion > closure_by_start).sum()), 0)

    def test_negative_duration_rejected(self):
        with self.assertRaises(ValueError):
            R.availability_time([0], [-1])


class TestTaPrime(unittest.TestCase):
    def setUp(self):
        self.b = np.array([0] * 10 + [1] * 10 + [3] * 10)  # hour 2 empty
        self.s = np.concatenate([np.linspace(0, 0.9, 10), np.linspace(0, 0.9, 10), np.linspace(0, 0.9, 10)])
        self.k = R.tie_key(np.arange(30.0)[:, None], np.zeros(30, np.int64))

    def test_ge_boundary(self):
        al, info = R.ta_prime(self.s, self.k, self.b, 0.2, fallback_tau=0.95)
        tau = info[1]["tau"]  # 2nd highest of hour 0 = 0.8
        self.assertAlmostEqual(tau, 0.8)
        m1 = self.b == 1
        self.assertTrue(np.all(al[m1] == (self.s[m1] >= tau)))
        self.assertTrue(al[m1][np.isclose(self.s[m1], 0.8)].all())  # equality alerted

    def test_identical_scores_exceed_quota(self):
        s = np.r_[np.linspace(0, 0.9, 10), np.full(10, 0.8)]
        b = np.r_[np.zeros(10, int), np.ones(10, int)]
        k = R.tie_key(np.arange(20.0)[:, None], np.zeros(20, np.int64))
        al, info = R.ta_prime(s, k, b, 0.2, fallback_tau=0.95)
        self.assertEqual(info[1]["achieved"], 10)
        self.assertEqual(info[1]["quota_reference"], 2)

    def test_first_hour_empty_previous_and_zero_quota(self):
        al_tb, inf_tb = R.ta_prime(self.s, self.k, self.b, 0.2, fallback_tau=0.85, fallback="tb")
        al_sp, inf_sp = R.ta_prime(self.s, self.k, self.b, 0.2, fallback_tau=0.85, fallback="suppress")
        self.assertEqual(inf_tb[0]["mode"], "fallback_tb")
        self.assertEqual(inf_tb[3]["mode"], "fallback_tb")   # previous hour (2) empty
        self.assertEqual(inf_sp[3]["mode"], "suppressed")
        self.assertEqual(inf_tb[2]["N"], 0)
        self.assertEqual(int(al_sp[self.b == 3].sum()), 0)
        self.assertEqual(int(al_tb[self.b == 3].sum()), int((self.s[self.b == 3] > 0.85).sum()))
        # zero previous quota: q*N_prev < 1
        al_z, inf_z = R.ta_prime(self.s, self.k, self.b, 0.05, fallback_tau=0.85, fallback="suppress")
        self.assertEqual(inf_z[1]["mode"], "suppressed")

    def test_operational_cost_of_suppression(self):
        y = self.s >= 0.85  # synthetic attacks = top-scoring flows
        al_tb, _ = R.ta_prime(self.s, self.k, self.b, 0.2, fallback_tau=0.85, fallback="tb")
        al_sp, _ = R.ta_prime(self.s, self.k, self.b, 0.2, fallback_tau=0.85, fallback="suppress")
        missed_extra = int((y & al_tb & ~al_sp).sum())
        self.assertGreater(missed_extra, 0)  # suppression costs detected attacks in fallback hours


class TestRecallAggregation(unittest.TestCase):
    def test_pooled_vs_grouped(self):
        y = np.r_[np.ones(1000, bool), np.ones(10, bool)]
        al = np.r_[np.ones(1000, bool), np.zeros(10, bool)]
        g = np.r_[np.zeros(1000, int), np.ones(10, int)]
        self.assertAlmostEqual(R.pooled_recall(al, y), 1000 / 1010)
        per, mean = R.grouped_recall(al, y, g)
        self.assertEqual(per, {0: 1.0, 1: 0.0}); self.assertEqual(mean, 0.5)

    def test_undefined_without_attacks(self):
        self.assertTrue(math.isnan(R.pooled_recall(np.ones(5, bool), np.zeros(5, bool))))


if __name__ == "__main__":
    import platform, sklearn  # noqa: E401
    print(f"python {platform.python_version()} numpy {np.__version__} scikit-learn {sklearn.__version__} platform {platform.platform()}")
    unittest.main(verbosity=2)
