#!/usr/bin/env python3
"""Regression tests for Amendment A1b (synthetic data only).
Run: python3 -I -B test_cv_threshold_reference_A1b.py
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_threshold_reference_A1b as B  # noqa: E402

A1 = B.A1
RNG = np.random.default_rng(11)


def keys_for(features, a=None):
    f = np.asarray(features, dtype=np.float64)
    if f.ndim == 1:
        f = f[:, None]
    a = np.zeros(len(f), np.int64) if a is None else a
    return A1.tie_key(f, a)


class TestHardCap(unittest.TestCase):
    def test_never_exceeds_quota_random(self):
        for _ in range(300):
            n = int(RNG.integers(0, 300))
            # heavy duplication: few distinct feature rows
            f = RNG.integers(0, int(RNG.integers(1, 20)), size=(n, 2)).astype(float)
            s = (f.sum(axis=1) % 7) / 7.0 if n else np.array([])
            k = keys_for(f) if n else np.array([], np.uint64)
            quota = int(RNG.integers(0, 50))
            al, info = B.ta_select_hardcap(s, k, quota)
            self.assertLessEqual(int(al.sum()), quota)
            self.assertEqual(info["alerted"] + info["unused_capacity"], quota)
            if al.any() and (~al).any():  # alerted set is a score-upper set (no backfill)
                # every non-alerted row ranks at or below the lowest alerted row
                self.assertTrue(np.all(s[~al] <= s[al].min()))

    def test_review_example_5_identical_quota_2(self):
        f = np.zeros((5, 3)); s = np.ones(5)
        al, info = B.ta_select_hardcap(s, keys_for(f), 2)
        self.assertEqual(int(al.sum()), 0)
        self.assertEqual(info["omitted_boundary_group_size"], 5)
        self.assertEqual(info["unused_capacity"], 2)
        al2, info2 = B.ta_select_overshoot(s, keys_for(f), 2)
        self.assertEqual(int(al2.sum()), 5)
        self.assertEqual(info2["overshoot"], 3)

    def test_tied_scores_distinct_keys_fill_exactly(self):
        s = np.full(10, 0.5); f = np.arange(10.0)
        al, info = B.ta_select_hardcap(s, keys_for(f), 3)
        self.assertEqual(int(al.sum()), 3); self.assertEqual(info["unused_capacity"], 0)

    def test_identical_hash_group_fits_exactly(self):
        s = np.r_[np.full(3, 0.9), np.full(5, 0.1)]
        f = np.r_[np.zeros(3), np.arange(1.0, 6.0)]
        al, info = B.ta_select_hardcap(s, keys_for(f), 3)
        self.assertTrue(al[:3].all()); self.assertEqual(int(al.sum()), 3)

    def test_boundary_group_omitted_no_backfill(self):
        # top singleton, then a group of 4 identical, then distinct lower rows; quota 3
        s = np.r_[0.9, np.full(4, 0.8), 0.7, 0.6]
        f = np.r_[1.0, np.zeros(4), 2.0, 3.0]
        al, info = B.ta_select_hardcap(s, keys_for(f), 3)
        self.assertEqual(list(np.flatnonzero(al)), [0])
        self.assertEqual(info["unused_capacity"], 2)
        self.assertEqual(info["omitted_boundary_group_size"], 4)
        self.assertFalse(al[5] or al[6])  # lower-ranked rows NOT used to backfill

    def test_zero_quota_and_empty_batch(self):
        al, info = B.ta_select_hardcap(np.array([0.9, 0.8]), keys_for([1.0, 2.0]), 0)
        self.assertEqual(int(al.sum()), 0); self.assertEqual(info["unused_capacity"], 0)
        al, info = B.ta_select_hardcap(np.array([]), np.array([], np.uint64), 5)
        self.assertTrue(info["empty"]); self.assertEqual(info["unused_capacity"], 5)

    def test_large_duplicate_group(self):
        s = np.r_[np.full(10000, 0.95), RNG.uniform(0, 0.9, 1000)]
        f = np.r_[np.zeros(10000), np.arange(1.0, 1001.0)]
        y = np.r_[np.ones(10000, bool), np.zeros(1000, bool)]
        k = keys_for(f)
        al_h, inf_h = B.ta_select_hardcap(s, k, 500)
        al_o, inf_o = B.ta_select_overshoot(s, k, 500)
        self.assertEqual(int(al_h.sum()), 0)
        self.assertEqual(inf_o["overshoot"], 9500)
        self.assertAlmostEqual(B.recall_loss(al_h, al_o, y), 1.0)  # recall loss recorded

    def test_equivalence_with_A1_when_no_indistinguishable_groups(self):
        for _ in range(50):
            n = int(RNG.integers(1, 200)); f = RNG.normal(size=(n, 3)); s = np.round(RNG.uniform(size=n), 2)
            k = keys_for(f); q = float(RNG.uniform(0, 0.5))
            al_a1, _ = A1.ta_select(s, k, q)
            al_h, _ = B.ta_select_hardcap(s, k, B.quota_relative(q, n))
            self.assertTrue(np.array_equal(al_a1, al_h))


class TestBudgets(unittest.TestCase):
    def test_relative_vs_absolute(self):
        self.assertEqual(B.quota_relative(0.01, 1999), 19)
        self.assertEqual(B.quota_relative(0.01, 50), 0)
        self.assertEqual(B.quota_absolute(100, 50), 50)
        self.assertEqual(B.quota_absolute(100, 650000), 100)
        self.assertEqual(B.quota_absolute(0, 10), 0)
        with self.assertRaises(ValueError):
            B.quota_absolute(2.5, 10)
        with self.assertRaises(ValueError):
            B.quota_relative(1.5, 10)

    def test_relative_quota_scales_with_volume_absolute_does_not(self):
        # same q gives ~100x more alerts in a 100x larger hour; K stays fixed
        self.assertEqual(B.quota_relative(0.01, 650000) // B.quota_relative(0.01, 6500), 100)
        self.assertEqual(B.quota_absolute(200, 650000), B.quota_absolute(200, 6500))


class TestT2c(unittest.TestCase):
    def test_precondition_pass_and_fail(self):
        self.assertEqual(B.check_t2c_precondition([0, 120_000_000]), 0)
        with self.assertRaises(ValueError):
            B.check_t2c_precondition([0, 120_000_001])
        with self.assertRaises(ValueError):
            B.availability_time_checked([0], [120_000_001], "T2c")
        self.assertEqual(int(B.availability_time_checked([0], [120_000_001], "T2")[0]), 120_000_001)


class TestVersionedTaPrime(unittest.TestCase):
    def setUp(self):
        self.b = np.repeat([0, 1, 2], 10)
        self.s = np.tile(np.linspace(0, 0.9, 10), 3)
        self.k = keys_for(np.arange(30.0))
        self.qf = lambda n: B.quota_relative(0.2, n)

    def test_same_version_carries_threshold(self):
        v = np.repeat(["m1", "m1", "m1"], 10)
        al, info = B.ta_prime_versioned(self.s, self.k, self.b, v, self.qf, {"m1": 0.95})
        self.assertEqual(info[0]["mode"], "fallback_tb")
        self.assertEqual(info[1]["mode"], "previous_hour")
        self.assertAlmostEqual(info[1]["tau"], 0.8)

    def test_version_change_resets_to_source_calibrated(self):
        v = np.repeat(["m1", "m2", "m2"], 10)
        al, info = B.ta_prime_versioned(self.s, self.k, self.b, v, self.qf, {"m1": 0.95, "m2": 0.55})
        self.assertEqual(info[1]["mode"], "reset_version_change")
        self.assertEqual(info[1]["tau"], 0.55)
        m1 = self.b == 1
        self.assertTrue(np.array_equal(al[m1], self.s[m1] > 0.55))
        self.assertEqual(info[2]["mode"], "previous_hour")  # same version again: carry allowed

    def test_mixed_versions_within_batch_rejected(self):
        v = np.r_[np.repeat("m1", 15), np.repeat("m2", 15)]
        with self.assertRaises(ValueError):
            B.ta_prime_versioned(self.s, self.k, self.b, v, self.qf, {"m1": 0.9, "m2": 0.9})

    def test_absolute_quota_fn(self):
        v = np.repeat(["m1"] * 3, 10)
        al, info = B.ta_prime_versioned(self.s, self.k, self.b, v, lambda n: B.quota_absolute(3, n), {"m1": 0.95})
        self.assertAlmostEqual(info[1]["tau"], 0.7)  # 3rd highest of hour 0


if __name__ == "__main__":
    import platform, sklearn  # noqa: E401
    print(f"python {platform.python_version()} numpy {np.__version__} scikit-learn {sklearn.__version__} platform {platform.platform()}")
    unittest.main(verbosity=2)
