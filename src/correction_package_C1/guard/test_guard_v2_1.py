#!/usr/bin/env python3
"""Regression tests for guard v2.1 (synthetic only; no model, no data).
Run: python3 -I -B test_guard_v2_1.py
"""
import os
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import guard_v2_1 as G  # noqa: E402

GRID = os.path.join(HERE, "..", "..", "next_workstream", "CONFIG_GRID_F3_v1.csv")
MB = 1024 ** 2


def work_ok(n):
    return float(n) / 10.0


def work_alloc(mb, hold_s):
    b = bytearray(mb * MB)
    for i in range(0, len(b), 4096):
        b[i] = 1
    time.sleep(hold_s)
    return 1.0


def ok(m, f, s=0.5):
    return dict(model_id=m, fold=f, status="OK", score=s)


class TestExpectedKeys(unittest.TestCase):
    def setUp(self):
        self.ids = G.grid_config_ids(GRID)

    def test_keys_from_frozen_grid(self):
        self.assertEqual(len(self.ids), 12)
        self.assertEqual(len(G.expected_keys("flat", self.ids)), 60)
        self.assertEqual(len(G.expected_keys("tb", self.ids, "HGB-05")), 20)
        self.assertEqual(len(G.expected_keys("nested", self.ids)), 240)
        with self.assertRaises(ValueError):
            G.expected_keys("tb", self.ids, "XX-99")

    def test_60_repeats_of_one_model_fold_never_complete(self):
        exp = G.expected_keys("flat", self.ids)
        d = G.grid_decision([ok("HGB-01", "Wed")] * 60, exp)
        self.assertNotEqual(d["decision"], "COMPLETE")
        self.assertEqual(d["decision"], "INVALID")
        self.assertIn(("HGB-01", "Wed"), d["duplicates"])
        self.assertEqual(len(d["missing"]), 59)

    def test_complete_only_with_every_unique_key(self):
        exp = G.expected_keys("flat", self.ids)
        recs = [ok(m, f) for m, f in sorted(exp)]
        self.assertEqual(G.grid_decision(recs, exp)["decision"], "COMPLETE")
        self.assertEqual(G.grid_decision(recs[:-1], exp)["decision"], "CONTINUE")

    def test_unexpected_key_rejected(self):
        exp = G.expected_keys("flat", self.ids)
        recs = [ok(m, f) for m, f in sorted(exp)] + [ok("XGB-01", "Mon")]
        self.assertEqual(G.grid_decision(recs, exp)["decision"], "INVALID")
        recs2 = [ok(m, f) for m, f in sorted(exp)][:-1] + [ok("LR-01", "Sat")]
        self.assertEqual(G.grid_decision(recs2, exp)["decision"], "INVALID")

    def test_duplicate_among_full_set_rejected(self):
        exp = G.expected_keys("flat", self.ids)
        recs = [ok(m, f) for m, f in sorted(exp)] + [ok("RF-04", "Fri", 0.9)]
        self.assertEqual(G.grid_decision(recs, exp)["decision"], "INVALID")

    def test_failure_records_must_not_carry_scores(self):
        exp = G.expected_keys("flat", self.ids)
        recs = [ok(m, f) for m, f in sorted(exp)]
        recs[0] = dict(model_id=recs[0]["model_id"], fold=recs[0]["fold"], status="RESOURCE_EXCEEDED", score=0.7)
        d = G.grid_decision(recs, exp)
        self.assertEqual(d["decision"], "INVALID"); self.assertEqual(len(d["failure_records_with_score"]), 1)

    def test_failure_without_score_stops(self):
        exp = G.expected_keys("flat", self.ids)
        recs = [ok(m, f) for m, f in sorted(exp)]
        recs[5] = dict(model_id=recs[5]["model_id"], fold=recs[5]["fold"], status="TIME_EXCEEDED", score=None)
        self.assertEqual(G.grid_decision(recs, exp)["decision"], "STOP")

    def test_invalid_ok_records(self):
        exp = G.expected_keys("flat", self.ids)
        for bad in (None, float("nan"), float("inf"), True):
            recs = [ok(m, f) for m, f in sorted(exp)]
            recs[0]["score"] = bad
            self.assertEqual(G.grid_decision(recs, exp)["decision"], "INVALID", bad)
        recs = [ok(m, f) for m, f in sorted(exp)]; recs[0]["status"] = "DONE"
        self.assertEqual(G.grid_decision(recs, exp)["decision"], "INVALID")


class TestPreflight(unittest.TestCase):
    def test_thresholds_fixed(self):
        self.assertEqual(G.MIN_FREE_DISK_BYTES, 10 * 1024 ** 3)
        self.assertEqual(G.MIN_AVAILABLE_MEM_BYTES, int(7.0 * 1024 ** 3))
        self.assertEqual(G.DEFAULT_LIMIT_BYTES, int(6.5 * 1024 ** 3))

    def test_disk_and_memory_checks(self):
        with tempfile.TemporaryDirectory() as d:
            lk = os.path.join(d, "fit.lock")
            self.assertTrue(G.preflight(d, lk, _disk_free=20 * 1024 ** 3, _available=7.5 * 1024 ** 3)["ok"])
            r = G.preflight(d, lk, _disk_free=9 * 1024 ** 3, _available=7.5 * 1024 ** 3)
            self.assertFalse(r["ok"]); self.assertFalse(r["checks"]["free_disk_ok"])
            r = G.preflight(d, lk, _disk_free=20 * 1024 ** 3, _available=6.99 * 1024 ** 3)
            self.assertFalse(r["ok"]); self.assertFalse(r["checks"]["available_mem_ok"])

    def test_real_measurement_runs(self):
        with tempfile.TemporaryDirectory() as d:
            r = G.preflight(d, os.path.join(d, "fit.lock"))
            self.assertIn("available_mem_bytes", r["checks"])  # value recorded; outcome host-dependent

    def test_lock_exclusive_and_stale_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            lk = os.path.join(d, "fit.lock")
            self.assertTrue(G.acquire_fit_lock(lk))
            r = G.preflight(d, lk, _disk_free=20 * 1024 ** 3, _available=8 * 1024 ** 3)
            self.assertFalse(r["ok"]); self.assertTrue(r["checks"]["concurrent_fit"])
            self.assertFalse(G.acquire_fit_lock(lk))  # own live PID holds it -> refuse second fit
            G.release_fit_lock(lk)
            self.assertFalse(os.path.exists(lk))
            open(lk, "w").write("999999")  # stale lock from a dead PID
            self.assertTrue(G.preflight(d, lk, _disk_free=20 * 1024 ** 3, _available=8 * 1024 ** 3)["checks"]["concurrent_ok"])
            self.assertTrue(G.acquire_fit_lock(lk))
            G.release_fit_lock(lk)


class TestGuardedRunUnchanged(unittest.TestCase):
    def test_ok_and_breach(self):
        r = G.run_guarded(work_ok, (5,), "LR-01", "Mon", poll_s=0.05)
        self.assertEqual((r["status"], r["score"]), ("OK", 0.5))
        r = G.run_guarded(work_alloc, (600, 5), "RF-04", "Fri", limit_bytes=300 * MB, poll_s=0.05)
        self.assertEqual(r["status"], "RESOURCE_EXCEEDED"); self.assertIsNone(r["score"])
        exp = {("RF-04", "Fri")}
        self.assertEqual(G.grid_decision([r], exp)["decision"], "STOP")


if __name__ == "__main__":
    import platform, psutil  # noqa: E401
    print(f"python {platform.python_version()} psutil {psutil.__version__}")
    unittest.main(verbosity=2)
