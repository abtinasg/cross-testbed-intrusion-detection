#!/usr/bin/env python3
"""Tests for the PROPOSED non-daemon guarded worker (synthetic only). Run: python3 -I -B test_nondaemon_worker.py"""
import os
import sys
import tempfile
import threading
import time
import unittest

import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import guarded_worker_nd as W  # noqa: E402
import nd_test_jobs as J  # noqa: E402

MIB = 1024 ** 2


class Base(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.lock = os.path.join(self._td.name, "child.lock")

    def tearDown(self):
        self._td.cleanup()

    def run_nd(self, fn, args=(), **kw):
        kw.setdefault("poll_s", 0.05)
        return W.run_guarded_nd(fn, args, "SYN", "fold", child_lock_path=self.lock, **kw)

    def assertNoSurvivors(self, rec, extra=()):
        self.assertEqual(rec["survivors"], [])
        for pid in extra:
            self.assertFalse(psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE, pid)


class T1_EffectiveWorkersAndEquivalence(Base):
    def test_effective_workers_in_non_daemon_worker(self):
        rec = self.run_nd(J.effective_workers)
        self.assertEqual(rec["status"], "OK", rec["error"])
        s = rec["score"]
        self.assertFalse(s["daemon"]); self.assertTrue(s["pgid_is_self"])
        self.assertEqual((s["joblib_effective_n_jobs_4"], s["rf_partition_n_jobs"]), (4, 4))   # frozen RF n_jobs=4 honoured
        self.assertTrue(s["openmp"]); self.assertNoSurvivors(rec)

    def test_outputs_equal_to_in_process_fits(self):
        for cid in ("LR-02", "HGB-01", "RF-02"):
            rec = self.run_nd(J.fit_predict, (cid,))
            self.assertEqual(rec["status"], "OK", rec["error"])
            ref = J.fit_predict(cid)
            self.assertTrue(np.array_equal(rec["score"]["proba"], ref["proba"]), cid)
            if cid.startswith("RF"):
                for (f1, t1, v1), (f2, t2, v2) in zip(rec["score"]["trees"], ref["trees"]):
                    self.assertTrue(np.array_equal(f1, f2) and np.array_equal(t1, t2) and np.array_equal(v1, v2))


class T2_TreeMemoryMonitoring(Base):
    def test_footprint_function(self):
        fp0 = W.phys_footprint(os.getpid())
        a = np.ones(300 * MIB // 8)
        fp1 = W.phys_footprint(os.getpid())
        self.assertIsNotNone(fp0)
        self.assertGreater(fp1 - fp0, 250 * MIB)
        del a

    def test_descendant_memory_is_counted(self):
        rec = self.run_nd(J.allocate_tree, (300, 250, 2, 4.0))
        self.assertEqual(rec["status"], "OK", rec["error"])
        self.assertGreaterEqual(rec["descendants_seen"], 2)
        self.assertGreater(rec["peak_tree_footprint_bytes"], 700 * MIB)                         # 300 + 2 x 250 MiB
        self.assertNoSurvivors(rec, rec["score"]["kids"])


class T3_TerminationOfTheWholeTree(Base):
    def test_memory_breach_in_grandchild_kills_tree(self):
        rec = self.run_nd(J.hog_grandchild, limit_bytes=800 * MIB)
        self.assertEqual(rec["status"], "RESOURCE_EXCEEDED"); self.assertIsNone(rec["score"])
        self.assertGreaterEqual(rec["descendants_seen"], 1); self.assertNoSurvivors(rec)

    def test_timeout_kills_tree(self):
        rec = self.run_nd(J.sleep_with_grandchild, (3600,), time_limit_s=3)
        self.assertEqual(rec["status"], "TIME_EXCEEDED"); self.assertNoSurvivors(rec)

    def test_cancellation_kills_tree(self):
        t0 = time.monotonic()
        rec = self.run_nd(J.sleep_with_grandchild, (3600,), cancel=lambda: time.monotonic() - t0 > 2)
        self.assertEqual(rec["status"], "CANCELLED"); self.assertIsNone(rec["score"]); self.assertNoSurvivors(rec)

    def test_parent_side_exception_kills_tree(self):
        seen = {}
        t0 = time.monotonic()

        def boom():
            if time.monotonic() - t0 > 2:
                raise KeyboardInterrupt
            return False
        with self.assertRaises(KeyboardInterrupt):
            W.run_guarded_nd(J.sleep_with_grandchild, (3600,), "SYN", "f", child_lock_path=self.lock, poll_s=0.05,
                             on_start=lambda ident: seen.update(ident), cancel=boom)
        time.sleep(0.5)
        self.assertFalse(psutil.pid_exists(seen["pid"]) and psutil.Process(seen["pid"]).status() != psutil.STATUS_ZOMBIE)
        self.assertFalse(W.child_lock_busy(self.lock))

    def test_no_orphan_after_normal_completion(self):
        rec = self.run_nd(J.leave_grandchild_running)
        self.assertEqual(rec["status"], "OK", rec["error"])
        self.assertNoSurvivors(rec, [rec["score"]["grandchild"]])


class T4_NoConcurrentFitting(Base):
    def test_second_worker_refuses_while_first_holds_fit_lock(self):
        out = {}
        th = threading.Thread(target=lambda: out.update(a=self.run_nd(J.short_sleep, (4,))))
        th.start()
        t0 = time.monotonic()
        while not W.child_lock_busy(self.lock) and time.monotonic() - t0 < 30:
            time.sleep(0.05)
        self.assertTrue(W.child_lock_busy(self.lock))
        b = self.run_nd(J.short_sleep, (0,))
        th.join()
        self.assertEqual(out["a"]["status"], "OK")
        self.assertEqual(b["status"], "FIT_ERROR"); self.assertIn("ConcurrentFit", b["error"]); self.assertIsNone(b["score"])
        self.assertFalse(W.child_lock_busy(self.lock))


if __name__ == "__main__":
    import platform
    print(f"python {platform.python_version()} machine {platform.machine()}")
    unittest.main(verbosity=2)
