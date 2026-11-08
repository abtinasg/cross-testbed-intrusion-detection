#!/usr/bin/env python3
"""Synthetic end-to-end tests of the G1 fail-closed pilot wrapper (guard v2.1 integration).
Every protected fit runs a real guarded child process on synthetic CSVs; no dataset file is opened and no real
model family is fitted. Run: python3 -I -B test_g1_guard_integration.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g1_fixtures as F  # noqa: E402
import pilot_guarded_runner as PR  # noqa: E402

G, S, R = PR.G, PR.S, PR.R
GIB = F.GIB


class Base(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory(); self.d = self._td.name
        self.fx = F.Fixture(os.path.join(self.d, "src"))
        self.work = os.path.join(self.d, "work")
        os.mkdir(self.work)

    def tearDown(self):
        self._td.cleanup()

    def assertFailed(self, rec, status, cls):
        self.assertEqual(rec["status"], status, rec.get("error"))
        self.assertEqual(rec["failure_class"], cls, rec.get("error"))
        self.assertIsNone(rec["score"])
        self.assertIsNone(rec["detail"])

    def assertLockFree(self, rn):
        self.assertFalse(os.path.exists(rn.lock_path))


class T1_HappyPathAndScope(Base):
    def test_ok_run_all_stages_in_monitored_child(self):
        rn = F.runner(self.fx, self.work)
        rec = rn.run(F.job())
        self.assertEqual(rec["status"], "OK", rec.get("error"))
        self.assertTrue(0.0 <= rec["score"] <= 1.0)
        det = rec["detail"]
        self.assertNotEqual(det["child_pid"], os.getpid())                       # data work ran in the child
        self.assertEqual([s for s, _ in det["stages"]], ["load", "assemble", "preprocess", "fit", "score"])
        self.assertEqual(det["source_verification"]["days"], ["Mon", "Tue", "Thu", "Fri"])
        self.assertEqual(det["n_train"], 60 + 50 + 40 + 80); self.assertEqual(det["val_day"], "Wed")
        self.assertGreater(rec["peak_rss_bytes"], 0); self.assertEqual(rec["limit_bytes"], G.DEFAULT_LIMIT_BYTES)
        self.assertTrue(rec["preflight_2"]["concurrent_fit"])                    # our own lock was seen ...
        self.assertEqual(rec["preflight_2"]["lock_holder"], os.getpid())         # ... and not treated as competing
        self.assertLockFree(rn); self.assertFalse(rn.stopped)
        self.assertTrue(any(f.startswith("R5_KAPPA_") for f in os.listdir(os.path.join(self.work, "evidence", "SYN-A__Wed"))))

    def test_tree_branch_and_inner_fold(self):
        rn = F.runner(self.fx, self.work, phase="tb", selected="SYN-A")
        rec = rn.run(F.job(outer="Wed", inner="Tue", branch="TREE"))
        self.assertEqual(rec["status"], "OK", rec.get("error"))
        self.assertEqual(rec["detail"]["source_verification"]["days"], ["Mon", "Thu", "Fri"])
        self.assertEqual(rec["detail"]["val_day"], "Tue")


class T2_ResourceChecksAndLock(Base):
    def test_preflight_fail_before_any_work(self):
        rn = F.runner(self.fx, self.work, probe=[(60 * GIB, int(6.9 * GIB))])
        with self.assertRaisesRegex(PR.PilotStop, "PREFLIGHT_FAIL"):
            rn.run(F.job())
        self.assertEqual(rn.records(), []); self.assertLockFree(rn); self.assertTrue(rn.stopped)
        with self.assertRaisesRegex(PR.PilotStop, "PILOT_STOPPED"):
            rn.run(F.job(outer="Tue"))

    def test_disk_preflight_fail(self):
        rn = F.runner(self.fx, self.work, probe=[(int(9.9 * GIB), 12 * GIB)])
        with self.assertRaisesRegex(PR.PilotStop, "PREFLIGHT_FAIL"):
            rn.run(F.job())

    def test_recheck_under_lock_fails_closed_and_releases_lock(self):
        rn = F.runner(self.fx, self.work, probe=[(60 * GIB, 12 * GIB), (60 * GIB, int(6.0 * GIB))])
        with self.assertRaisesRegex(PR.PilotStop, "PREFLIGHT_UNDER_LOCK_FAIL"):
            rn.run(F.job())
        self.assertEqual(rn.records(), []); self.assertLockFree(rn); self.assertTrue(rn.stopped)

    def test_live_competing_lock_refuses_and_is_not_removed(self):
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            rn = F.runner(self.fx, self.work)
            with open(rn.lock_path, "w") as fh:
                fh.write(str(other.pid))
            with self.assertRaisesRegex(PR.PilotStop, "PREFLIGHT_FAIL"):
                rn.run(F.job())
            with open(rn.lock_path) as fh:
                self.assertEqual(fh.read(), str(other.pid))                      # the other holder's lock is untouched
            self.assertFalse(G.acquire_fit_lock(rn.lock_path))                   # atomic exclusive acquisition refuses
            self.assertEqual(rn.records(), [])
        finally:
            other.kill(); other.wait()

    def test_stale_lock_is_replaced(self):
        dead = subprocess.Popen([sys.executable, "-c", "pass"]); dead.wait()
        rn = F.runner(self.fx, self.work)
        with open(rn.lock_path, "w") as fh:
            fh.write(str(dead.pid))
        probe = [(60 * GIB, 12 * GIB)] * 5
        rn.probe = probe
        # preflight #1 sees a lock whose PID is dead -> not a concurrent fit
        rec = rn.run(F.job())
        self.assertEqual(rec["status"], "OK", rec.get("error")); self.assertLockFree(rn)

    def test_wrapper_error_releases_lock_and_stops(self):
        rn = F.runner(self.fx, self.work)
        orig = G.run_guarded
        try:
            G.run_guarded = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
            with self.assertRaises(RuntimeError):
                rn.run(F.job())
        finally:
            G.run_guarded = orig
        self.assertLockFree(rn); self.assertTrue(rn.stopped)
        with open(rn.stop_path) as fh:
            self.assertEqual(json.load(fh)["reason"], "WRAPPER_ERROR")


class T3_StopOnFailure(Base):
    def test_memory_stop_at_every_stage(self):
        for stage in ("load", "assemble", "preprocess", "fit"):
            work = os.path.join(self.d, f"w_{stage}"); os.mkdir(work)
            rn = F.runner(self.fx, work, limit_bytes=512 * 1024 * 1024)               # test-only limit
            rec = rn.run(F.job(faults=((stage, "hog"),)))
            self.assertFailed(rec, "RESOURCE_EXCEEDED", "MEMORY")
            self.assertLockFree(rn); self.assertTrue(rn.stopped)

    def test_timeout(self):
        rn = F.runner(self.fx, self.work, time_limit_s=3)
        self.assertFailed(rn.run(F.job(faults=(("fit", "sleep"),))), "TIME_EXCEEDED", "TIMEOUT")

    def test_fit_failure(self):
        rn = F.runner(self.fx, self.work)
        self.assertFailed(rn.run(F.job(faults=(("fit", "raise"),))), "FIT_ERROR", "FIT")

    def test_invalid_score_never_returned(self):
        rn = F.runner(self.fx, self.work)
        self.assertFailed(rn.run(F.job(faults=(("score", "nan"),))), "FIT_ERROR", "INVALID_SCORE")

    def test_disk_exhausted_after_fit(self):
        rn = F.runner(self.fx, self.work, probe=[(60 * GIB, 12 * GIB), (60 * GIB, 12 * GIB), (5 * GIB, 12 * GIB)])
        self.assertFailed(rn.run(F.job()), "RESOURCE_EXCEEDED", "DISK")

    def test_parent_memory_outside_scope_is_accounted(self):
        vals = iter([100 * 1024 ** 2, 100 * 1024 ** 2 + PR.PARENT_MARGIN_BYTES + 1])
        orig = PR._parent_rss
        try:
            PR._parent_rss = lambda: next(vals)
            rec = F.runner(self.fx, self.work).run(F.job())
        finally:
            PR._parent_rss = orig
        self.assertFailed(rec, "RESOURCE_EXCEEDED", "PARENT_MEMORY_UNACCOUNTED")
        self.assertEqual(PR.PARENT_MARGIN_BYTES, int(7.0 * GIB) - int(6.5 * GIB))

    def test_numerical_review_stop_is_a_failure(self):
        for d in S.DAYS:
            self.fx.X[d][:, F.POS["Flow IAT Mean"]] = 1e9
        self.fx.X["Mon"][0, F.POS["Flow IAT Mean"]] = 1e9 + 1
        src = os.path.join(self.d, "ill"); fx = F.Fixture(src)
        for d in S.DAYS:
            F.write_day_csv(fx.files[d], self.fx.X[d], self.fx.att[d])
        fx2 = F.Fixture.__new__(F.Fixture); fx2.__dict__.update(fx.__dict__)
        al = {"P1": {F.C1.sha256_file(fx.files[d]): dict(day=d, file=d, bytes=os.path.getsize(fx.files[d])) for d in S.DAYS}, "forbidden": {}}
        fx2.al_path = os.path.join(src, "allow_ill.json")
        with open(fx2.al_path, "w") as fh:
            json.dump(al, fh)
        fx2.al = F.C1.load_allowlist(fx2.al_path)
        import dataclasses
        fx2.profile = dataclasses.replace(fx.profile, allowlist_sha256=fx2.al["_sha256"])
        rec = F.runner(fx2, self.work).run(F.job())
        self.assertFailed(rec, "FIT_ERROR", "NUMERICAL_REVIEW")
        ev = os.listdir(os.path.join(self.work, "evidence", "SYN-A__Wed"))
        self.assertTrue(any(f.startswith("R5_STOP_") and f.endswith(".json") for f in ev))


class T4_NoRetryAndExactKeys(Base):
    def test_no_duplicate_and_no_retry_after_failure(self):
        rn = F.runner(self.fx, self.work, config_ids=("SYN-A", "SYN-B"))
        self.assertEqual(rn.run(F.job()).get("status"), "OK")
        with self.assertRaisesRegex(PR.PilotStop, "DUPLICATE_KEY"):
            rn.run(F.job())
        rec = rn.run(F.job(config_id="SYN-B", faults=(("fit", "raise"),)))
        self.assertEqual(rec["status"], "FIT_ERROR")
        for j in (F.job(config_id="SYN-B"), F.job(outer="Tue")):
            with self.assertRaisesRegex(PR.PilotStop, "PILOT_STOPPED"):
                rn.run(j)
        # a new runner on the same pilot directory, even with a different dataset or target, is still stopped
        alt = F.Fixture(os.path.join(self.d, "alt"), sizes=(30, 30, 30, 30, 30))
        rn2 = F.runner(alt, self.work, config_ids=("SYN-A", "SYN-B"), target="sensitivity")
        with self.assertRaisesRegex(PR.PilotStop, "PILOT_STOPPED"):
            rn2.run(F.job(config_id="SYN-B"))

    def test_unexpected_keys_refused(self):
        rn = F.runner(self.fx, self.work)
        for j in (F.job(config_id="SYN-Z"), F.job(outer="Wed", inner="Tue")):
            with self.assertRaisesRegex(PR.PilotStop, "UNEXPECTED_KEY"):
                rn.run(j)
        self.assertEqual(rn.records(), [])

    def test_grid_complete_only_with_exact_unique_keys(self):
        rn = F.runner(self.fx, self.work, config_ids=("SYN-A", "SYN-B"))
        self.assertEqual(rn.grid_status()["decision"], "CONTINUE")
        for c in ("SYN-A", "SYN-B"):
            for d in S.DAYS:
                self.assertEqual(rn.run(F.job(config_id=c, outer=d)).get("status"), "OK")
                if (c, d) == ("SYN-A", "Mon"):
                    self.assertEqual(rn.grid_status()["decision"], "CONTINUE")
        st = rn.grid_status()
        self.assertEqual(st["decision"], "COMPLETE"); self.assertEqual(st["unique_seen"], 10)
        recs = rn.records()
        self.assertEqual(G.grid_decision(recs + [recs[0]], rn.expected)["decision"], "INVALID")       # duplicate
        bad = dict(recs[1], status="FIT_ERROR")                                                     # failure with score
        self.assertEqual(G.grid_decision([bad] + recs[2:] + recs[:1], rn.expected)["decision"], "INVALID")
        self.assertEqual(G.grid_decision(recs[:9], rn.expected)["decision"], "CONTINUE")


class T5_NotAuthorised(Base):
    def test_real_profile_or_real_allowlist_refused_without_touching_sources(self):
        missing = ["/nonexistent/monday.csv"]
        with self.assertRaisesRegex(PR.PilotStop, "NOT_AUTHORIZED"):
            PR.PilotRunner(self.work, R.P1_ALLOWLIST_PATH, missing, S.P1_PROFILE, phase="flat", config_ids=["HGB-01"])
        with self.assertRaisesRegex(PR.PilotStop, "NOT_AUTHORIZED"):
            PR.PilotRunner(self.work, R.P1_ALLOWLIST_PATH, missing, self.fx.profile, phase="flat", config_ids=["HGB-01"])
        self.assertFalse(PR.PILOT_TRAINING_AUTHORIZED); self.assertEqual(PR.AUTHORIZED_MODEL_FAMILIES, ())

    def test_real_model_family_refused(self):
        rn = F.runner(self.fx, self.work, config_ids=("HGB-01",))
        with self.assertRaisesRegex(PR.PilotStop, "only synthetic"):
            rn.run(PR.JobSpec("HGB-01", S.FoldSpec("Wed"), "TREE", "HistGradientBoosting"))

    def test_guard_thresholds_unchanged(self):
        self.assertEqual(G.DEFAULT_LIMIT_BYTES, int(6.5 * GIB)); self.assertEqual(G.MIN_AVAILABLE_MEM_BYTES, int(7.0 * GIB))
        self.assertEqual(G.MIN_FREE_DISK_BYTES, 10 * GIB); self.assertEqual(G.DEFAULT_TIME_LIMIT_S, 3 * 3600)
        self.assertEqual(R.sha256_path(G.__file__), PR.GUARD_SHA256)


if __name__ == "__main__":
    import platform
    print(f"python {platform.python_version()} machine {platform.machine()} runner {PR.VERSION}")
    unittest.main(verbosity=2)
