#!/usr/bin/env python3
"""G2 tests: execution-state recovery after abrupt parent termination (SIGKILL of a real pilot parent process
mid-fit), crash after recording, corrupt ledger; no silent success, no retry. Synthetic only.
Run: python3 -I -B test_g2_recovery.py
"""
import dataclasses
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "PILOT_INTEGRATION_G1"))
import g2_fixtures as X  # noqa: E402
import pilot_runner_g2 as P2  # noqa: E402

G1, S = P2.G1, P2.S


class Base(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory(); self.d = self._td.name
        self.fx = X.fixture(os.path.join(self.d, "src"))
        self.work = os.path.join(self.d, "work")
        self.orphans = []

    def tearDown(self):
        for pid in self.orphans:
            try:
                psutil.Process(pid).kill()
            except psutil.Error:
                pass
        self._td.cleanup()

    def crash_parent_mid_fit(self, kill_child=False):
        args = dict(work=self.work, al_path=self.fx.al_path, sources=self.fx.paths(), config_id="LR-01", outer="Wed",
                    profile=dataclasses.asdict(self.fx.profile))
        parent = subprocess.Popen([sys.executable, "-I", "-B", os.path.join(HERE, "g2_crash_driver.py"), json.dumps(args)])
        pid_file = os.path.join(self.work, "INFLIGHT_CHILD.pid")
        t0 = time.time()
        while time.time() - t0 < 120:
            if os.path.exists(pid_file) and open(pid_file).read().strip():
                break
            time.sleep(0.1)
        else:
            parent.kill(); self.fail("child never started")
        time.sleep(2.0)                                        # child is now inside the sleeping fit stage
        child = int(open(pid_file).read())
        self.orphans.append(child)
        os.kill(parent.pid, signal.SIGKILL); parent.wait()       # abrupt parent termination
        if kill_child:
            psutil.Process(child).kill(); psutil.Process(child).wait(10)
        return parent.pid, child


class T1_AbruptParentTermination(Base):
    def test_orphan_alive_is_reported_and_nothing_is_retried(self):
        ppid, child = self.crash_parent_mid_fit()
        self.assertTrue(os.path.exists(os.path.join(self.work, "INFLIGHT.json")))
        with open(os.path.join(self.work, ".pilot_fit.lock")) as fh:
            self.assertEqual(int(fh.read()), ppid)               # stale lock of the dead parent remains
        rn = X.runner(self.fx, self.work)
        st = rn.recovery
        self.assertEqual(st["state"], "INCOMPLETE_FIT_DETECTED"); self.assertEqual(st["key"], ("LR-01", "Wed"))
        self.assertEqual(st["orphan"], dict(child_pid=child, alive=True))
        recs = rn.records()
        self.assertEqual(len(recs), 1)
        self.assertEqual((recs[0]["status"], recs[0]["failure_class"], recs[0]["score"]),
                         ("FIT_ERROR", "INCOMPLETE_PARENT_TERMINATED", None))
        self.assertEqual(rn.stop_reason(), "RECOVERY_DECISION_REQUIRED")
        for j in (X.job("LR-01", "Wed"), X.job("LR-02", "Tue")):
            with self.assertRaisesRegex(G1.PilotStop, "PILOT_STOPPED"):
                rn.run(j)
        self.assertEqual(rn.grid_status()["decision"], "STOP")
        rn2 = X.runner(self.fx, self.work)                        # re-opening never duplicates the record
        self.assertEqual(len(rn2.records()), 1); self.assertTrue(rn2.recovery["record_existed"])
        self.assertTrue(psutil.pid_exists(child))                 # the orphan is reported, not killed automatically

    def test_parent_and_child_dead(self):
        _, child = self.crash_parent_mid_fit(kill_child=True)
        rn = X.runner(self.fx, self.work)
        self.assertEqual(rn.recovery["orphan"], dict(child_pid=child, alive=False))
        self.assertEqual(rn.records()[0]["failure_class"], "INCOMPLETE_PARENT_TERMINATED")
        with self.assertRaisesRegex(G1.PilotStop, "RECOVERY_DECISION_REQUIRED"):
            rn.run(X.job("LR-01", "Tue"))


class T2_OtherCrashPoints(Base):
    def test_crash_after_record_before_intent_cleared(self):
        rn = X.runner(self.fx, self.work)
        rec = rn.run(X.job("LR-01", "Wed"))
        self.assertEqual(rec["status"], "OK", rec.get("error"))
        self.assertFalse(os.path.exists(rn.inflight_path)); self.assertFalse(os.path.exists(rn.child_pid_path))
        with open(rn.inflight_path, "w") as fh:                  # simulate a crash between append and intent removal
            json.dump(dict(key=["LR-01", "Wed"], started_epoch=time.time(), parent_pid=999999), fh)
        rn2 = X.runner(self.fx, self.work)
        self.assertEqual(rn2.recovery["state"], "INCOMPLETE_FIT_DETECTED"); self.assertTrue(rn2.recovery["record_existed"])
        self.assertEqual(len(rn2.records()), 1)                  # no extra record, and ...
        self.assertEqual(rn2.stop_reason(), "RECOVERY_DECISION_REQUIRED")   # ... still stopped (ambiguous state)

    def test_corrupt_ledger_stops(self):
        rn = X.runner(self.fx, self.work)
        self.assertEqual(rn.run(X.job("LR-01", "Wed"))["status"], "OK")
        with open(rn.ledger_path, "a") as fh:
            fh.write('{"model_id": "LR-01", "fold": "Tue", "status": "OK", "sco')   # torn write
        with self.assertRaisesRegex(G1.PilotStop, "LEDGER_CORRUPT"):
            X.runner(self.fx, self.work)
        self.assertTrue(os.path.exists(os.path.join(self.work, "PILOT_STOPPED.json")))

    def test_wrapper_error_leaves_intent_that_is_recovered_as_failure(self):
        rn = X.runner(self.fx, self.work)
        orig = P2.G.run_guarded
        try:
            P2.G.run_guarded = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
            with self.assertRaises(RuntimeError):
                rn.run(X.job("LR-01", "Wed"))
        finally:
            P2.G.run_guarded = orig
        self.assertFalse(os.path.exists(rn.lock_path))
        rn2 = X.runner(self.fx, self.work)
        self.assertEqual(rn2.records()[0]["failure_class"], "INCOMPLETE_PARENT_TERMINATED")
        self.assertEqual(rn2.stop_reason(), "WRAPPER_ERROR")

    def test_no_resume_or_retry_api(self):
        names = {n for n in dir(P2.PilotRunnerG2) if not n.startswith("__")}
        for bad in ("resume", "retry", "clear", "reset", "recover_and", "unstop", "force"):
            self.assertFalse(any(bad in n for n in names), names)

    def test_clean_completion_leaves_no_intent(self):
        rn = X.runner(self.fx, self.work)
        rec = rn.run(X.job("HGB-02", "Fri"))
        self.assertEqual(rec["status"], "OK", rec.get("error"))
        self.assertEqual(rn.check_recovery()["state"], "CLEAN")
        self.assertFalse(any(os.path.exists(p) for p in (rn.inflight_path, rn.child_pid_path, rn.lock_path)))


if __name__ == "__main__":
    import platform
    print(f"python {platform.python_version()} machine {platform.machine()} runner {P2.VERSION}")
    unittest.main(verbosity=2)
