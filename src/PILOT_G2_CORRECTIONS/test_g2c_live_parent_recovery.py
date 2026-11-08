#!/usr/bin/env python3
"""G2-C regression tests: live-parent-aware recovery, simultaneous runners, real SIGKILL, orphan workers, PID reuse,
stale locks, repeated/concurrent recovery attempts, durable ledger. Synthetic data only.
Run: python3 -I -B test_g2c_live_parent_recovery.py
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
import pilot_runner_g2c as C  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "PILOT_BINDING_G2"))
import g2_fixtures as X  # noqa: E402

S, P2, W = C.S, C.P2, C.W
DRIVER = os.path.join(HERE, "g2c_driver.py")


class Base(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory(); self.d = self._td.name
        self.fx = X.fixture(os.path.join(self.d, "src"))
        self.work = os.path.join(self.d, "work")
        self.kill_later = []

    def tearDown(self):
        for pid in self.kill_later:
            try:
                psutil.Process(pid).kill()
            except psutil.Error:
                pass
        self._td.cleanup()

    def runner(self, **kw):
        kw.setdefault("poll_s", 0.05)
        return C.PilotRunnerG2C(self.work, self.fx.al_path, self.fx.paths(), self.fx.profile, phase="flat",
                                probe=X.F.ok_probe(2000), **kw)

    def args(self, **kw):
        a = dict(work=self.work, al_path=self.fx.al_path, sources=self.fx.paths(), profile=dataclasses.asdict(self.fx.profile),
                 mode="fit", config_id="LR-01", outer="Wed", faults=[["fit", "sleep"]])
        a.update(kw)
        return a

    def spawn(self, **kw):
        return subprocess.Popen([sys.executable, "-I", "-B", DRIVER, json.dumps(self.args(**kw))], stdout=subprocess.PIPE, text=True)

    def wait_worker(self, timeout=120):
        t0 = time.time()
        while time.time() - t0 < timeout:
            it = self._intent()
            if it and it.get("child") and W.child_lock_busy(os.path.join(self.work, ".pilot_child.flock")):
                time.sleep(1.0)
                return it
            time.sleep(0.1)
        self.fail("worker never started")

    def _intent(self):
        try:
            with open(os.path.join(self.work, "INFLIGHT.json")) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    def n_records(self):
        p = os.path.join(self.work, "pilot_ledger.jsonl")
        return 0 if not os.path.exists(p) else sum(1 for l in open(p) if l.strip())


class T1_LiveParent(Base):
    def test_live_parent_is_active_execution_then_real_sigkill_is_recovered_once(self):
        parent = self.spawn()
        intent = self.wait_worker()
        self.kill_later.append(intent["child"]["pid"])
        self.assertEqual(C.identity_alive(intent["parent"]), "alive")
        rn = self.runner()                                                   # second runner while the fit is live
        self.assertEqual(rn.recovery["state"], "ACTIVE_EXECUTION")
        self.assertEqual(rn.recovery["parent"], "alive")
        with self.assertRaisesRegex(C.PilotStop, "CONCURRENT_RUNNER"):
            rn.run(P2.BoundJob("LR-02", S.FoldSpec("Tue")))
        self.assertEqual(self.n_records(), 0); self.assertFalse(rn.stopped)    # no false failure, pilot not stopped
        self.assertEqual(rn.check_recovery()["state"], "ACTIVE_EXECUTION")
        os.kill(parent.pid, signal.SIGKILL); parent.wait()                    # genuine abrupt termination
        self.assertEqual(C.identity_alive(intent["parent"]), "dead")
        rn2 = self.runner()
        st = rn2.recovery
        self.assertEqual(st["state"], "INCOMPLETE_FIT_DETECTED"); self.assertEqual(st["key"], ("LR-01", "Wed"))
        self.assertTrue(st["orphan"]["orphan_worker_running"]); self.assertEqual(st["orphan"]["child_identity"], "alive")
        self.assertEqual(rn2.stop_reason(), "RECOVERY_DECISION_REQUIRED")
        recs = rn2.records()
        self.assertEqual(len(recs), 1)
        self.assertEqual((recs[0]["failure_class"], recs[0]["score"]), ("INCOMPLETE_PARENT_TERMINATED", None))
        for _ in range(3):                                                    # repeated recovery attempts
            self.assertEqual(self.runner().recovery["record_existed"], True)
        self.assertEqual(self.n_records(), 1)
        with self.assertRaisesRegex(C.PilotStop, "PILOT_STOPPED"):
            rn2.run(P2.BoundJob("LR-02", S.FoldSpec("Tue")))                  # nothing new while the orphan runs
        psutil.Process(intent["child"]["pid"]).kill(); psutil.Process(intent["child"]["pid"]).wait(10)
        st = self.runner().recovery
        self.assertFalse(st["orphan"]["orphan_worker_running"]); self.assertEqual(st["orphan"]["child_identity"], "dead")
        self.assertEqual(self.n_records(), 1)


class T2_Concurrency(Base):
    def test_simultaneous_runners_only_one_executes(self):
        start = time.time() + 6
        ps = [self.spawn(config_id=c, outer="Wed", time_limit_s=8, start_at=start) for c in ("LR-01", "LR-02", "LR-03", "HGB-01")]
        outs = [json.loads(p.communicate(timeout=300)[0].strip().splitlines()[-1]) for p in ps]
        ran = [o for o in outs if "status" in o]
        refused = [o for o in outs if "CONCURRENT_RUNNER" in o.get("pilot_stop", "") or o.get("recovery_state") == "ACTIVE_EXECUTION" and "status" not in o]
        self.assertEqual(len(ran), 1, outs)
        self.assertEqual(ran[0]["status"], "TIME_EXCEEDED")
        self.assertEqual(len(refused), 3, outs)
        self.assertEqual(self.n_records(), 1)                                   # only the executing runner recorded
        self.assertFalse(W.child_lock_busy(os.path.join(self.work, ".pilot_child.flock")))

    def test_concurrent_recovery_attempts_append_exactly_one_record(self):
        parent = self.spawn()
        intent = self.wait_worker()
        os.kill(parent.pid, signal.SIGKILL); parent.wait()
        psutil.Process(intent["child"]["pid"]).kill(); psutil.Process(intent["child"]["pid"]).wait(10)
        start = time.time() + 5
        ps = [self.spawn(mode="construct", start_at=start) for _ in range(5)]
        outs = [json.loads(p.communicate(timeout=120)[0].strip().splitlines()[-1]) for p in ps]
        self.assertEqual(self.n_records(), 1, outs)
        self.assertIn("INCOMPLETE_FIT_DETECTED", [o.get("recovery_state") for o in outs])
        self.assertEqual(self.runner().stop_reason(), "RECOVERY_DECISION_REQUIRED")


class T3_IdentityAndLocks(Base):
    def test_pid_reuse_cannot_fake_a_live_parent_or_orphan(self):
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        try:
            os.makedirs(self.work)
            fake = dict(pid=other.pid, create_time=psutil.Process(other.pid).create_time() - 1000.0, boot_time=psutil.boot_time())
            with open(os.path.join(self.work, "INFLIGHT.json"), "w") as fh:
                json.dump(dict(key=["LR-01", "Wed"], parent=fake, child=fake, started_epoch=time.time()), fh)
            self.assertEqual(C.identity_alive(fake), "pid_reused")
            rn = self.runner()
            st = rn.recovery
            self.assertEqual(st["state"], "INCOMPLETE_FIT_DETECTED")
            self.assertEqual(st["parent_state"], "pid_reused"); self.assertEqual(st["orphan"]["child_identity"], "pid_reused")
            self.assertFalse(st["orphan"]["orphan_worker_running"]); self.assertEqual(self.n_records(), 1)
        finally:
            other.kill(); other.wait()

    def test_stale_locks_do_not_block_or_mislead(self):
        dead = subprocess.Popen([sys.executable, "-c", "pass"]); dead.wait()
        os.makedirs(self.work)
        for name in (".pilot_fit.lock", ".pilot_parent.flock", ".pilot_child.flock"):
            with open(os.path.join(self.work, name), "w") as fh:
                fh.write(str(dead.pid))                                      # leftovers of a dead process
        rn = self.runner()
        self.assertEqual(rn.recovery["state"], "CLEAN")
        rec = rn.run(P2.BoundJob("LR-01", S.FoldSpec("Thu")))
        self.assertEqual(rec["status"], "OK", rec.get("error")); self.assertEqual(rec["survivors"], [])
        self.assertFalse(rec["worker_daemon"])

    def test_own_wrapper_error_is_recovered_as_incomplete(self):
        rn = self.runner()
        orig = W.run_guarded_nd
        try:
            W.run_guarded_nd = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
            with self.assertRaises(RuntimeError):
                rn.run(P2.BoundJob("LR-01", S.FoldSpec("Wed")))
        finally:
            W.run_guarded_nd = orig
        st = rn.check_recovery()
        self.assertEqual(st["state"], "INCOMPLETE_FIT_DETECTED"); self.assertEqual(st["parent_state"], "this_process_earlier_run")
        self.assertEqual(self.n_records(), 1); self.assertEqual(rn.stop_reason(), "WRAPPER_ERROR")

    def test_torn_intent_fails_closed(self):
        os.makedirs(self.work)
        with open(os.path.join(self.work, "INFLIGHT.json"), "w") as fh:
            fh.write('{"key": ["LR-01", "We')
        rn = self.runner()
        self.assertTrue(rn.recovery.get("intent_unreadable")); self.assertEqual(rn.stop_reason(), "RECOVERY_DECISION_REQUIRED")
        self.assertEqual(self.n_records(), 0)

    def test_clean_run_is_durable_and_leaves_nothing_behind(self):
        rn = self.runner()
        rec = rn.run(P2.BoundJob("HGB-01", S.FoldSpec("Fri")))
        self.assertEqual(rec["status"], "OK", rec.get("error"))
        self.assertEqual(rec["g2_sha256"], C.G2_SHA256)
        self.assertEqual(rn.check_recovery()["state"], "CLEAN")
        for p in ("INFLIGHT.json", "INFLIGHT.json.tmp", "INFLIGHT_CHILD.pid"):
            self.assertFalse(os.path.exists(os.path.join(self.work, p)))
        self.assertFalse(W.child_lock_busy(rn.child_lock_path))
        self.assertEqual(json.loads(open(rn.ledger_path).read().strip())["model_id"], "HGB-01")


if __name__ == "__main__":
    import platform
    print(f"python {platform.python_version()} machine {platform.machine()} runner {C.VERSION}")
    unittest.main(verbosity=2)
