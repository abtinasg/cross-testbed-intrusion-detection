#!/usr/bin/env python3
"""RG8 unified guard: synthetic unit + integration tests. No real P1 data, no P2/P3, no project labels.
Run with the pinned interpreter in isolated mode:
  <PINNED_INTERPRETER:anaconda3/bin/python3> -I -B test_rg8_guard.py            (all except the full-route test)
  RG8_FULL_ROUTE=1 <PINNED_INTERPRETER:anaconda3/bin/python3> -I -B test_rg8_guard.py   (-I ignores PYTHON* vars only)
Groups: P policy, E environment, F preflight, M monitor/worker, R runner integration, X real frozen route (12 configs).
"""
import copy
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
import rg8_guard as RG  # noqa: E402
import rg8_runner as RR  # noqa: E402
import rg8_test_jobs as J  # noqa: E402
import g2_fixtures as X  # noqa: E402  (path added by rg8_runner)

S, P2, W = RR.S, RR.P2, RG.W
GIB = RG.GIB
BASE = json.load(open(RG.POLICY_PATH))
FULL_ROUTE = os.path.exists(os.path.join(HERE, ".run_full_route")) or os.environ.get("RG8_FULL_ROUTE") == "1"


def make_policy(d, name="policy.json", **over):
    """Unpinned TEST policy: the reviewed file with test-scale overrides (dotted keys)."""
    pol = copy.deepcopy(BASE)
    for k, v in over.items():
        sec, key = k.split("__")
        pol[sec][key] = v
    path = os.path.join(d, name)
    with open(path, "w") as fh:
        json.dump(pol, fh)
    return RG.load_policy(path, _unpinned_test_policy=True), path


TEST_LIMITS = dict(monitor__tree_mem_limit_bytes=int(1.0 * GIB), monitor__poll_s=0.1, monitor__max_sample_gap_s=5.0,
                   monitor__time_limit_s=120, preflight__min_available_bytes=int(0.25 * GIB), preflight__wait_s=0,
                   preflight__max_pressure_level=2)


class FakeSensors(RG.Sensors):
    """Real sensors unless scripted. Scripts are callables of the call count n (1-based) for that sensor."""

    def __init__(self, **scripts):
        self.s, self.n = scripts, {}

    def _call(self, name, real, *a):
        self.n[name] = self.n.get(name, 0) + 1
        f = self.s.get(name)
        return real(*a) if f is None else f(self.n[name], real, *a)

    def installed_ram(self):
        return self._call("installed_ram", super().installed_ram)

    def pressure(self):
        return self._call("pressure", super().pressure)

    def swap_used(self):
        return self._call("swap_used", super().swap_used)

    def available(self):
        return self._call("available", super().available)

    def disk_free(self, path):
        return self._call("disk_free", super().disk_free, path)

    def proc_footprint(self, pid):
        return self._call("proc_footprint", super().proc_footprint, pid)


def raise_after(k):
    def f(n, real, *a):
        if n > k:
            raise RG.SensorUnavailable("scripted sensor failure")
        return real(*a)
    return f


class Base(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory(); self.d = self._td.name
        self.pol, self.pol_path = make_policy(self.d, **TEST_LIMITS)
        self.lock = os.path.join(self.d, ".child.flock")
        self.n = 0

    def tearDown(self):
        self._td.cleanup()

    def guard(self, fn, args=(), pol=None, sensors=None, **kw):
        self.n += 1
        return RG.run_guarded_rg8(fn, args, policy=pol or self.pol, work_dir=self.d, child_lock_path=self.lock,
                                  sample_log_path=os.path.join(self.d, f"s{self.n}.jsonl"),
                                  stage_file=os.path.join(self.d, f"st{self.n}.jsonl"), sensors=sensors, **kw)

    def assertStopped(self, g, status, fclass):
        self.assertEqual((g["status"], g["failure_class"]), (status, fclass), g.get("error"))
        self.assertIsNone(g["payload"])
        self.assertEqual(g["survivors"], [])


# ============================================================================ P: policy
class P_Policy(Base):
    def test_P1_pinned_policy_loads_and_is_uniform(self):
        pol = RG.load_policy()
        self.assertTrue(pol["_pinned"]); self.assertEqual(pol["_sha256"], RG.POLICY_SHA256)
        self.assertEqual(pol["applies_to"], "ALL_12_F3_CONFIGURATIONS_UNIFORMLY")
        m = pol["monitor"]
        self.assertEqual((m["tree_mem_limit_bytes"], m["time_limit_s"]), (int(4.75 * GIB), 3600))

    def test_P2_modified_policy_refused_in_production(self):
        raw = open(RG.POLICY_PATH).read().replace("3600", "3601", 1)
        p = os.path.join(self.d, "mod.json"); open(p, "w").write(raw)
        with self.assertRaises(RG.RG8Refusal):
            RG.load_policy(p)

    def test_P3_family_specific_limit_refused(self):
        pol = copy.deepcopy(BASE); pol["monitor"]["tree_mem_limit_bytes_RF"] = 1
        p = os.path.join(self.d, "fam.json"); json.dump(pol, open(p, "w"))
        with self.assertRaisesRegex(RG.RG8Refusal, "family"):
            RG.load_policy(p, _unpinned_test_policy=True)

    def test_P4_unsafe_values_refused(self):
        for over in (dict(monitor__tree_mem_limit_bytes=7 * GIB), dict(preflight__min_free_disk_bytes=11 * GIB),
                     dict(monitor__max_sample_gap_s=0.05), dict(preflight__max_pressure_level=4)):
            with self.assertRaises(RG.RG8Refusal, msg=str(over)):
                make_policy(self.d, **over)
        pol = copy.deepcopy(BASE); pol["on_breach"]["retry"] = True
        p = os.path.join(self.d, "retry.json"); json.dump(pol, open(p, "w"))
        with self.assertRaises(RG.RG8Refusal):
            RG.load_policy(p, _unpinned_test_policy=True)


# ============================================================================ E: environment
class E_Environment(Base):
    def test_E1_pinned_interpreter_and_isolated_mode(self):
        env = RG.check_environment(self.pol)
        self.assertTrue(env["ok"], env)

    def test_E2_wrong_interpreter_refused_in_parent(self):
        env = RG.check_environment(self.pol, _executable="/usr/bin/python3")
        self.assertFalse(env["ok"]); self.assertFalse(env["checks"]["executable"])

    def test_E3_worker_rechecks_interpreter(self):
        g = self.guard(J.instant_job, _executable_for_env="/usr/bin/python3")
        self.assertStopped(g, "FIT_ERROR", "ENVIRONMENT")


# ============================================================================ F: preflight
class F_Preflight(Base):
    def test_F1_real_host_preflight_shape(self):
        pf = RG.preflight(self.pol, self.d)
        self.assertTrue(pf["checks"]["sensors_ok"]); self.assertGreaterEqual(pf["measurements"]["installed_ram"], 8 * GIB)

    def test_F2_each_condition_fails_closed(self):
        cases = dict(available=FakeSensors(available=lambda n, r: int(0.1 * GIB)),
                     pressure=FakeSensors(pressure=lambda n, r: 4),
                     disk=FakeSensors(disk_free=lambda n, r, p: 12 * GIB),
                     ram=FakeSensors(installed_ram=lambda n, r: 4 * GIB),
                     footprint_sensor=FakeSensors(proc_footprint=raise_after(0)),
                     swap_sensor=FakeSensors(swap_used=raise_after(0)))
        for name, sen in cases.items():
            pf = RG.preflight(self.pol, self.d, sensors=sen)
            self.assertFalse(pf["ok"], name)

    def test_F3_disk_requirement_includes_temp_and_swap_allowance(self):
        p, m = self.pol["preflight"], self.pol["monitor"]
        self.assertGreaterEqual(p["min_free_disk_bytes"], m["min_free_disk_bytes"] + p["temp_reserve_bytes"] + m["swap_growth_limit_bytes"])
        need = p["min_free_disk_bytes"]
        self.assertFalse(RG.preflight(self.pol, self.d, sensors=FakeSensors(disk_free=lambda n, r, q: need - 1))["ok"])
        self.assertTrue(RG.preflight(self.pol, self.d, sensors=FakeSensors(disk_free=lambda n, r, q: need,
                                                                          available=lambda n, r: 8 * GIB, pressure=lambda n, r: 1))["ok"])

    def test_F4_bounded_wait_then_pass(self):
        pol, _ = make_policy(self.d, "w.json", **dict(TEST_LIMITS, preflight__wait_s=60, preflight__recheck_s=15))
        sen = FakeSensors(available=lambda n, r: int(0.1 * GIB) if n < 3 else 8 * GIB, pressure=lambda n, r: 1)
        clock = [0.0]
        pf = RG.preflight_with_wait(pol, self.d, sensors=sen, _sleep=lambda s: clock.__setitem__(0, clock[0] + s), _clock=lambda: clock[0])
        self.assertTrue(pf["ok"]); self.assertEqual(pf["attempts"], 3)

    def test_F5_bounded_wait_then_refuse(self):
        pol, _ = make_policy(self.d, "w.json", **dict(TEST_LIMITS, preflight__wait_s=60, preflight__recheck_s=15))
        sen = FakeSensors(available=lambda n, r: int(0.1 * GIB))
        clock = [0.0]
        pf = RG.preflight_with_wait(pol, self.d, sensors=sen, _sleep=lambda s: clock.__setitem__(0, clock[0] + s), _clock=lambda: clock[0])
        self.assertFalse(pf["ok"]); self.assertEqual(pf["attempts"], 5)

    def test_F6_sensor_failure_is_not_waited_out(self):
        pol, _ = make_policy(self.d, "w.json", **dict(TEST_LIMITS, preflight__wait_s=600))
        pf = RG.preflight_with_wait(pol, self.d, sensors=FakeSensors(pressure=raise_after(0)), _sleep=lambda s: None)
        self.assertFalse(pf["ok"]); self.assertEqual(pf["attempts"], 1)

    def test_F7_live_worker_lock_fails_preflight(self):
        import fcntl
        fd = os.open(self.lock, os.O_CREAT | os.O_RDWR); fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            self.assertFalse(RG.preflight(self.pol, self.d, lock_paths=(self.lock,))["checks"]["locks_free"])
        finally:
            os.close(fd)


# ============================================================================ M: worker and monitor
class M_Monitor(Base):
    def test_M1_ok_nondaemon_own_group_full_parallelism(self):
        g = self.guard(J.ok_job, (21,))
        self.assertEqual(g["status"], "OK", g["error"])
        p = g["payload"]
        self.assertEqual((p["value"], p["daemon"], p["effective_n_jobs"], p["pgid_is_self"]), (42, False, 4, True))
        self.assertFalse(g["worker_daemon"]); self.assertEqual(g["survivors"], [])
        self.assertGreater(g["samples"], 0); self.assertGreater(g["peak_tree_footprint"], 0)
        self.assertEqual(g["stage_at_end"], "job_returned")
        lines = [json.loads(l) for l in open(os.path.join(self.d, "s1.jsonl"))]
        self.assertEqual(len(lines), g["samples"])
        self.assertTrue(all({"rss", "footprint", "swap_growth", "pressure", "available", "disk_free"} <= set(l) for l in lines))

    def test_M2_tree_memory_breach(self):
        g = self.guard(J.hog_job)
        self.assertStopped(g, "RESOURCE_EXCEEDED", "TREE_MEMORY")
        self.assertGreater(g["peak_tree_mem"], self.pol["monitor"]["tree_mem_limit_bytes"])
        self.assertEqual(g["stage_at_end"], "hog")

    def test_M3_grandchild_memory_counted(self):
        g = self.guard(J.grandchild_hog_job)
        self.assertStopped(g, "RESOURCE_EXCEEDED", "TREE_MEMORY")
        self.assertGreaterEqual(g["descendants_seen"], 1)

    def test_M4_footprint_unavailable_mid_run_fails_closed(self):
        g = self.guard(J.sleep_job, sensors=FakeSensors(proc_footprint=raise_after(5)))
        self.assertStopped(g, "FIT_ERROR", "MEASUREMENT_UNAVAILABLE")

    def test_M5_pressure_swap_disk_sensor_failures_fail_closed(self):
        for name in ("pressure", "swap_used", "disk_free", "available"):
            k = 2 if name == "swap_used" else 1                     # swap is read once before the worker starts
            g = self.guard(J.sleep_job, sensors=FakeSensors(**{name: raise_after(k + 2)}))
            self.assertStopped(g, "FIT_ERROR", "MEASUREMENT_UNAVAILABLE")

    def test_M6_swap_growth_breach(self):
        lim = self.pol["monitor"]["swap_growth_limit_bytes"]
        g = self.guard(J.sleep_job, sensors=FakeSensors(swap_used=lambda n, r: GIB + (0 if n < 5 else lim + 1)))
        self.assertStopped(g, "RESOURCE_EXCEEDED", "SWAP_GROWTH")

    def test_M7_sustained_critical_pressure_stops(self):
        pol, _ = make_policy(self.d, "c.json", **dict(TEST_LIMITS, monitor__critical_pressure_stop_s=1))
        g = self.guard(J.sleep_job, pol=pol, sensors=FakeSensors(pressure=lambda n, r: 4 if n > 3 else 1))
        self.assertStopped(g, "RESOURCE_EXCEEDED", "CRITICAL_PRESSURE")

    def test_M8_transient_critical_pressure_tolerated(self):
        pol, _ = make_policy(self.d, "c.json", **dict(TEST_LIMITS, monitor__critical_pressure_stop_s=1))
        g = self.guard(J.short_sleep_job, (3.0,), pol=pol, sensors=FakeSensors(pressure=lambda n, r: 4 if n in (3, 4) else 1))
        self.assertEqual(g["status"], "OK", g["error"]); self.assertEqual(g["max_pressure"], 4)

    def test_M9_disk_breach(self):
        g = self.guard(J.sleep_job, sensors=FakeSensors(disk_free=lambda n, r, p: 50 * GIB if n < 4 else 9 * GIB))
        self.assertStopped(g, "RESOURCE_EXCEEDED", "DISK")

    def test_M10_timeout(self):
        pol, _ = make_policy(self.d, "t.json", **dict(TEST_LIMITS, monitor__time_limit_s=2))
        g = self.guard(J.sleep_job, pol=pol)
        self.assertStopped(g, "TIME_EXCEEDED", "TIMEOUT")

    def test_M11_monitor_stall(self):
        def slow(n, real, *a):
            if n == 6:
                time.sleep(6.5)
            return real(*a)
        g = self.guard(J.sleep_job, sensors=FakeSensors(available=slow))
        self.assertStopped(g, "FIT_ERROR", "MONITOR_STALLED")

    def test_M12_breach_wins_over_delivered_result(self):
        g = self.guard(J.instant_job, sensors=FakeSensors(proc_footprint=lambda n, r, pid: 5 * GIB))
        self.assertStopped(g, "RESOURCE_EXCEEDED", "TREE_MEMORY")

    def test_M13_exclusive_worker(self):
        import threading
        out = {}
        t = threading.Thread(target=lambda: out.setdefault("a", self.guard(J.short_sleep_job, (4.0,))))
        t.start(); time.sleep(1.5)
        b = RG.run_guarded_rg8(J.instant_job, (), policy=self.pol, work_dir=self.d, child_lock_path=self.lock,
                               sample_log_path=os.path.join(self.d, "b.jsonl"), stage_file=os.path.join(self.d, "bst.jsonl"))
        t.join()
        self.assertStopped(b, "FIT_ERROR", "CONCURRENT_FIT")
        self.assertEqual(out["a"]["status"], "OK")

    def test_M14_no_tunable_limit_arguments(self):
        import inspect
        params = set(inspect.signature(RG.run_guarded_rg8).parameters) | set(inspect.signature(RR.PilotRunnerRG8.__init__).parameters)
        self.assertFalse({"limit_bytes", "time_limit_s", "poll_s", "retry", "max_rows", "subsample"} & params)


# ============================================================================ R: runner integration (G2 synthetic child)
class R_Runner(Base):
    def setUp(self):
        super().setUp()
        self.fx = X.fixture(os.path.join(self.d, "src"))
        self.work = os.path.join(self.d, "work")

    def runner(self, path=None, **kw):
        return RR.PilotRunnerRG8(self.work, self.fx.al_path, self.fx.paths(), self.fx.profile, phase="flat",
                                 _unpinned_test_policy_path=path or self.pol_path, **kw)

    def test_R1_ok_fit_is_durably_recorded(self):
        r = self.runner()
        rec = r.run(X.job("LR-01", "Wed"))
        self.assertEqual(rec["status"], "OK", rec.get("error"))
        led = r.records()
        self.assertEqual(len(led), 1)
        self.assertEqual(led[0]["policy_sha256"], self.pol["_sha256"]); self.assertFalse(led[0]["policy_pinned"])
        self.assertEqual(led[0]["environment"]["executable"], os.path.realpath(BASE["environment"]["executable"]))
        self.assertFalse(os.path.exists(r.inflight_path))
        want = RR.expected_counts(self.fx.profile, S.FoldSpec("Wed", None), "primary")
        self.assertEqual((led[0]["detail"]["n_train"], led[0]["detail"]["n_val"]), want)

    def test_R2_breach_in_every_stage_stops_without_retry(self):
        for stage in ("load", "assemble", "preprocess", "fit", "score"):
            with self.subTest(stage=stage):
                work = os.path.join(self.d, f"w_{stage}")
                r = RR.PilotRunnerRG8(work, self.fx.al_path, self.fx.paths(), self.fx.profile, phase="flat",
                                      _unpinned_test_policy_path=self.pol_path)
                rec = r.run(X.job("HGB-01", "Thu", faults=[(stage, "hog")]))
                self.assertEqual((rec["status"], rec["failure_class"], rec["score"]), ("RESOURCE_EXCEEDED", "TREE_MEMORY", None))
                self.assertEqual(rec["survivors"], [])
                self.assertTrue(r.stopped)
                with self.assertRaisesRegex(RR.PilotStop, "PILOT_STOPPED"):
                    r.run(X.job("HGB-01", "Thu"))                     # same key: no retry
                with self.assertRaisesRegex(RR.PilotStop, "PILOT_STOPPED"):
                    r.run(X.job("LR-01", "Thu"))                      # nothing else either
                self.assertEqual(len(r.records()), 1)

    def test_R3_preflight_failure_starts_nothing(self):
        r = self.runner(_test_sensors=FakeSensors(available=lambda n, q: int(0.1 * GIB)))
        with self.assertRaisesRegex(RR.PilotStop, "PREFLIGHT_FAIL"):
            r.run(X.job("LR-01", "Wed"))
        self.assertEqual(r.records(), []); self.assertFalse(os.path.exists(r.inflight_path))

    def test_R4_wrong_interpreter_refused_at_construction(self):
        with self.assertRaisesRegex(RR.PilotStop, "ENVIRONMENT_MISMATCH"):
            self.runner(_test_executable="/usr/bin/python3")

    def test_R5_duplicate_key_refused(self):
        r = self.runner()
        self.assertEqual(r.run(X.job("LR-01", "Wed"))["status"], "OK")
        with self.assertRaisesRegex(RR.PilotStop, "DUPLICATE_KEY"):
            r.run(X.job("LR-01", "Wed"))

    def test_R6_hidden_row_reduction_detected(self):
        real = RR.expected_counts
        RR.expected_counts = lambda prof, fold, target: (real(prof, fold, target)[0] + 1, real(prof, fold, target)[1])
        try:
            rec = self.runner().run(X.job("LR-01", "Wed"))
        finally:
            RR.expected_counts = real
        self.assertEqual((rec["status"], rec["failure_class"], rec["score"]), ("FIT_ERROR", "ROW_COUNT_MISMATCH", None))

    def test_R7_sigkill_of_parent_is_recovered_fail_closed(self):
        args = dict(work=self.work, al_path=self.fx.al_path, sources=self.fx.paths(), profile=dataclasses.asdict(self.fx.profile),
                    policy_path=self.pol_path, config_id="RF-01", outer="Wed", faults=[["fit", "sleep"]])
        drv = subprocess.Popen([sys.executable, "-I", "-B", os.path.join(HERE, "rg8_driver.py"), json.dumps(args)],
                               stdout=subprocess.PIPE, text=True)
        intent_p = os.path.join(self.work, "INFLIGHT.json")
        stage_p = os.path.join(self.work, "evidence", "RF-01__Wed", "rg8_stages.jsonl")
        t0 = time.time()
        while time.time() - t0 < 120:                     # worker has taken its flock once it writes a stage
            try:
                it = json.load(open(intent_p))
                if it.get("child") and os.path.exists(stage_p) and os.path.getsize(stage_p) > 0:
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.2)
        else:
            drv.kill(); self.fail("worker did not start")
        os.kill(drv.pid, signal.SIGKILL); drv.wait()
        child = it["child"]["pid"]
        busy = RR.PilotRunnerRG8(self.work, self.fx.al_path, self.fx.paths(), self.fx.profile, phase="flat",
                                 _unpinned_test_policy_path=self.pol_path)
        self.assertEqual(busy.recovery["state"], "INCOMPLETE_FIT_DETECTED")
        self.assertTrue(busy.recovery["orphan"]["orphan_worker_running"])     # reported, never adopted
        self.assertEqual(busy.recovery["orphan"]["child_identity"], "alive")
        rec = busy.records()[0]
        self.assertEqual((rec["status"], rec["failure_class"], rec["score"]), ("FIT_ERROR", "INCOMPLETE_PARENT_TERMINATED", None))
        with self.assertRaisesRegex(RR.PilotStop, "PILOT_STOPPED"):
            busy.run(X.job("LR-01", "Wed"))
        try:
            os.killpg(child, signal.SIGKILL)                                  # test cleanup of the orphan group
        except OSError:
            pass

    def test_R8_all_12_configurations_one_uniform_policy(self):
        r = self.runner()
        cfgs = sorted({k[0] for k in r.expected})
        self.assertEqual(len(cfgs), 12)
        recs = [r.run(X.job(c, "Thu")) for c in cfgs]
        self.assertTrue(all(x["status"] == "OK" for x in recs), [(x["model_id"], x.get("error")) for x in recs])
        self.assertEqual({x["policy_sha256"] for x in recs}, {self.pol["_sha256"]})
        self.assertEqual({x["family"] for x in recs}, {"LR", "HGB", "RF"})


# ============================================================================ K: V21 post-start cleanup repair
class K_PostStartCleanup(Base):
    """Any failure after the worker is created must kill and reap the whole worker tree (V21 repair)."""

    def _wait_grandchild(self, pid_file, timeout=30):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            try:
                with open(pid_file) as fh:
                    return int(fh.read())
            except (OSError, ValueError):
                time.sleep(0.05)
        self.fail("grandchild did not start")

    def _assert_tree_gone(self, pids, pgid):
        time.sleep(0.2)
        for pid in pids:
            alive = psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
            self.assertFalse(alive, f"pid {pid} survived")
        with self.assertRaises(ProcessLookupError):
            os.killpg(pgid, 0)
        self.assertFalse(W.child_lock_busy(self.lock), "worker flock still held")

    def test_K1_on_start_failure_kills_and_reaps_worker(self):
        seen = {}

        def on_start(ident):
            seen.update(ident)
            raise RuntimeError("injected on_start failure")
        with self.assertRaises(RuntimeError) as cm:
            self.guard(J.sleep_job, on_start=on_start)
        self.assertEqual(cm.exception.rg8_survivors, [])
        self._assert_tree_gone([seen["pid"]], seen["pid"])

    def test_K2_sample_log_creation_failure_kills_and_reaps_worker(self):
        seen = {}
        bad_log = os.path.join(self.d, "no_such_dir", "samples.jsonl")
        with self.assertRaises(FileNotFoundError) as cm:
            RG.run_guarded_rg8(J.sleep_job, (), policy=self.pol, work_dir=self.d, child_lock_path=self.lock,
                               sample_log_path=bad_log, stage_file=os.path.join(self.d, "st.jsonl"),
                               on_start=lambda ident: seen.update(ident))
        self.assertEqual(cm.exception.rg8_survivors, [])
        self._assert_tree_gone([seen["pid"]], seen["pid"])

    def test_K3_no_surviving_descendant_after_post_start_failures(self):
        """The failure happens only after the worker has spawned a grandchild: on_start failure and log failure."""
        for mode in ("on_start", "sample_log"):
            with self.subTest(mode=mode):
                pid_file = os.path.join(self.d, f"gc_{mode}.pid")
                seen = {}

                def on_start(ident):
                    seen.update(ident)
                    seen["gc"] = self._wait_grandchild(pid_file)
                    self.assertTrue(psutil.pid_exists(seen["gc"]))
                    if mode == "on_start":
                        raise RuntimeError("injected on_start failure after grandchild exists")
                log_path = os.path.join(self.d, "s_ok.jsonl") if mode == "on_start" else self.d   # a directory: open fails
                with self.assertRaises((RuntimeError, IsADirectoryError)) as cm:
                    RG.run_guarded_rg8(J.grandchild_sleep_job, (pid_file,), policy=self.pol, work_dir=self.d,
                                       child_lock_path=self.lock, sample_log_path=log_path,
                                       stage_file=os.path.join(self.d, f"st_{mode}.jsonl"), on_start=on_start)
                self.assertEqual(cm.exception.rg8_survivors, [])
                self._assert_tree_gone([seen["pid"], seen["gc"]], seen["pid"])


# ============================================================================ X: the real frozen route (12 configs)
@unittest.skipUnless(FULL_ROUTE, "set RG8_FULL_ROUTE=1 (or create .run_full_route) to run the 12-config frozen-route test")
class X_FullRoute(unittest.TestCase):
    """Complete frozen route (hashed load -> assembly -> F7 fit -> gate -> transform -> F3 estimator fit -> predict) of
    feasibility_harness.job_full_route on SMALL synthetic CSVs (scale 0.01), under the PINNED RG8 policy limits."""

    def test_X1_all_12_frozen_configs_under_pinned_policy(self):
        sys.path.insert(0, os.path.join(RG.BRIEF, "M1_8GB_FEASIBILITY"))
        import run_campaign as RC
        import feasibility_harness as H
        pol = RG.load_policy()
        with tempfile.TemporaryDirectory() as d:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                meta = RC.prepare(d, 0.01)
            tmp = os.path.join(d, "tmp"); os.makedirs(tmp)
            out = {}
            for cid in sorted(P2.B.load_bound_grid()):
                ev = os.path.join(d, "ev_" + cid); os.makedirs(ev)
                spec = dict(config_id=cid, variant="V1", outer="Tue", paths=meta["paths"], allowlist_path=meta["allowlist_path"],
                            profile=meta["profile"], evidence_dir=ev, tmp_dir=tmp)
                g = RG.run_guarded_rg8(H.job_full_route, (spec,), policy=pol, work_dir=d, child_lock_path=os.path.join(d, ".lock"),
                                       sample_log_path=os.path.join(ev, "s.jsonl"), stage_file=os.path.join(ev, "st.jsonl"))
                out[cid] = g
                self.assertEqual(g["status"], "OK", f"{cid}: {g['error']}")
                self.assertEqual(g["survivors"], [])
                pl = g["payload"]
                stages = [mk["stage"] for mk in pl["marks"]]
                for st in ("loaded_training_days", "preprocessing_fitted", "validation_loaded_transformed", "model_fitted", "predicted"):
                    self.assertIn(st, stages, cid)
                self.assertFalse(pl["model_info"]["child_daemon"])
                if pl["model_info"]["family"] == "RF":
                    self.assertEqual((pl["model_info"]["n_jobs_param"], pl["model_info"]["joblib_effective_n_jobs"]), (4, 4))
            self.assertEqual(len(out), 12)
            self.assertEqual({g["policy_sha256"] for g in out.values()}, {RG.POLICY_SHA256})
            with open(os.path.join(HERE, "test_runs", "X1_full_route_small_scale.json"), "w") as fh:
                json.dump({k: {kk: vv for kk, vv in v.items() if kk != "payload"} | dict(
                    family=v["payload"]["model_info"]["family"], n_train=v["payload"]["n_train"], fit_s=v["payload"]["fit_s"],
                    joblib_effective_n_jobs=v["payload"]["model_info"]["joblib_effective_n_jobs"]) for k, v in out.items()}, fh, indent=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
