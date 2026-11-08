#!/usr/bin/env python3
"""A1 amendment validation. Synthetic only: no real P1 rows are parsed, no real fit is run.
  <PINNED_INTERPRETER:anaconda3/bin/python3> -I -B test_a1_smoke.py
Groups: P policy derivation, F preflight gate, O orchestrator refusals, J job scope + equivalence with the measured route.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_p1_smoke as SM  # noqa: E402

RG, S, P2, J = SM.RG, SM.S, SM.P2, SM.J
GIB = SM.GIB
EVID = os.path.join(SM.BRIEF, "RG8_UNIFIED_GUARD_V20", "RG8_POLICY_EVIDENCE.json")
ceil_q = lambda x: math.ceil(round(x / 0.25, 9)) * 0.25


class P_Policy(unittest.TestCase):
    def setUp(self):
        self.a1 = RG.load_policy(SM.A1_POLICY_PATH, expected_sha256=SM.A1_POLICY_SHA256)
        self.v20 = RG.load_policy()
        self.ev = json.load(open(EVID))

    def test_P1_pinned_and_loads_under_unchanged_v21_loader(self):
        self.assertTrue(self.a1["_pinned"])
        self.assertEqual(RG.sha256_file(RG.__file__), SM.V21_GUARD_SHA256)
        self.assertEqual(self.v20["_sha256"], SM.V20_POLICY_SHA256)

    def test_P2_values_follow_the_stated_rules_from_the_evidence(self):
        v1 = {k: r for k, r in self.ev["jobs"].items() if ":V1" in k}
        D = max(max(r["rss"], r["footprint"] or 0) for r in v1.values())
        self.assertAlmostEqual(D, 2.393, places=3)
        self.assertEqual(self.a1["preflight"]["min_available_bytes"], int(ceil_q(D) * GIB))           # 2.5
        self.assertEqual(self.a1["monitor"]["tree_mem_limit_bytes"], int(ceil_q(1.25 * D) * GIB))     # 3.0
        self.assertEqual(self.a1["monitor"]["swap_growth_limit_bytes"], int(ceil_q(self.ev["max_swap_growth_any"]) * GIB))  # 1.5
        self.assertEqual(self.ev["critical_pressure_samples"], 0)
        m, p = self.a1["monitor"], self.a1["preflight"]
        self.assertGreaterEqual(p["min_free_disk_bytes"], m["min_free_disk_bytes"] + p["temp_reserve_bytes"] + m["swap_growth_limit_bytes"])

    def test_P3_only_the_start_threshold_is_relaxed_everything_else_equal_or_stricter(self):
        a, b = self.a1, self.v20
        self.assertLess(a["preflight"]["min_available_bytes"], b["preflight"]["min_available_bytes"])
        self.assertLess(a["preflight"]["max_pressure_level"], b["preflight"]["max_pressure_level"])
        for k in ("tree_mem_limit_bytes", "swap_growth_limit_bytes", "critical_pressure_stop_s"):
            self.assertLess(a["monitor"][k], b["monitor"][k], k)
        for k in ("min_free_disk_bytes", "time_limit_s", "poll_s", "max_sample_gap_s", "fsync_s"):
            self.assertEqual(a["monitor"][k], b["monitor"][k], k)
        for k in ("min_free_disk_bytes", "temp_reserve_bytes", "wait_s", "recheck_s"):
            self.assertEqual(a["preflight"][k], b["preflight"][k], k)
        for k in ("environment", "host", "worker", "on_breach", "applies_to"):
            self.assertEqual(a[k], b[k], k)

    def test_P4_no_measured_v1_fit_would_have_been_stopped_by_the_tighter_in_fit_limits(self):
        m = self.a1["monitor"]
        for k, r in self.ev["jobs"].items():
            if ":V1" not in k:
                continue
            self.assertLessEqual(max(r["rss"], r["footprint"] or 0) * GIB, m["tree_mem_limit_bytes"], k)
            self.assertLessEqual(r["swap_growth"] * GIB, m["swap_growth_limit_bytes"], k)
            self.assertLessEqual(r["elapsed_s"], m["time_limit_s"], k)


class _Fake:
    def __init__(self, available, pressure=1, disk=50 * GIB):
        self.a, self.p, self.d = available, pressure, disk
    installed_ram = lambda self: 8 * GIB
    pressure = lambda self: self.p
    available = lambda self: self.a
    swap_used = lambda self: 0
    disk_free = lambda self, path: self.d
    proc_footprint = lambda self, pid: 1
    proc_rss = lambda self, pid: 1


class F_Preflight(unittest.TestCase):
    def setUp(self):
        self.pol = RG.load_policy(SM.A1_POLICY_PATH, expected_sha256=SM.A1_POLICY_SHA256)

    def _ok(self, **kw):
        return RG.preflight(self.pol, HERE, sensors=_Fake(**kw))["ok"]

    def test_F1_threshold_boundary(self):
        self.assertFalse(self._ok(available=int(2.49 * GIB)))
        self.assertTrue(self._ok(available=int(2.5 * GIB)))

    def test_F2_start_pressure_must_be_normal(self):
        self.assertFalse(self._ok(available=3 * GIB, pressure=2))

    def test_F3_disk(self):
        self.assertFalse(self._ok(available=3 * GIB, disk=int(13.9 * GIB)))

    def test_F4_recorded_probes_against_A1(self):
        """The two headless probes of 2026-11-06 (first official reading): v20 fails both, A1 passes both."""
        for fn, exp in (("probe_2026-11-06T215715Z.json", True), ("probe_2026-11-06T220928Z.json", True)):
            d = json.load(open(os.path.join(SM.V21, "preflight_probes", fn)))
            av = d["official_preflight"]["first"]["measurements"]["available"]
            self.assertLess(av, 3 * GIB, fn)
            self.assertEqual(av >= self.pol["preflight"]["min_available_bytes"], exp, fn)


class O_Orchestrator(unittest.TestCase):
    def test_O1_wrong_or_missing_token_refused_before_anything_is_read_or_created(self):
        existed = os.path.exists(SM.RUN_DIR)
        for args in (["--execute"], ["--execute", "--authorize", "GO"], ["--execute", "--authorize", J.AUTH_PREFIX + "0" * 12]):
            r = subprocess.run([sys.executable, "-I", "-B", os.path.join(HERE, "run_p1_smoke.py")] + args,
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn("REFUSED", r.stdout)
        self.assertEqual(os.path.exists(SM.RUN_DIR), existed)

    def test_O2_token_is_bound_to_the_policy_hash(self):
        self.assertEqual(J.expected_token(SM.A1_POLICY_SHA256), J.AUTH_PREFIX + SM.A1_POLICY_SHA256[:12])


class J_Job(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(SM.BRIEF, "M1_8GB_FEASIBILITY"))
        import contextlib, io
        import run_campaign as RC
        cls.d = tempfile.TemporaryDirectory()
        with contextlib.redirect_stdout(io.StringIO()):
            cls.meta = RC.prepare(cls.d.name, 0.01)
        cls.cfg = P2.B.load_bound_grid()["LR-01"]
        cls.pol = RG.load_policy(SM.A1_POLICY_PATH, expected_sha256=SM.A1_POLICY_SHA256)

    @classmethod
    def tearDownClass(cls):
        cls.d.cleanup()

    def _spec(self, tag, **over):
        root = os.path.join(self.d.name, tag)
        for sub in ("ev", "tmp", "sc"):
            os.makedirs(os.path.join(root, sub))
        s = dict(config_id="LR-01", config_sha256=self.cfg.config_sha256, outer="Tue", paths=self.meta["paths"],
                 allowlist_path=self.meta["allowlist_path"], profile=self.meta["profile"], policy_sha256=SM.A1_POLICY_SHA256,
                 evidence_dir=os.path.join(root, "ev"), tmp_dir=os.path.join(root, "tmp"), score_dir=os.path.join(root, "sc"),
                 authorization=None)
        s.update(over)
        return root, s

    def test_J1_scope_and_authorization_refusals(self):
        import dataclasses
        for tag, over, msg in (("j1a", dict(config_id="LR-02"), "A1 scope"), ("j1b", dict(outer="Wed"), "A1 scope"),
                               ("j1c", dict(profile=dataclasses.asdict(S.P1_PROFILE), allowlist_path="/nonexistent"), "NOT_AUTHORIZED")):
            _, s = self._spec(tag, **over)
            with self.assertRaises(RuntimeError) as cm:
                J.job_lr01_smoke(s)
            self.assertIn(msg, str(cm.exception))

    def test_J2_guarded_small_scale_run_equals_the_measured_route_and_persists_write_once(self):
        import numpy as np
        import feasibility_harness as H
        root, s = self._spec("j2")
        g = RG.run_guarded_rg8(J.job_lr01_smoke, (s,), policy=self.pol, work_dir=root, child_lock_path=os.path.join(root, ".lock"),
                               sample_log_path=os.path.join(root, "ev", "s.jsonl"), stage_file=os.path.join(root, "ev", "st.jsonl"))
        self.assertEqual(g["status"], "OK", g["error"])
        self.assertEqual(g["survivors"], [])
        self.assertEqual(g["policy_sha256"], SM.A1_POLICY_SHA256)
        pl = g["payload"]
        prof = S.SourceProfile(**{k: (tuple(map(tuple, v)) if isinstance(v, list) else v) for k, v in self.meta["profile"].items()})
        self.assertEqual((pl["n_train"], pl["n_val"]), SM.RR.expected_counts(prof, S.FoldSpec("Tue"), "primary"))
        sc = np.load(os.path.join(s["score_dir"], pl["persisted"]["scores"]["file"]))
        self.assertEqual(sc.dtype, np.float64)
        self.assertEqual(sc.shape[0], pl["n_val"])
        for meta in pl["persisted"].values():
            self.assertEqual(RG.sha256_file(os.path.join(s["score_dir"], meta["file"])), meta["sha256"])
        with self.assertRaises(FileExistsError):
            J._write_once_npy(os.path.join(s["score_dir"], pl["persisted"]["scores"]["file"]), sc)
        self.assertEqual(os.listdir(s["tmp_dir"]), [])
        # equivalence: the measured harness route (V1) gives the same AP and row counts on the same data
        root2, s2 = self._spec("j2h")
        hs = dict(config_id="LR-01", variant="V1", outer="Tue", paths=self.meta["paths"], allowlist_path=self.meta["allowlist_path"],
                  profile=self.meta["profile"], evidence_dir=s2["evidence_dir"], tmp_dir=s2["tmp_dir"])
        h = RG.run_guarded_rg8(H.job_full_route, (hs,), policy=self.pol, work_dir=root2, child_lock_path=os.path.join(root2, ".lock"),
                               sample_log_path=os.path.join(root2, "ev", "s.jsonl"), stage_file=os.path.join(root2, "ev", "st.jsonl"))
        self.assertEqual(h["status"], "OK", h["error"])
        self.assertEqual((h["payload"]["n_train"], h["payload"]["n_val"]), (pl["n_train"], pl["n_val"]))
        self.assertEqual(h["payload"]["synthetic_ap_no_signal"], pl["ap_heldout"])
        self.assertEqual(h["payload"]["model_info"]["n_iter"], pl["n_iter"])
        with open(os.path.join(HERE, "test_runs", "J2_small_scale_record.json"), "w") as fh:
            json.dump(dict(smoke={k: v for k, v in g.items() if k != "payload"} | dict(ap=pl["ap_heldout"], n_iter=pl["n_iter"],
                           n_train=pl["n_train"], n_val=pl["n_val"], stages=[m_["stage"] for m_ in pl["marks"]]),
                           harness_ap=h["payload"]["synthetic_ap_no_signal"]), fh, indent=1, default=str)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "test_runs"), exist_ok=True)
    unittest.main(verbosity=2)
