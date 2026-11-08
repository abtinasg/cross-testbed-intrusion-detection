#!/usr/bin/env python3
"""G2 tests: execution plan bound to frozen F3; no family/branch substitution; exact grid completion; proposed
real-model adapter (constructor validation only, no training). Synthetic data only.
Run: python3 -I -B test_g2_f3_binding_and_adapter.py
"""
import csv
import dataclasses
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "PILOT_INTEGRATION_G1"))
import g2_fixtures as X  # noqa: E402
import pilot_runner_g2 as P2  # noqa: E402

B, A, AP, G, S, R, G1 = P2.B, P2.A, P2.AP, P2.G, P2.S, P2.R, P2.G1


def f3_rows():
    with open(B.F3_PATH) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


class T1_F3Binding(unittest.TestCase):
    def test_exactly_12_bound_configs_with_frozen_content(self):
        g = B.load_bound_grid()
        self.assertEqual(tuple(g), B.APPROVED_IDS)
        rows = {r["config_id"]: r for r in f3_rows()}
        for cid, s in g.items():
            r = rows[cid]
            self.assertEqual(s.family, r["family"]); self.assertEqual(s.estimator, r["estimator"])
            self.assertEqual(s.hyperparameters, json.loads(r["hyperparameters_json"]))
            self.assertEqual(s.preprocessing_variant, r["preprocessing_variant"])
            self.assertEqual(s.branch, "LR" if s.family == "LR" else "TREE")
            self.assertEqual(s.config_sha256, B.EXPECTED_CONFIG_SHA256[cid])
        self.assertEqual(sum(s.family == "LR" for s in g.values()), 3)
        self.assertEqual(sum(s.family == "HGB" for s in g.values()), 5)
        self.assertEqual(sum(s.family == "RF" for s in g.values()), 4)
        self.assertEqual(R.sha256_path(B.F3_PATH), B.F3_SHA256)
        self.assertEqual(B.F3_SHA256, R.PINNED_INPUTS["next_workstream/CONFIG_GRID_F3_v1.csv"])

    def _tampered(self, mutate):
        rows = f3_rows(); rows = mutate(rows)
        return rows

    def test_unexpected_missing_or_altered_configs_rejected(self):
        def change_c(rows):
            r = dict(rows[0]); hp = json.loads(r["hyperparameters_json"]); hp["C"] = 0.02
            r["hyperparameters_json"] = json.dumps(hp); return [r] + rows[1:]
        def add_rf05(rows):
            r = dict(rows[-1], config_id="RF-05", complexity_rank="13"); return rows + [r]
        cases = {
            "missing": lambda rows: rows[:-1],
            "unexpected": add_rf05,
            "duplicate": lambda rows: rows[:-1] + [rows[0]],
            "altered hyperparameter": change_c,
            "family/branch swap": lambda rows: [dict(rows[3], preprocessing_variant="P-LR")] + rows[:3] + rows[4:],
            "wrong estimator": lambda rows: rows[:8] + [dict(rows[8], estimator="sklearn.ensemble.ExtraTreesClassifier")] + rows[9:],
            "duplicate json key": lambda rows: [dict(rows[0], hyperparameters_json='{"C": 0.01, "C": 1.0}')] + rows[1:],
            "reordered": lambda rows: [rows[1], rows[0]] + rows[2:],
            "renamed id": lambda rows: [dict(rows[0], config_id="LR-09")] + rows[1:],
        }
        for name, mut in cases.items():
            with self.assertRaises(B.BindingError, msg=name):
                B.validate_rows(mut(f3_rows()))
        with tempfile.TemporaryDirectory() as d:                         # a modified COPY (F3 itself untouched)
            p = os.path.join(d, "f3.csv"); shutil.copyfile(B.F3_PATH, p)                 # copy content, not the read-only mode
            with open(p, "a") as fh:
                fh.write("13,RF-05,RF,sklearn.ensemble.RandomForestClassifier,\"{}\",P-TREE\n")
            with self.assertRaisesRegex(B.BindingError, "SHA-256"):
                B.load_bound_grid(p)

    def test_expected_plan_matches_approved_cv_phases(self):
        g = B.load_bound_grid()
        flat = B.expected_plan(g, "flat")
        self.assertEqual(flat, {(c, d) for c in B.APPROVED_IDS for d in S.DAYS})
        tb = B.expected_plan(g, "tb", "HGB-05")
        self.assertEqual(tb, {("HGB-05", f"{o}/{i}") for o in S.DAYS for i in S.DAYS if i != o})
        nested = B.expected_plan(g, "nested")
        self.assertEqual(len(nested), 240); self.assertEqual({c for c, _ in nested}, set(B.APPROVED_IDS))
        for bad in (("tb", None), ("tb", "XGB-01"), ("flat", "LR-01"), ("final", None)):
            with self.assertRaises(B.BindingError):
                B.expected_plan(g, *bad)

    def test_runner_takes_no_config_ids_and_derives_plan_from_f3(self):
        params = inspect.signature(P2.PilotRunnerG2.__init__).parameters
        self.assertNotIn("config_ids", params)
        with tempfile.TemporaryDirectory() as d:
            fx = X.fixture(os.path.join(d, "src"))
            rn = X.runner(fx, os.path.join(d, "w"))
            self.assertEqual(rn.expected, B.expected_plan(B.load_bound_grid(), "flat"))
            for j in (X.job("XGB-01"), X.job("LR-01", "Wed", "Tue")):
                with self.assertRaisesRegex(G1.PilotStop, "UNEXPECTED_KEY"):
                    rn.run(j)
            with self.assertRaisesRegex(G1.PilotStop, "BoundJob"):
                rn.run(G1.JobSpec("HGB-01", S.FoldSpec("Wed"), "LR", "SYNTH-CENTROID"))   # G1 job with a swapped branch
            self.assertEqual(rn.records(), [])
            self.assertEqual({f.name for f in dataclasses.fields(P2.BoundJob)}, {"config_id", "fold", "faults"})


class T2_NoSubstitutionEndToEnd(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory(); self.d = self._td.name
        self.fx = X.fixture(os.path.join(self.d, "src"))

    def tearDown(self):
        self._td.cleanup()

    def test_branch_and_family_come_from_f3(self):
        for cid, branch in (("LR-02", "LR"), ("HGB-05", "TREE"), ("RF-03", "TREE")):
            rn = X.runner(self.fx, os.path.join(self.d, f"w_{cid}"))
            rec = rn.run(X.job(cid))
            self.assertEqual(rec["status"], "OK", rec.get("error"))
            self.assertEqual((rec["branch"], rec["detail"]["branch"]), (branch, branch))
            self.assertEqual(rec["detail"]["config_sha256"], B.EXPECTED_CONFIG_SHA256[cid])
            self.assertEqual(rec["detail"]["kappa_max"] is None, branch == "TREE")

    def test_child_rejects_forged_spec(self):
        g = B.load_bound_grid()
        base = dict(config_id="HGB-01", config_sha256=g["HGB-01"].config_sha256, fold=("Wed", None), faults=[],
                    target="primary", source_paths=self.fx.paths(), allowlist_path=self.fx.al_path,
                    allowlist_sha256=self.fx.al["_sha256"], profile=dataclasses.asdict(self.fx.profile),
                    evidence_dir=self.d, child_pid_path=None)
        forged = [dict(base, config_sha256=g["LR-01"].config_sha256),        # HGB id with LR content hash
                  dict(base, branch="LR"),                                    # extra branch field
                  dict(base, config_id="HGB-09")]
        for k, sp in enumerate(forged):
            sp["child_pid_path"] = os.path.join(self.d, f"pid{k}")
            with self.assertRaises(B.BindingError):
                P2._protected_job_g2(sp)

    def test_payload_branch_swap_is_a_failure(self):
        rn = X.runner(self.fx, os.path.join(self.d, "w"))
        rec = rn.run(X.job("RF-01", faults=(("payload", "branch_swap"),)))
        self.assertEqual((rec["status"], rec["failure_class"]), ("FIT_ERROR", "INVALID_SCORE"))
        self.assertIsNone(rec["score"]); self.assertIn("branch", rec["error"])


class T3_GridCompletion(unittest.TestCase):
    def _ok(self, cid, fold, score=0.5, n_pos=10, n_neg=20):
        cfg = B.load_bound_grid().get(cid)
        det = {k: None for k in P2.PAYLOAD_KEYS if k != "score"}
        det.update(n_pos=n_pos, n_neg=n_neg, config_id=cid, config_sha256=getattr(cfg, "config_sha256", "0" * 64),
                   family=getattr(cfg, "family", "XGB"), branch=getattr(cfg, "branch", "TREE"))
        return dict(model_id=cid, fold=fold, status="OK", score=score, detail=det)

    def test_ledger_rules(self):
        with tempfile.TemporaryDirectory() as d:
            fx = X.fixture(os.path.join(d, "src"))
            rn = X.runner(fx, os.path.join(d, "w"))
            full = [self._ok(c, f) for c, f in sorted(rn.expected)]
            def status(recs):
                with open(rn.ledger_path, "w") as fh:
                    fh.write("".join(json.dumps(r) + "\n" for r in recs))
                return rn.grid_status()["decision"]
            self.assertEqual(status(full), "COMPLETE")
            self.assertEqual(status(full[:-1]), "CONTINUE")
            self.assertEqual(status(full + [full[0]]), "INVALID")                                     # duplicate
            self.assertEqual(status(full[:-1] + [dict(full[-1], status="FIT_ERROR", score=None)]), "STOP")
            self.assertEqual(status(full + [self._ok("XGB-01", "Wed")]), "INVALID")                   # unexpected
            self.assertEqual(status(full[:-1] + [dict(full[-1], score=1.5)]), "INVALID")              # AP > 1
            self.assertEqual(status(full[:-1] + [dict(full[-1], score=AP.AP_UNDEFINED)]), "INVALID")  # undefined on positive day
            mon = [r for r in full if r["fold"] == "Mon"][0]
            undefined_ok = [dict(r, score=dict(AP.AP_UNDEFINED), detail=dict(r["detail"], n_pos=0)) if r is mon else r for r in full]
            self.assertEqual(status(undefined_ok), "COMPLETE")
            swapped = full[:-1] + [dict(full[-1], detail=dict(full[-1]["detail"], branch="LR" if full[-1]["detail"]["branch"] == "TREE" else "TREE"))]
            self.assertEqual(status(swapped), "INVALID")

    def test_full_flat_grid_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            fx = X.fixture(os.path.join(d, "src"), mon_benign=True)
            rn = X.runner(fx, os.path.join(d, "w"))
            for cid in B.APPROVED_IDS:
                for day in S.DAYS:
                    rec = rn.run(X.job(cid, day))
                    self.assertEqual(rec["status"], "OK", (cid, day, rec.get("error")))
                    if day == "Mon":
                        self.assertEqual(rec["score"], AP.AP_UNDEFINED); self.assertEqual(rec["detail"]["n_pos"], 0)
                    else:
                        self.assertTrue(0.0 <= rec["score"] <= 1.0)
                    if (cid, day) == ("RF-04", "Thu"):
                        self.assertEqual(rn.grid_status()["decision"], "CONTINUE")
            st = rn.grid_status()
            self.assertEqual(st["decision"], "COMPLETE", st); self.assertEqual(st["unique_seen"], 60)
            with self.assertRaisesRegex(G1.PilotStop, "DUPLICATE_KEY"):
                rn.run(X.job("LR-01", "Wed"))


class T4_ProposedAdapter(unittest.TestCase):
    def test_all_12_construct_and_validate_without_fit(self):
        pins = A.load_pins(); A.check_environment(pins)
        self.assertEqual(pins["environment"], dict(python="3.13.5", numpy="2.4.6", scikit_learn="1.9.0", scipy="1.18.0",
                                                   joblib="1.5.3", threadpoolctl="3.5.0"))
        cls = {"LR": "LogisticRegression", "HGB": "HistGradientBoostingClassifier", "RF": "RandomForestClassifier"}
        for cid, cfg in B.load_bound_grid().items():
            est = A.build_estimator(cfg, pins)
            self.assertEqual(type(est).__name__, cls[cfg.family])
            self.assertFalse(hasattr(est, "classes_"))                               # never fitted
            self.assertEqual(est.get_params(deep=False), pins["configs"][cid]["full_constructor_params"])
            for k, v in cfg.hyperparameters.items():
                self.assertEqual(est.get_params()[k], v)

    def test_adapter_rejects_mismatches(self):
        g = B.load_bound_grid()
        forged = dataclasses.replace(g["LR-01"], hyperparameters_json='{"C": 5.0, "l1_ratio": 0, "solver": "lbfgs", "max_iter": 2000, "tol": 1e-4, "class_weight": null}')
        with self.assertRaisesRegex(A.AdapterError, "does not match"):
            A.build_estimator(forged)
        with self.assertRaisesRegex(A.AdapterError, "not allowed"):
            A.estimator_class(dataclasses.replace(g["RF-01"], estimator="sklearn.ensemble.ExtraTreesClassifier"))
        with self.assertRaises(A.AdapterError):
            A.build_estimator("RF-01")
        pins = A.load_pins(); pins["configs"]["RF-01"]["full_constructor_params"]["n_jobs"] = 8
        with self.assertRaisesRegex(A.AdapterError, "differ from pins"):
            A.build_estimator(g["RF-01"], pins)
        bad_env = dict(A.load_pins(), environment=dict(A.load_pins()["environment"], scikit_learn="1.8.0"))
        with self.assertRaisesRegex(A.AdapterError, "environment"):
            A.check_environment(bad_env)
        E = A.estimator_class(g["LR-01"])
        with self.assertRaises(Exception):                                            # constructor validation is effective
            E(C=-1.0)._validate_params()

    def test_training_not_authorised(self):
        for mod in (A, P2, G1):
            self.assertFalse(mod.PILOT_TRAINING_AUTHORIZED)
        with self.assertRaisesRegex(A.AdapterError, "NOT_AUTHORIZED"):
            A.train_and_score(B.load_bound_grid()["HGB-01"], None, None, None)
        self.assertEqual(R.sha256_path(A.PINS_PATH), A.PINS_SHA256)


class T5_FrozenArtefactsPreserved(unittest.TestCase):
    def test_frozen_files(self):
        brief = os.path.dirname(HERE)
        for log in ("FREEZE_LOG.md", "FREEZE_LOG_PART_2.md"):
            with open(os.path.join(brief, log), encoding="utf-8") as fh:
                rows = __import__("re").findall(r"^\| (F[0-9][^|]*) \| [^|]* \| `([^`]+)` \| `([0-9a-f]{64})` \|", fh.read(), __import__("re").M)
            self.assertGreater(len(rows), 15)
            for fid, rel, h in rows:
                self.assertEqual(R.sha256_path(os.path.join(brief, rel)), h, (log, fid, rel))
        self.assertEqual(R.sha256_path(G1.__file__), P2.G1_SHA256)


if __name__ == "__main__":
    import platform
    print(f"python {platform.python_version()} machine {platform.machine()} runner {P2.VERSION}")
    unittest.main(verbosity=2)
