#!/usr/bin/env python3
"""Regression tests for R5-FINAL-v1 (Lean NumPy R5 + ZV-1), the kappa diagnostic and source verification.
Synthetic data only: no dataset file is opened; the real allow-list is read for its structure and hash only.
Run: python3 -I -B test_r5_final.py
"""
import csv
import dataclasses
import hashlib
import inspect
import json
import math
import os
import re
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r5_final as R  # noqa: E402
import r5_p1_source as S  # noqa: E402
import r5_hp_reference as HP  # noqa: E402

REF, C1, C2 = R.REF, R.C1, R.C2
BRIEF = R.BRIEF
sys.path.insert(0, os.path.join(BRIEF, "correction_package_C4"))
import r5_candidates as C4  # noqa: E402  (approved Candidate A, used as the equivalence reference)

CON = R.load_contract_verified()
SCHEMA = CON["schema"]
POS = {c: j for j, c in enumerate(SCHEMA)}
NUMERIC = [c for c in SCHEMA if c not in REF.CATEGORICAL]
HEADER = S.load_expected_header()


def _read(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def _load(p):
    return json.loads(_read(p))


def _dump(obj, p):
    with open(p, "w") as fh:
        json.dump(obj, fh)


def synth(n, seed, const=None):
    r = np.random.default_rng(seed)
    X = np.abs(r.lognormal(3, 3, size=(n, 72)))
    for c in CON["integer"]:
        X[:, POS[c]] = np.floor(X[:, POS[c]])
    X[:, POS["Protocol"]] = r.choice([0, 1, 6, 17], n)
    X[:, POS["ICMP Code"]] = r.choice([-1, 0, 3], n)
    X[:, POS["ICMP Type"]] = r.choice([-1, 0, 8], n)
    for c, v in (const or {}).items():
        X[:, POS[c]] = v
    return X


def fit(X, branch="LR", fold="test", **kw):
    return R.fit_core(X, SCHEMA, branch, "P1:x", fold, **kw)


def col_out(art, X, c):
    o, cols = R.transform(X, art, CON)
    return o[:, cols.index(c)]


CONST = {"Flow IAT Mean": 42.0, "Active Max": 1e7, "Bwd IAT Min": 3.0, "Fwd PSH Flags": 0.0}


class T1_EquivalenceWithApprovedCandidateA(unittest.TestCase):
    def test_statistics_zv1_and_outputs_bitwise_equal_to_C4_A(self):
        for n, seed, const in ((1, 1, CONST), (2900, 2, CONST), (20000, 3, None), (54321, 4, CONST)):
            X = synth(n, seed, const)
            a = fit(X); b = C4.fit_lean(X, SCHEMA, "LR", "P1:x", "A_numpy")
            self.assertEqual(a["r5"], b["r5"], n)
            self.assertEqual(a["zv1_constant_columns"], b["zv1_constant_columns"], n)
            for k in ("r3", "r4", "numeric", "output_columns", "schema", "schema_sha256", "n_train"):
                self.assertEqual(a[k], b[k])
            self.assertTrue(np.array_equal(R.transform(X, a, CON)[0], C4.transform(X, b)[0]))

    def test_tree_branch_equal_to_C4_and_v10(self):
        X = synth(30000, 5)
        t = fit(X, "TREE"); t4 = C4.fit_lean(X, SCHEMA, "TREE", "P1:x", "A_numpy"); t10 = REF.fit(X, SCHEMA, "TREE", "P1:x")
        self.assertIsNone(t["r5"]); self.assertIsNone(t["r5_kappa"])
        self.assertTrue(np.array_equal(R.transform(X, t, CON)[0], C4.transform(X, t4)[0]))
        self.assertTrue(np.array_equal(R.transform(X, t, CON)[0], REF.transform(X, t10)[0]))

    def test_signed_log1p_after_r3_and_numpy_ddof0_whole_column(self):
        X = synth(20000, 6); X[3, POS["Flow Bytes/s"]] = np.inf; X[9, POS["Flow Packets/s"]] = np.inf
        a = fit(X); cap = REF._apply_r3(X, a)
        for i, c in enumerate(NUMERIC):
            z = np.ascontiguousarray(np.sign(cap[:, POS[c]]) * np.log1p(np.abs(cap[:, POS[c]])))
            self.assertEqual(a["r5"]["mean"][i], float(np.mean(z)).hex(), c)
            self.assertEqual(a["r5"]["std"][i], float(np.std(z, ddof=0)).hex(), c)

    def test_candidate_B_not_used(self):
        for mod in (R, S):
            src = inspect.getsource(mod)
            for bad in ("import er5", "er5_mu_sd", "fsum", "stats_B", "r5_candidates"):
                self.assertNotIn(bad, src.split('"""', 2)[-1], (mod.__name__, bad))


class T2_ZV1(unittest.TestCase):
    def test_constant_columns_mean_is_constant_scale_one_output_zero(self):
        for n in (1, 2, 100, 2900, 54321):
            X = synth(n, n, CONST); a = fit(X)
            self.assertTrue(set(CONST) <= set(a["zv1_constant_columns"]))
            for c, v in CONST.items():
                i = NUMERIC.index(c)
                self.assertEqual(REF._unhx(a["r5"]["std"][i]), 1.0)
                self.assertEqual(REF._unhx(a["r5"]["mean"][i]), float(REF.signed_log1p(np.array([v]))[0]))
                self.assertTrue(np.all(col_out(a, X, c) == 0.0))

    def test_constant_detection_uses_min_eq_max_only(self):
        src = inspect.getsource(R.r5_column_statistics)
        self.assertIn("z.min() == z.max()", src)
        body = src.split('"""')[-1]
        for bad in ("eps", "tol", "1e-", "isclose", "allclose", "var("):
            self.assertNotIn(bad, body)

    def test_near_constant_not_constant(self):
        z0 = REF.signed_log1p(np.array([1e3]))[0]
        z = np.full(1000, z0); z[7] = np.nextafter(z0, np.inf)
        mu, sd, const, k = R.r5_column_statistics(z)
        self.assertFalse(const); self.assertNotEqual(sd, 1.0); self.assertIsNotNone(k)

    def test_constant_columns_excluded_from_kappa(self):
        a = fit(synth(3000, 7, CONST))
        for i, c in enumerate(NUMERIC):
            v = a["r5_kappa"]["values_hex"][i]
            self.assertEqual(v is None, c in a["zv1_constant_columns"], c)
        rec = R.kappa_record(a)
        self.assertEqual(rec["n_nonconstant"], 69 - len(a["zv1_constant_columns"]))

    def test_capped_constant_detected(self):
        X = synth(300, 8); X[:, POS["Flow Bytes/s"]] = 5.0; X[0, POS["Flow Bytes/s"]] = np.inf
        a = fit(X)
        self.assertIn("Flow Bytes/s", a["zv1_constant_columns"])
        self.assertTrue(np.all(col_out(a, X, "Flow Bytes/s") == 0.0))


class T3_KappaDiagnostic(unittest.TestCase):
    def test_threshold_is_provisional_1e11_and_strict(self):
        self.assertEqual(R.KAPPA_INVESTIGATION_THRESHOLD, 1e11)
        self.assertFalse(R.kappa_exceeds(1e11))
        self.assertTrue(R.kappa_exceeds(np.nextafter(1e11, np.inf)))
        self.assertTrue(R.kappa_exceeds(math.inf)); self.assertTrue(R.kappa_exceeds(math.nan))

    def test_kappa_recorded_for_every_nonconstant_column_and_fold(self):
        X = synth(20000, 9); a = fit(X, fold="outer=Tue")
        cap = REF._apply_r3(X, a)
        for i, c in enumerate(NUMERIC):
            z = REF.signed_log1p(cap[:, POS[c]])
            self.assertEqual(a["r5_kappa"]["values_hex"][i], (abs(float(np.mean(z))) / float(np.std(z))).hex())
        rec = R.kappa_record(a)
        self.assertEqual(rec["fold_id"], "outer=Tue"); self.assertEqual(len(rec["columns"]), 69)
        self.assertFalse(rec["any_exceeds"]); self.assertLess(rec["max_kappa"], 1e11)

    def _illcond(self, n=1000):
        X = synth(n, 10); X[:, POS["Flow IAT Mean"]] = 1e9; X[0, POS["Flow IAT Mean"]] = 1e9 + 1  # kappa ~ 6.5e11
        return X

    def test_stop_preserves_evidence_and_returns_no_artifact(self):
        X = self._illcond()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(R.NumericalReviewRequired) as cm:
                fit(X, fold="outer=Wed", evidence_dir=d)
            rec = cm.exception.record
            self.assertEqual(rec["affected_columns"], ["Flow IAT Mean"])
            self.assertEqual(len(rec["columns"]), 69)                    # nothing dropped
            self.assertIn("separate decision", rec["instruction"])
            files = sorted(os.listdir(d)); self.assertEqual(len(files), 2)
            j = _load(os.path.join(d, [f for f in files if f.endswith(".json")][0]))
            ev = j["transformed_training_columns"]["Flow IAT Mean"]
            p = os.path.join(d, ev["path"])
            self.assertEqual(R.sha256_path(p), ev["sha256"])
            z = np.load(p)
            self.assertTrue(np.array_equal(z, REF.signed_log1p(X[:, POS["Flow IAT Mean"]])))
            k = float.fromhex([t for t in j["columns"] if t["column"] == "Flow IAT Mean"][0]["kappa_hex"])
            self.assertGreater(k, 1e11)
            a = HP.assess_column(z, float(np.mean(z)), float(np.std(z)))     # independent reference can assess it
            self.assertGreater(float(a["kappa_exact"]), 1e11)
            with self.assertRaises(FileExistsError):                        # evidence is never overwritten
                fit(X, fold="outer=Wed", evidence_dir=d)

    def test_no_override_or_fallback_parameters(self):
        for fn in (R.fit_core, S.fit_p1_final):
            params = set(inspect.signature(fn).parameters)
            for bad in ("threshold", "kappa", "resume", "override", "force", "drop", "fallback", "estimator"):
                self.assertFalse(any(bad in p for p in params), (fn.__name__, params))

    def test_underflow_sd_zero_nonconstant_stops(self):
        X = synth(1000, 11); X[:, POS["Active Mean"]] = 1e-300; X[::2, POS["Active Mean"]] = 2e-300
        with self.assertRaises(R.NumericalReviewRequired) as cm:
            fit(X)
        t = [t for t in cm.exception.record["columns"] if t["column"] == "Active Mean"][0]
        self.assertEqual(t["kappa"], "inf")

    def test_moderate_conditioning_does_not_stop(self):
        X = synth(5000, 12); X[:, POS["Flow Duration"]] = np.floor(np.random.default_rng(0).uniform(1.19e8, 1.2e8, 5000))
        a = fit(X); self.assertFalse(R.kappa_record(a)["any_exceeds"])


class T4_EnvironmentAndPins(unittest.TestCase):
    def test_numpy_pin_enforced(self):
        self.assertEqual(R.REQUIRED_NUMPY, "2.4.6"); self.assertEqual(np.__version__, "2.4.6")
        orig = R._numpy_version
        try:
            R._numpy_version = lambda: "2.4.5"
            with self.assertRaisesRegex(R.PreprocessingError, "requires NumPy 2.4.6"):
                fit(synth(10, 1))
        finally:
            R._numpy_version = orig

    def test_dependency_and_input_pins(self):
        self.assertTrue(R.verify_pins())
        for rel, h in {**R.DEPENDENCIES, **R.PINNED_INPUTS}.items():
            self.assertEqual(R.sha256_path(os.path.join(BRIEF, rel)), h, rel)
        self.assertEqual(REF.schema_sha256(SCHEMA), R.SCHEMA_ORDERED_SHA256)

    def test_artifact_reproducible_for_fixed_order_and_layout(self):
        X = synth(40000, 13)
        a1 = fit(X); a2 = fit(X.copy()); a3 = fit(np.asfortranarray(X))
        self.assertEqual(REF.artifact_sha256(a1), REF.artifact_sha256(a2))
        self.assertEqual(a1["r5"], a3["r5"])


# ----------------------------------------------------------------------------- synthetic source files
def _fmt(v):
    return "inf" if v == np.inf else (str(int(v)) if float(v).is_integer() and abs(v) < 1e15 else repr(float(v)))


def write_day_csv(path, X, attempted, ids=None, header=None):
    header = list(header or HEADER)
    n = X.shape[0]
    ids = list(range(1, n + 1)) if ids is None else ids
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in range(n):
            row = []
            for h in header:
                if h in POS:
                    row.append(_fmt(X[r, POS[h]]))
                elif h == "id":
                    row.append(str(ids[r]))
                elif h == "Label":
                    row.append("BENIGN" if r % 3 else "DoS")
                elif h == "Attempted Category":
                    row.append(f"{float(attempted[r]):.1f}")
                elif h in ("Flow ID", "Src IP", "Dst IP", "Timestamp"):
                    row.append(f"s{r}")
                else:
                    row.append("0")
            w.writerow(row)


class SourceFixture:
    """Five synthetic day files, a synthetic allow-list (+ a forbidden P2 file) and a matching test profile."""

    def __init__(self, d, sizes=(60, 50, 70, 40, 80), attempted_every=7):
        self.d, self.files, self.X, self.att = d, {}, {}, {}
        al = {"P1": {}, "forbidden": {}}
        for k, (day, n) in enumerate(zip(S.DAYS, sizes)):
            X = synth(n, 100 + k); X[0, POS["Flow Bytes/s"]] = np.inf if day == "Tue" else X[0, POS["Flow Bytes/s"]]
            att = np.where(np.arange(n) % attempted_every == 3, 0, -1) if day != "Mon" else np.full(n, -1)
            p = os.path.join(d, f"{day}.csv"); write_day_csv(p, X, att)
            self.files[day], self.X[day], self.att[day] = p, X, att
            al["P1"][C1.sha256_file(p)] = dict(day=day, file=f"{day}.csv", bytes=os.path.getsize(p))
        q = os.path.join(d, "p2.csv"); write_day_csv(q, synth(10, 99), np.full(10, -1)); self.p2 = q
        al["forbidden"][C1.sha256_file(q)] = dict(partition="P2", file="p2.csv", bytes=os.path.getsize(q))
        self.al_path = os.path.join(d, "allow.json"); _dump(al, self.al_path)
        self.al = C1.load_allowlist(self.al_path)
        self.profile = S.SourceProfile("TEST", self.al["_sha256"], tuple(zip(S.DAYS, sizes)),
                                       tuple((day, int(np.sum(self.att[day] != -1))) for day in S.DAYS))

    def block(self, day, target="primary"):
        return S.load_day_block(self.files[day], self.al, CON, target, self.profile)

    def matrix(self, fold, target="primary", order=None):
        days = order or fold.training_days()
        return S.assemble_training_matrix([self.block(d, target) for d in days], fold, self.al, CON, target, self.profile)


class T5_SourceProvenance(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory(); self.d = self._td.name
        self.fx = SourceFixture(self.d)
        self.ev = os.path.join(self.d, "evidence"); os.mkdir(self.ev)

    def tearDown(self):
        self._td.cleanup()

    def test_outer_fold_rows_order_and_fit(self):
        fold = S.FoldSpec("Wed")
        tm = self.fx.matrix(fold, order=("Fri", "Mon", "Thu", "Tue"))       # input order does not matter
        self.assertEqual(tm.days, ("Mon", "Tue", "Thu", "Fri"))
        exp = np.concatenate([self.fx.X[d] for d in ("Mon", "Tue", "Thu", "Fri")])
        self.assertTrue(np.array_equal(tm.X, exp))                            # Mon -> Fri, file order
        art = S.fit_p1_final(tm, CON, "LR", self.fx.al, fold, self.ev, self.fx.profile)
        self.assertEqual(tuple(art["provenance"]["days"]), ("Mon", "Tue", "Thu", "Fri"))
        self.assertEqual(art["source_verification"]["n_train"], 60 + 50 + 40 + 80)
        self.assertEqual(art["r5"], fit(exp)["r5"])                           # same numbers as the core fit
        self.assertTrue(os.path.exists(os.path.join(self.ev, "R5_KAPPA_outer=Wed_primary.json")))
        with self.assertRaises(FileExistsError):                              # kappa log never overwritten
            S.fit_p1_final(tm, CON, "LR", self.fx.al, fold, self.ev, self.fx.profile)

    def test_inner_fold(self):
        fold = S.FoldSpec("Wed", "Fri")
        tm = self.fx.matrix(fold)
        self.assertEqual(tm.days, ("Mon", "Tue", "Thu"))
        S.fit_p1_final(tm, CON, "TREE", self.fx.al, fold, self.ev, self.fx.profile)
        with self.assertRaises(R.PreprocessingError):
            S.FoldSpec("Wed", "Wed")

    def test_fold_membership_mismatch_rejected(self):
        fold = S.FoldSpec("Wed")
        b = {d: self.fx.block(d) for d in S.DAYS}
        bad_sets = ([b["Mon"], b["Tue"], b["Thu"]],                           # missing day
                    [b[d] for d in S.DAYS],                                    # held-out day included
                    [b["Mon"], b["Tue"], b["Thu"], b["Fri"], b["Fri"]])        # duplicate
        for blocks in bad_sets:
            with self.assertRaises(R.PreprocessingError):
                S.assemble_training_matrix(blocks, fold, self.fx.al, CON, "primary", self.fx.profile)
        tm = self.fx.matrix(S.FoldSpec("Tue"))
        with self.assertRaisesRegex(R.PreprocessingError, "assembled for fold"):
            S.fit_p1_final(tm, CON, "LR", self.fx.al, fold, self.ev, self.fx.profile)

    def test_forbidden_and_unknown_sources_rejected(self):
        with self.assertRaisesRegex(R.PreprocessingError, "forbidden"):
            S.load_day_block(self.fx.p2, self.fx.al, CON, "primary", self.fx.profile)
        p3 = os.path.join(self.d, "p3.csv"); write_day_csv(p3, synth(10, 98), np.full(10, -1))
        with self.assertRaisesRegex(R.PreprocessingError, "unrecognised"):
            S.load_day_block(p3, self.fx.al, CON, "primary", self.fx.profile)

    def test_feature_order_mismatch_rejected(self):
        h = list(HEADER); i, j = h.index("Flow IAT Max"), h.index("Flow IAT Min"); h[i], h[j] = h[j], h[i]
        p = os.path.join(self.d, "swapped.csv"); write_day_csv(p, synth(60, 100), np.full(60, -1), header=h)
        al = _load(self.fx.al_path); al["P1"] = {C1.sha256_file(p): dict(day="Mon", file="s", bytes=os.path.getsize(p))}
        ap = os.path.join(self.d, "al2.json"); _dump(al, ap); al2 = C1.load_allowlist(ap)
        prof = dataclasses.replace(self.fx.profile, allowlist_sha256=al2["_sha256"])
        with self.assertRaisesRegex(R.PreprocessingError, "order"):
            S.load_day_block(p, al2, CON, "primary", prof)
        tm = self.fx.matrix(S.FoldSpec("Wed"))
        with self.assertRaisesRegex(R.PreprocessingError, "feature order"):
            S.fit_p1_final(dataclasses.replace(tm, feature_order_sha256="0" * 64), CON, "LR", self.fx.al, S.FoldSpec("Wed"), self.ev, self.fx.profile)

    def test_row_count_mismatch_rejected(self):
        prof = dataclasses.replace(self.fx.profile, file_rows=tuple((d, n + (d == "Mon")) for d, n in self.fx.profile.file_rows))
        with self.assertRaisesRegex(R.PreprocessingError, "pinned"):
            S.load_day_block(self.fx.files["Mon"], self.fx.al, CON, "primary", prof)
        prof2 = dataclasses.replace(self.fx.profile, file_rows=tuple((d, n - (d == "Tue")) for d, n in self.fx.profile.file_rows))
        with self.assertRaisesRegex(R.PreprocessingError, "pinned"):
            S.load_day_block(self.fx.files["Tue"], self.fx.al, CON, "primary", prof2)

    def test_file_order_enforced_by_id(self):
        n = 50; ids = list(range(1, n + 1)); ids[3], ids[4] = ids[4], ids[3]
        p = os.path.join(self.d, "perm.csv"); write_day_csv(p, synth(n, 101), np.full(n, -1), ids=ids)
        al = _load(self.fx.al_path); al["P1"] = {C1.sha256_file(p): dict(day="Tue", file="p", bytes=os.path.getsize(p))}
        ap = os.path.join(self.d, "al3.json"); _dump(al, ap); al3 = C1.load_allowlist(ap)
        prof = dataclasses.replace(self.fx.profile, allowlist_sha256=al3["_sha256"])
        with self.assertRaisesRegex(R.PreprocessingError, "file order"):
            S.load_day_block(p, al3, CON, "primary", prof)

    def test_streaming_hash_detects_changed_bytes(self):
        good = C1.sha256_file(self.fx.files["Mon"])
        p = os.path.join(self.d, "changed.csv")
        X = self.fx.X["Mon"].copy(); X[5, POS["Flow Duration"]] += 1
        write_day_csv(p, X, self.fx.att["Mon"])
        orig = C1.sha256_file
        try:
            C1.sha256_file = lambda path, chunk=1 << 22: good          # pre-hash spoofed
            with self.assertRaisesRegex(R.PreprocessingError, "streaming hash mismatch"):
                S.load_day_block(p, self.fx.al, CON, "primary", self.fx.profile)
        finally:
            C1.sha256_file = orig

    def test_manual_provenance_raw_matrix_and_forged_blocks_rejected(self):
        fold = S.FoldSpec("Wed")
        prov = C1.provenance_from_files([self.fx.files[d] for d in fold.training_days()], self.fx.al)
        for bad in (prov, np.zeros((5, 72)), "P1:Mon,Tue,Thu,Fri"):
            with self.assertRaisesRegex(R.PreprocessingError, "TrainingMatrix"):
                S.fit_p1_final(bad, CON, "LR", self.fx.al, fold, self.ev, self.fx.profile)
        b = self.fx.block("Mon")
        X2 = b.X.copy(); X2[0, 1] += 1; X2.setflags(write=False)
        for forged in (dataclasses.replace(b, X=X2),                                         # content changed
                       dataclasses.replace(b, X=X2, x_sha256=S._sha_array(X2))):              # hash recomputed
            with self.assertRaises(R.PreprocessingError):
                S.assemble_training_matrix([forged] + [self.fx.block(d) for d in ("Tue", "Thu", "Fri")], fold, self.fx.al, CON, "primary", self.fx.profile)
        tm = self.fx.matrix(fold)
        Xt = tm.X.copy(); Xt[0, 1] += 1; Xt.setflags(write=False)
        for forged in (dataclasses.replace(tm, X=Xt), dataclasses.replace(tm, X=Xt, x_sha256=S._sha_array(Xt)),
                       dataclasses.replace(tm, n_rows_by_day=(61, 50, 40, 79))):
            with self.assertRaises(R.PreprocessingError):
                S.fit_p1_final(forged, CON, "LR", self.fx.al, fold, self.ev, self.fx.profile)

    def test_profile_rules_protect_real_p1(self):
        real = C1.load_allowlist(R.P1_ALLOWLIST_PATH)
        self.assertEqual(real["_sha256"], S.P1_PROFILE.allowlist_sha256)
        self.assertEqual(set(real["P1"]), set(S.REAL_P1_FILE_SHA256))
        self.assertEqual({v["day"]: v["bytes"] for v in real["P1"].values()}["Tue"], 178397720)
        self.assertTrue(S.check_profile(S.P1_PROFILE, real, tuple(real["P1"])))
        with self.assertRaises(R.PreprocessingError):
            S.check_profile(self.fx.profile, real)                                  # test profile + real allow-list
        with self.assertRaises(R.PreprocessingError):
            S.check_profile(self.fx.profile, self.fx.al, (sorted(S.REAL_P1_FILE_SHA256)[0],))
        with self.assertRaises(R.PreprocessingError):
            S.check_profile(S.P1_PROFILE, self.fx.al)
        self.assertEqual(sum(n for _, n in S.P1_FILE_ROWS) - S.P1_PROFILE.expected_rows("Tue", "primary"), 1777898)
        self.assertEqual(S.P1_PROFILE.expected_rows("Wed", "sensitivity"), 490765)

    def test_sensitivity_target_excludes_attempted(self):
        fold = S.FoldSpec("Mon")
        tm = self.fx.matrix(fold, "sensitivity")
        self.assertTrue(np.all(tm.attempted == -1))
        keep = np.concatenate([self.fx.X[d][self.fx.att[d] == -1] for d in fold.training_days()])
        self.assertTrue(np.array_equal(tm.X, keep))
        S.fit_p1_final(tm, CON, "LR", self.fx.al, fold, self.ev, self.fx.profile)

    def test_training_only_statistics_and_saved_statistics_reused(self):
        fold = S.FoldSpec("Wed")
        tm = self.fx.matrix(fold)
        art = S.fit_p1_final(tm, CON, "LR", self.fx.al, fold, self.ev, self.fx.profile)
        h = REF.artifact_sha256(art)
        ho = self.fx.X["Wed"].copy(); ho[1, POS["Flow Bytes/s"]] = np.inf
        for c in NUMERIC:                    # scale numeric columns only (ICMP -1 * 3 would violate C-ICMP)
            ho[:, POS[c]] *= 3.0
        out, cols = R.transform(ho, art, CON)
        self.assertEqual(REF.artifact_sha256(art), h)                                    # never refits
        fin = tm.X[np.isfinite(tm.X[:, POS["Flow Bytes/s"]]), POS["Flow Bytes/s"]]
        self.assertEqual(REF._unhx(art["r3"]["Flow Bytes/s"][1]), fin.max())             # cap from training days
        i = NUMERIC.index("Flow Duration")
        exp = (REF.signed_log1p(ho[:, POS["Flow Duration"]]) - REF._unhx(art["r5"]["mean"][i])) / REF._unhx(art["r5"]["std"][i])
        self.assertTrue(np.array_equal(out[:, cols.index("Flow Duration")], exp))
        bad = ho.copy(); bad[0, POS["Flow Duration"]] = -1.0
        with self.assertRaisesRegex(R.PreprocessingError, "C-NN"):
            R.transform(bad, art, CON)

    def test_stop_through_real_data_entry_point(self):
        fold = S.FoldSpec("Wed")
        for d in S.DAYS:
            X = self.fx.X[d]; X[:, POS["Flow IAT Mean"]] = 1e9
        self.fx.X["Mon"][0, POS["Flow IAT Mean"]] = 1e9 + 1
        for d in S.DAYS:
            write_day_csv(self.fx.files[d] + ".ill", self.fx.X[d], self.fx.att[d])
        al = {"P1": {C1.sha256_file(self.fx.files[d] + ".ill"): dict(day=d, file=d, bytes=os.path.getsize(self.fx.files[d] + ".ill")) for d in S.DAYS}, "forbidden": {}}
        ap = os.path.join(self.d, "al_ill.json"); _dump(al, ap); al = C1.load_allowlist(ap)
        prof = dataclasses.replace(self.fx.profile, allowlist_sha256=al["_sha256"])
        blocks = [S.load_day_block(self.fx.files[d] + ".ill", al, CON, "primary", prof) for d in fold.training_days()]
        tm = S.assemble_training_matrix(blocks, fold, al, CON, "primary", prof)
        with self.assertRaises(R.NumericalReviewRequired) as cm:
            S.fit_p1_final(tm, CON, "LR", al, fold, self.ev, prof)
        self.assertEqual(cm.exception.record["context"]["source_verification"]["fold_id"], "outer=Wed")
        self.assertFalse(any(f.startswith("R5_KAPPA_") for f in os.listdir(self.ev)))      # no success log
        self.assertTrue(any(f.startswith("R5_STOP_") for f in os.listdir(self.ev)))

    def test_evidence_dir_required(self):
        fold = S.FoldSpec("Wed"); tm = self.fx.matrix(fold)
        with self.assertRaisesRegex(R.PreprocessingError, "evidence"):
            S.fit_p1_final(tm, CON, "LR", self.fx.al, fold, None, self.fx.profile)


class T6_HighPrecisionReference(unittest.TestCase):
    def test_exact_reference_on_known_values(self):
        mean, var, sd = HP.exact_moments(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(float(mean), 2.5); self.assertEqual(sd, math.sqrt(1.25))
        a = HP.assess_column(np.array([0.1, 0.2, 0.3]), float(np.mean([0.1, 0.2, 0.3])), float(np.std([0.1, 0.2, 0.3])))
        self.assertFalse(a["material"])

    def test_candidate_A_within_tolerance_on_synthetic_columns(self):
        X = synth(50000, 14); a = fit(X); cap = REF._apply_r3(X, a)
        for i, c in enumerate(NUMERIC[:20]):
            z = REF.signed_log1p(cap[:, POS[c]])
            r = HP.assess_column(z, REF._unhx(a["r5"]["mean"][i]), REF._unhx(a["r5"]["std"][i]))
            self.assertFalse(r["material"], c)


class T7_FrozenAndEarlierFilesPreserved(unittest.TestCase):
    def test_every_freeze_log_entry_still_matches(self):
        text = _read(os.path.join(BRIEF, "FREEZE_LOG.md"))
        rows = re.findall(r"^\| (F[0-9][^|]*) \| [^|]* \| `([^`]+)` \| `([0-9a-f]{64})` \|", text, re.M)
        self.assertGreaterEqual(len(rows), 20)
        for fid, rel, h in rows:
            self.assertEqual(R.sha256_path(os.path.join(BRIEF, rel)), h, (fid, rel))

    def test_key_files_unchanged(self):
        h = lambda p: R.sha256_path(os.path.join(BRIEF, p))
        self.assertEqual(h("FREEZE_LOG.md"), "032f9dfc0c20759b9f9905833c7745824a13a063659f852a33fa56b56925c456")
        self.assertEqual(h("next_workstream/CONFIG_GRID_F3_v1.csv"), "34b282f13e52a7a5507a450af40513e93da60b27d782245de0f7332c09e7c179")
        self.assertEqual(h("FEATURE_SCHEMA_PRIMARY_72_DRAFT.csv"), "6bfc24b8332111fa03729d3597ea4ef8685e11aab999af2906738fd07eee1e9f")
        self.assertEqual(h("next_workstream/PREPROCESSING_CONFIG_PROPOSED.json"), "55673db5bce6c1dd028a1b8eb3790d11cdf14388bb9942263288eb43640feccb")

    def test_c4_package_unchanged(self):
        d = os.path.join(BRIEF, "correction_package_C4")
        for ln in _read(os.path.join(d, "MANIFEST_SHA256.txt")).splitlines()[1:]:
            hx, rel = ln.split("  ", 1)
            self.assertEqual(R.sha256_path(os.path.join(d, rel)), hx, rel)


if __name__ == "__main__":
    import platform
    print(f"python {platform.python_version()} numpy {np.__version__} machine {platform.machine()} spec {R.SPEC_ID}")
    unittest.main(verbosity=2)
