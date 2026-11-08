#!/usr/bin/env python3
"""Source-provenance enforcement tests for the only approved real-data route:
   P1 file hashes -> verified DayBlocks -> approved FoldSpec -> verified TrainingMatrix -> fit_p1_final
Tested both end to end (through the guarded G1 runner) and at the API, on synthetic files only.
No real P1 file is parsed or hashed. Run: python3 -I -B test_g1_source_enforcement.py
"""
import dataclasses
import os
import sys
import tempfile
import textwrap
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g1_fixtures as F  # noqa: E402
import pilot_guarded_runner as PR  # noqa: E402

S, R, C1 = PR.S, PR.R, PR.R.C1
CON = F.CON


class Base(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory(); self.d = self._td.name
        self.src = os.path.join(self.d, "src")
        self.fx = F.Fixture(self.src)
        self.ev = os.path.join(self.d, "ev"); os.mkdir(self.ev)
        self.n = 0

    def tearDown(self):
        self._td.cleanup()

    def run_e2e(self, fx, sources=None, outer="Wed", inner=None, phase="flat", selected=None):
        self.n += 1
        work = os.path.join(self.d, f"work{self.n}"); os.mkdir(work)
        rn = F.runner(fx, work, phase=phase, selected=selected, sources=sources)
        return rn.run(F.job(outer=outer, inner=inner))

    def assertProvenanceStop(self, rec, pattern):
        self.assertEqual(rec["status"], "FIT_ERROR"); self.assertEqual(rec["failure_class"], "PROVENANCE_OR_INPUT")
        self.assertIsNone(rec["score"]); self.assertRegex(rec["error"], pattern)

    def allowlisted(self, name, day, X=None, att=None, **kw):
        """A manipulated file that IS on a (test) allow-list: the content checks must still reject it."""
        p = os.path.join(self.src, name)
        F.write_day_csv(p, self.fx.X[day] if X is None else X, self.fx.att[day] if att is None else att, **kw)
        fx = F.Fixture(os.path.join(self.d, f"fx_{name}"), extra_allowed=[(p, day)])
        files = dict(fx.files); files[day] = p
        return fx, [files[d] for d in S.DAYS]


class T1_SourceIdentity(Base):
    def test_unknown_source_rejected(self):
        p3 = os.path.join(self.src, "p3_like.csv"); F.write_day_csv(p3, F.synth(20, 7), np.full(20, -1))
        self.assertProvenanceStop(self.run_e2e(self.fx, self.fx.paths() + [p3]), "unrecognised source hash")
        with self.assertRaisesRegex(R.PreprocessingError, "unrecognised"):
            S.load_day_block(p3, self.fx.al, CON, "primary", self.fx.profile)

    def test_p2_used_for_fitting_rejected(self):
        self.assertProvenanceStop(self.run_e2e(self.fx, self.fx.paths() + [self.fx.p2]), "forbidden source")
        with self.assertRaisesRegex(R.PreprocessingError, "forbidden"):
            S.load_day_block(self.fx.p2, self.fx.al, CON, "primary", self.fx.profile)

    def test_p3_substituted_for_a_training_day_rejected(self):
        p3 = os.path.join(self.src, "thu_p3.csv"); F.write_day_csv(p3, F.synth(40, 8), np.full(40, -1))
        srcs = [self.fx.files[d] for d in ("Mon", "Tue", "Wed", "Fri")] + [p3]
        self.assertProvenanceStop(self.run_e2e(self.fx, srcs), "unrecognised")


class T2_HeldOutDays(Base):
    def test_invalid_fold_specs_rejected(self):
        for args in (("Sun",), ("Wed", "Wed"), ("Wed", "Sat")):
            with self.assertRaises(R.PreprocessingError):
                S.FoldSpec(*args)

    def test_held_out_day_in_training_or_wrong_fold_rejected(self):
        fold = S.FoldSpec("Wed")
        blocks = {d: S.load_day_block(self.fx.files[d], self.fx.al, CON, "primary", self.fx.profile) for d in S.DAYS}
        with self.assertRaisesRegex(R.PreprocessingError, "training days"):
            S.assemble_training_matrix(list(blocks.values()), fold, self.fx.al, CON, "primary", self.fx.profile)
        tm_tue = S.assemble_training_matrix([blocks[d] for d in ("Mon", "Wed", "Thu", "Fri")], S.FoldSpec("Tue"), self.fx.al, CON, "primary", self.fx.profile)
        with self.assertRaisesRegex(R.PreprocessingError, "assembled for fold"):
            S.fit_p1_final(tm_tue, CON, "LR", self.fx.al, fold, self.ev, self.fx.profile)
        forged = dataclasses.replace(tm_tue, fold=fold)                       # relabel the fold only
        with self.assertRaises(R.PreprocessingError):
            S.fit_p1_final(forged, CON, "LR", self.fx.al, fold, self.ev, self.fx.profile)

    def test_validation_day_missing_from_sources_rejected(self):
        srcs = [self.fx.files[d] for d in ("Mon", "Tue", "Thu", "Fri")]        # Wed (held out) absent
        self.assertProvenanceStop(self.run_e2e(self.fx, srcs), "lacks days")

    def test_inner_fold_excludes_both_held_out_days(self):
        rec = self.run_e2e(self.fx, outer="Wed", inner="Fri", phase="tb", selected="SYN-A")
        self.assertEqual(rec["status"], "OK", rec.get("error"))
        self.assertEqual(rec["detail"]["source_verification"]["days"], ["Mon", "Tue", "Thu"])


class T3_RowOrderFeatureOrderRowCount(Base):
    def test_changed_row_order_rejected(self):
        n = 40; perm = np.random.default_rng(1).permutation(n)
        X = self.fx.X["Thu"][perm]; att = self.fx.att["Thu"][perm]
        fx, srcs = self.allowlisted("thu_perm.csv", "Thu", X=X, att=att, ids=[int(i) + 1 for i in perm])
        self.assertProvenanceStop(self.run_e2e(fx, srcs), "file order not preserved")
        p = os.path.join(self.src, "thu_perm_renum.csv"); F.write_day_csv(p, X, att)   # renumbered: new bytes
        srcs = [self.fx.files[d] for d in ("Mon", "Tue", "Wed", "Fri")] + [p]
        self.assertProvenanceStop(self.run_e2e(self.fx, srcs), "unrecognised")
        fold = S.FoldSpec("Wed")
        tm = S.assemble_training_matrix([S.load_day_block(self.fx.files[d], self.fx.al, CON, "primary", self.fx.profile) for d in fold.training_days()],
                                        fold, self.fx.al, CON, "primary", self.fx.profile)
        Xp = tm.X[::-1].copy(); Xp.setflags(write=False)
        for forged in (dataclasses.replace(tm, X=Xp), dataclasses.replace(tm, X=Xp, x_sha256=S._sha_array(Xp)),
                       dataclasses.replace(tm, days=("Fri", "Thu", "Tue", "Mon"))):
            with self.assertRaises(R.PreprocessingError):
                S.fit_p1_final(forged, CON, "LR", self.fx.al, fold, self.ev, self.fx.profile)

    def test_invalid_feature_order_rejected(self):
        h = list(F.HEADER); i, j = h.index("Flow IAT Max"), h.index("Flow IAT Min"); h[i], h[j] = h[j], h[i]
        fx, srcs = self.allowlisted("mon_swapped.csv", "Mon", header=h)
        self.assertProvenanceStop(self.run_e2e(fx, srcs), "column order differs")
        bad_con = dict(CON, schema=list(reversed(CON["schema"])))
        with self.assertRaisesRegex(R.PreprocessingError, "order"):
            S.load_day_block(self.fx.files["Mon"], self.fx.al, bad_con, "primary", self.fx.profile)

    def test_missing_rows_rejected(self):
        X = self.fx.X["Tue"][:-1]; att = self.fx.att["Tue"][:-1]
        fx, srcs = self.allowlisted("tue_short.csv", "Tue", X=X, att=att)
        self.assertProvenanceStop(self.run_e2e(fx, srcs), "pinned Stage 02A-0 count|pinned count")

    def test_unexpected_extra_rows_rejected(self):
        X = np.vstack([self.fx.X["Fri"], self.fx.X["Fri"][:1]]); att = np.append(self.fx.att["Fri"], -1)
        fx, srcs = self.allowlisted("fri_long.csv", "Fri", X=X, att=att)
        self.assertProvenanceStop(self.run_e2e(fx, srcs), "more training rows|pinned")

    def test_extra_rows_on_validation_day_rejected(self):
        X = np.vstack([self.fx.X["Wed"], self.fx.X["Wed"][:2]]); att = np.append(self.fx.att["Wed"], [-1, -1])
        fx, srcs = self.allowlisted("wed_long.csv", "Wed", X=X, att=att)
        self.assertProvenanceStop(self.run_e2e(fx, srcs), "more training rows|pinned")


class T4_NoBypassOfFitP1Final(Base):
    def test_artifact_from_fit_core_rejected_by_model_gate(self):
        fold = S.FoldSpec("Wed")
        tm = S.assemble_training_matrix([S.load_day_block(self.fx.files[d], self.fx.al, CON, "primary", self.fx.profile) for d in fold.training_days()],
                                        fold, self.fx.al, CON, "primary", self.fx.profile)
        direct = R.fit_core(np.asarray(tm.X), CON["schema"], "LR", "P1:Mon,Tue,Thu,Fri", fold.fold_id)
        with self.assertRaisesRegex(R.PreprocessingError, "bypass rejected"):
            PR.require_verified_artifact(direct, tm, fold, self.fx.al, CON, self.fx.profile)
        good = S.fit_p1_final(tm, CON, "LR", self.fx.al, fold, self.ev, self.fx.profile)
        self.assertTrue(PR.require_verified_artifact(good, tm, fold, self.fx.al, CON, self.fx.profile))
        Xo = np.array(tm.X); Xo[0, F.POS["Flow Duration"]] += 12345.0               # unverified matrix, same shape
        other = R.fit_core(Xo, CON["schema"], "LR", "P1:Mon,Tue,Thu,Fri", fold.fold_id)
        fake = dict(other, source_verification=good["source_verification"], provenance=good["provenance"])
        with self.assertRaisesRegex(R.PreprocessingError, "not computed from the verified rows"):
            PR.require_verified_artifact(fake, tm, fold, self.fx.al, CON, self.fx.profile)
        Xc = np.array(tm.X); Xc[3, F.POS["Flow Bytes/s"]] = 1e12                    # changes the R3 cap
        fake2 = dict(R.fit_core(Xc, CON["schema"], "TREE", "P1:Mon,Tue,Thu,Fri", fold.fold_id),
                     source_verification=good["source_verification"], provenance=good["provenance"])
        with self.assertRaisesRegex(R.PreprocessingError, "R3 cap"):
            PR.require_verified_artifact(fake2, tm, fold, self.fx.al, CON, self.fx.profile)
        other_fold = S.FoldSpec("Tue")
        tm2 = S.assemble_training_matrix([S.load_day_block(self.fx.files[d], self.fx.al, CON, "primary", self.fx.profile) for d in other_fold.training_days()],
                                         other_fold, self.fx.al, CON, "primary", self.fx.profile)
        with self.assertRaises(R.PreprocessingError):
            PR.require_verified_artifact(good, tm2, other_fold, self.fx.al, CON, self.fx.profile)

    def test_raw_matrix_and_manual_provenance_rejected(self):
        fold = S.FoldSpec("Wed")
        prov = C1.provenance_from_files(self.fx.paths(fold.training_days()), self.fx.al)
        for bad in (np.zeros((230, 72)), prov, "P1:Mon,Tue,Thu,Fri"):
            with self.assertRaisesRegex(R.PreprocessingError, "TrainingMatrix"):
                S.fit_p1_final(bad, CON, "LR", self.fx.al, fold, self.ev, self.fx.profile)

    def test_static_audit_runner_has_no_bypass(self):
        a = PR.audit_no_bypass(os.path.join(HERE, "pilot_guarded_runner.py"))
        self.assertTrue(a["ok"], a); self.assertTrue(a["uses_fit_p1_final"])
        m = PR.audit_no_bypass(os.path.join(HERE, "synthetic_models.py"))
        self.assertEqual(m["violations"], [])
        p = os.path.join(self.d, "rogue_runner.py")
        with open(p, "w") as fh:
            fh.write(textwrap.dedent("""
                import r5_final as R
                import r5_candidates
                from prep_exact import fit_lean
                def go(X, schema):
                    return R.fit_core(X, schema, "LR", "P1:Mon", "x")
            """))
        bad = PR.audit_no_bypass(p)
        self.assertFalse(bad["ok"]); self.assertGreaterEqual(len(bad["violations"]), 3)

    def test_real_p1_not_parsed_at_this_stage(self):
        with self.assertRaisesRegex(PR.PilotStop, "NOT_AUTHORIZED"):
            PR.PilotRunner(os.path.join(self.d, "w"), R.P1_ALLOWLIST_PATH, ["/nonexistent/x.csv"], S.P1_PROFILE,
                           phase="flat", config_ids=["HGB-01"])


if __name__ == "__main__":
    import platform
    print(f"python {platform.python_version()} machine {platform.machine()} runner {PR.VERSION}")
    unittest.main(verbosity=2)
