#!/usr/bin/env python3
"""U1 tests (synthetic only): lean == dense exact reference BITWISE; exact stats order-independent;
deviation from v10 NumPy-stat reference quantified; all v10 behaviours preserved.
Run: python3 -I -B test_prep_exact.py
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prep_exact as E  # noqa: E402

REF = E.REF
BRIEF = os.path.abspath(os.path.join(HERE, "..", ".."))
SCHEMA = REF.load_schema(os.path.join(BRIEF, "FEATURE_SCHEMA_PRIMARY_72_DRAFT.csv"))
POS = {c: j for j, c in enumerate(SCHEMA)}
RNG = np.random.default_rng(21)


def synth(n, seed, heavy=True):
    r = np.random.default_rng(seed)
    X = np.abs(r.lognormal(3, 4 if heavy else 1, size=(n, 72)))
    X[:, POS["Protocol"]] = r.choice([0, 1, 6, 17], n)
    X[:, POS["ICMP Code"]] = r.choice([-1, 0, 3, 8, 10], n)
    X[:, POS["ICMP Type"]] = r.choice([-1, 0, 3, 8, 11], n)
    X[: max(1, n // 50), POS["Fwd PSH Flags"]] = 0.0
    if n > 3:
        X[1, POS["Flow Bytes/s"]] = np.inf
        X[2, POS["Flow Packets/s"]] = np.inf
    return X


class TestBitwiseEquivalence(unittest.TestCase):
    def test_lean_equals_dense_bitwise_many_shapes(self):
        for n, seed in ((5, 1), (37, 2), (1000, 3), (54321, 4), (200003, 5)):
            tr = synth(n, seed); ho = synth(max(3, n // 3), seed + 100)
            ho[0, POS["Flow Bytes/s"]] = np.inf; ho[-1, POS["Protocol"]] = 2  # unseen category
            for br in ("LR", "TREE"):
                a_d = E.fit_dense(tr, SCHEMA, br, "P1:x"); a_l = E.fit_lean(tr, SCHEMA, br, "P1:x")
                self.assertEqual(json.dumps(a_d, sort_keys=True), json.dumps(a_l, sort_keys=True), (n, br))
                for M in (tr, ho):
                    o_d, c_d = E.transform_dense(M, a_d); o_l, c_l = E.transform_lean(M, a_l)
                    self.assertEqual(c_d, c_l)
                    self.assertTrue(np.array_equal(o_d, o_l), (n, br))
                    self.assertEqual(np.ascontiguousarray(o_d).tobytes(), np.ascontiguousarray(o_l).tobytes())

    def test_lean_on_fortran_and_memmap_inputs_and_disk_output(self):
        tr = synth(20000, 7)
        a = E.fit_dense(tr, SCHEMA, "LR", "P1:x")
        ref, _ = E.transform_dense(tr, a)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "X.npy"); np.save(p, np.asfortranarray(tr))
            Xm = np.load(p, mmap_mode="r")
            a_l = E.fit_lean(Xm, SCHEMA, "LR", "P1:x")
            self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(a_l, sort_keys=True))
            o, _ = E.transform_lean(Xm, a_l, out_path=os.path.join(d, "out.npy"))
            self.assertTrue(np.array_equal(np.asarray(o), ref))
            self.assertTrue(np.array_equal(np.load(os.path.join(d, "out.npy")), ref))


class TestExactStatistics(unittest.TestCase):
    def test_order_and_layout_independent(self):
        z = RNG.lognormal(0, 3, 100003)
        m1, s1 = E._exact_mu_sd(z); m2, s2 = E._exact_mu_sd(z[RNG.permutation(z.size)])
        self.assertEqual((m1, s1), (m2, s2))
        a = E.fit_dense(synth(5000, 9), SCHEMA, "LR", "P1:x")
        b = E.fit_dense(synth(5000, 9)[RNG.permutation(5000)], SCHEMA, "LR", "P1:x")
        self.assertEqual(a["r5"], b["r5"])

    def test_exactness_against_fraction_arithmetic(self):
        from fractions import Fraction
        z = RNG.normal(1e6, 1, 2001)
        mu, _ = E._exact_mu_sd(z)
        exact = sum(Fraction(float(v)) for v in z) / len(z)
        self.assertEqual(mu, float(exact))  # fsum/n == correctly rounded sum / n (sum exact, one division)

    def test_zero_variance(self):
        mu, sd = E._exact_mu_sd(np.full(10, 3.0)); self.assertEqual((mu, sd), (3.0, 1.0))

    def test_deviation_from_v10_numpy_stats_is_tiny(self):
        tr = synth(200003, 11)
        a_v10 = REF.fit(tr, SCHEMA, "LR", "P1:x"); a_ex = E.fit_dense(tr, SCHEMA, "LR", "P1:x")
        mu10 = np.array([REF._unhx(v) for v in a_v10["r5"]["mean"]]); muex = np.array([REF._unhx(v) for v in a_ex["r5"]["mean"]])
        sd10 = np.array([REF._unhx(v) for v in a_v10["r5"]["std"]]); sdex = np.array([REF._unhx(v) for v in a_ex["r5"]["std"]])
        rel_mu = np.max(np.abs(mu10 - muex) / np.maximum(np.abs(muex), 1e-300))
        rel_sd = np.max(np.abs(sd10 - sdex) / sdex)
        o10, _ = REF.transform(tr, a_v10); oex, _ = E.transform_dense(tr, a_ex)
        abs_out = float(np.max(np.abs(o10 - oex)))
        print(f"\n  v10-vs-exact: max rel diff mean {rel_mu:.3e}, std {rel_sd:.3e}; max abs output diff {abs_out:.3e}; "
              f"identical outputs {np.array_equal(o10, oex)}")
        self.assertLess(rel_mu, 1e-12); self.assertLess(rel_sd, 1e-12); self.assertLess(abs_out, 1e-9)
        self.assertEqual(a_v10["r3"], a_ex["r3"]); self.assertEqual(a_v10["r4"], a_ex["r4"])  # R3/R4 identical


class TestBehaviourPreserved(unittest.TestCase):
    def test_lean_rejections(self):
        tr = synth(100, 12)
        bad = tr.copy(); bad[0, POS["Flow Duration"]] = np.nan
        with self.assertRaises(REF.PreprocessingError):
            E.fit_lean(bad, SCHEMA, "LR", "P1:x")
        bad = tr.copy(); bad[0, POS["Idle Max"]] = np.inf
        with self.assertRaises(REF.PreprocessingError):
            E.fit_lean(bad, SCHEMA, "TREE", "P1:x")
        bad = tr.copy(); bad[:, POS["Flow Bytes/s"]] = np.inf
        with self.assertRaises(REF.PreprocessingError):
            E.fit_lean(bad, SCHEMA, "TREE", "P1:x")
        for tag in ("P2:x", "", None):
            with self.assertRaises(REF.PreprocessingError):
                E.fit_lean(tr, SCHEMA, "TREE", tag)
        a = E.fit_lean(tr, SCHEMA, "TREE", "P1:x"); ho = synth(5, 13); ho[0, POS["Idle Max"]] = np.inf
        with self.assertRaises(REF.PreprocessingError):
            E.transform_lean(ho, a)

    def test_v10_and_c1_suites_still_pass(self):
        for rel in ("preprocessing_review/test_prep_reference.py", "correction_package_C1/prep/test_prep_v0_2.py"):
            r = subprocess.run([sys.executable, "-I", "-B", os.path.join(BRIEF, rel)], capture_output=True, text=True)
            self.assertEqual(r.stderr.strip().splitlines()[-1], "OK", rel)


if __name__ == "__main__":
    import platform
    print(f"python {platform.python_version()} numpy {np.__version__} machine {platform.machine()}")
    unittest.main(verbosity=2)
