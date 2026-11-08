#!/usr/bin/env python3
"""Synthetic tests for the proposed preprocessing (R3-R5). NO real data.
Run: python3 -I -B test_prep_reference.py
"""
import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prep_reference as P  # noqa: E402

BRIEF = os.path.dirname(HERE)
SCHEMA_PATH = os.path.join(BRIEF, "FEATURE_SCHEMA_PRIMARY_72_DRAFT.csv")
GRID_PATH = os.path.join(BRIEF, "next_workstream", "CONFIG_GRID_F3_v1.csv")
HEADER = open(os.path.join(HERE, "EXPECTED_CSV_HEADER_91.txt")).read().splitlines()
SCHEMA = P.load_schema(SCHEMA_PATH)
POS = {c: j for j, c in enumerate(SCHEMA)}
NUMERIC = [c for c in SCHEMA if c not in P.CATEGORICAL]
RNG = np.random.default_rng(5)


def synth(n, protos=(0, 1, 6, 17), icmp=(-1, 0, 3, 8), seed=None):
    r = np.random.default_rng(seed) if seed is not None else RNG
    X = np.abs(r.lognormal(3, 3, size=(n, 72)))
    X[:, POS["Protocol"]] = r.choice(protos, n)
    X[:, POS["ICMP Code"]] = r.choice(icmp, n)
    X[:, POS["ICMP Type"]] = r.choice(icmp, n)
    return X


class TestR3(unittest.TestCase):
    def test_cap_is_training_max_and_heldout_uses_training_cap(self):
        tr = synth(500); tr[3, POS["Flow Bytes/s"]] = np.inf
        art = P.fit(tr, SCHEMA, "TREE", "P1:Tue,Wed,Thu,Fri")
        fin = tr[np.isfinite(tr[:, POS["Flow Bytes/s"]]), POS["Flow Bytes/s"]]
        self.assertEqual(float.fromhex(art["r3"]["Flow Bytes/s"][1]), fin.max())
        ho = synth(50); ho[0, POS["Flow Bytes/s"]] = np.inf; ho[1, POS["Flow Bytes/s"]] = fin.max() * 10
        out, cols = P.transform(ho, art)
        j = cols.index("Flow Bytes/s")
        self.assertEqual(out[0, j], fin.max())          # training cap, not held-out max
        self.assertEqual(out[1, j], fin.max() * 10)     # finite held-out value above cap is kept (documented property)

    def test_neg_inf_maps_to_training_min(self):
        tr = synth(200); art = P.fit(tr, SCHEMA, "TREE", "P1:x")
        ho = synth(5); ho[0, POS["Flow Packets/s"]] = -np.inf
        out, cols = P.transform(ho, art)
        self.assertEqual(out[0, cols.index("Flow Packets/s")], tr[:, POS["Flow Packets/s"]].min())

    def test_no_finite_training_values_fails_closed(self):
        tr = synth(20); tr[:, POS["Flow Bytes/s"]] = np.inf
        with self.assertRaises(P.PreprocessingError):
            P.fit(tr, SCHEMA, "TREE", "P1:x")

    def test_inf_in_non_permitted_column_rejected(self):
        tr = synth(20); tr[0, POS["Flow Duration"]] = np.inf
        with self.assertRaises(P.PreprocessingError):
            P.fit(tr, SCHEMA, "TREE", "P1:x")
        art = P.fit(synth(20), SCHEMA, "TREE", "P1:x")
        ho = synth(3); ho[0, POS["Idle Max"]] = np.inf
        with self.assertRaises(P.PreprocessingError):
            P.transform(ho, art)

    def test_no_heldout_statistics(self):
        tr = synth(300, seed=1)
        art1 = P.fit(tr, SCHEMA, "LR", "P1:x")
        h = P.artifact_sha256(art1)
        P.transform(synth(100, seed=2) * 1e6, art1)       # transforming held-out data never changes the artefact
        self.assertEqual(P.artifact_sha256(art1), h)
        import inspect
        self.assertEqual(list(inspect.signature(P.fit).parameters), ["X", "schema", "branch", "partition"])


class TestR4(unittest.TestCase):
    def test_vocabulary_deterministic_sorted_and_permutation_invariant(self):
        tr = synth(400)
        a1 = P.fit(tr, SCHEMA, "TREE", "P1:x"); a2 = P.fit(tr[RNG.permutation(400)], SCHEMA, "TREE", "P1:x")
        self.assertEqual(a1["r4"], a2["r4"]); self.assertEqual(P.artifact_sha256(a1).__len__(), 64)
        vals = [float.fromhex(v) for v in a1["r4"]["Protocol"]]
        self.assertEqual(vals, sorted(vals)); self.assertEqual(vals, [0.0, 1.0, 6.0, 17.0])

    def test_unseen_category_to_other_and_one_hot_exact(self):
        tr = synth(300, protos=(6, 17)); art = P.fit(tr, SCHEMA, "TREE", "P1:x")
        ho = synth(4, protos=(6,)); ho[:, POS["Protocol"]] = [6, 17, 1, 0]
        out, cols = P.transform(ho, art)
        block = out[:, [cols.index("Protocol=6"), cols.index("Protocol=17"), cols.index("Protocol=__other__")]]
        self.assertTrue(np.array_equal(block, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]))

    def test_icmp_sentinel_minus1_is_a_category_and_rows_sum_to_one(self):
        tr = synth(300); art = P.fit(tr, SCHEMA, "TREE", "P1:x")
        out, cols = P.transform(tr, art)
        for c in ("ICMP Code", "ICMP Type", "Protocol"):
            idx = [i for i, n in enumerate(cols) if n.startswith(c + "=")]
            self.assertTrue(np.array_equal(out[:, idx].sum(axis=1), np.ones(len(tr))))
        self.assertIn("ICMP Code=-1", cols); self.assertIn("ICMP Type=-1", cols)

    def test_column_order_consistent(self):
        art = P.fit(synth(200), SCHEMA, "LR", "P1:x")
        _, c1 = P.transform(synth(5), art); _, c2 = P.transform(synth(7), art)
        self.assertEqual(c1, c2); self.assertEqual(c1[:69], NUMERIC)
        cat_part = [n.split("=")[0] for n in c1[69:]]
        self.assertEqual(cat_part, sorted(cat_part, key=lambda x: P.CATEGORICAL.index(x)))


class TestR5(unittest.TestCase):
    def test_signed_log1p_odd_monotone(self):
        x = np.linspace(-1e9, 1e9, 10001)
        y = P.signed_log1p(x)
        self.assertTrue(np.allclose(P.signed_log1p(-x), -y)); self.assertTrue(np.all(np.diff(y) > 0))
        self.assertEqual(P.signed_log1p(np.array([0.0]))[0], 0.0)

    def test_fold_specific_standardisation(self):
        tr = synth(2000); art = P.fit(tr, SCHEMA, "LR", "P1:x")
        out, cols = P.transform(tr, art)
        num = out[:, :69]
        self.assertTrue(np.allclose(num.mean(axis=0), 0, atol=1e-9))
        self.assertTrue(np.allclose(num.std(axis=0), 1, atol=1e-9))
        art_b = P.fit(synth(2000, seed=9) * 3, SCHEMA, "LR", "P1:y")
        self.assertNotEqual(art["r5"], art_b["r5"])   # parameters depend on the fold's training data

    def test_zero_variance_and_finite_outputs(self):
        tr = synth(100); tr[:, POS["Fwd PSH Flags"]] = 0.0
        art = P.fit(tr, SCHEMA, "LR", "P1:x")
        out, cols = P.transform(tr, art)
        self.assertTrue(np.all(out[:, cols.index("Fwd PSH Flags")] == 0.0))
        ho = synth(10); ho[:, POS["Fwd PSH Flags"]] = 5.0; ho[0, POS["Flow Duration"]] = 1.7e308
        out2, _ = P.transform(ho, art)
        self.assertTrue(np.isfinite(out2).all())


class TestColumnsAndTokens(unittest.TestCase):
    def test_header_rejections(self):
        P.validate_header(HEADER, HEADER)
        for bad in (HEADER[:-1], HEADER + ["Extra"], HEADER[:5] + [HEADER[4]] + HEADER[6:],
                    [HEADER[1], HEADER[0]] + HEADER[2:]):
            with self.assertRaises(P.PreprocessingError):
                P.validate_header(bad, HEADER)

    def test_token_rejections_and_inf_tokens(self):
        row = ["1"] * 91
        P.parse_rows(HEADER, [row], SCHEMA)
        jfb = HEADER.index("Flow Bytes/s"); jd = HEADER.index("Flow Duration")
        for tok in ("Infinity", "inf", "-inf"):
            r = list(row); r[jfb] = tok
            P.parse_rows(HEADER, [r], SCHEMA)
        for tok in ("", "NaN", "abc", "1,2", "0x10"):
            r = list(row); r[jd] = tok
            with self.assertRaises(P.PreprocessingError):
                P.parse_rows(HEADER, [r], SCHEMA)
        r = list(row); r[jd] = "inf"
        with self.assertRaises(P.PreprocessingError):
            P.parse_rows(HEADER, [r], SCHEMA)
        with self.assertRaises(P.PreprocessingError):
            P.parse_rows(HEADER, [row[:-1]], SCHEMA)

    def test_nan_matrix_rejected(self):
        X = synth(10); X[2, 5] = np.nan
        with self.assertRaises(P.PreprocessingError):
            P.fit(X, SCHEMA, "TREE", "P1:x")


class TestProvenanceSchemaArtifactBranch(unittest.TestCase):
    def test_no_fit_on_p2_p3(self):
        for tag in ("P2:wed14", "P3:20-02", "", None, "p1:lower"):
            with self.assertRaises(P.PreprocessingError):
                P.fit(synth(10), SCHEMA, "TREE", tag)

    def test_schema_exact_and_ordered_hash(self):
        self.assertEqual(len(SCHEMA), 72)
        self.assertEqual(P.schema_sha256(SCHEMA), hashlib.sha256("\n".join(SCHEMA).encode()).hexdigest())
        self.assertEqual(hashlib.sha256(open(SCHEMA_PATH, "rb").read()).hexdigest(),
                         "6bfc24b8332111fa03729d3597ea4ef8685e11aab999af2906738fd07eee1e9f")
        self.assertNotEqual(P.schema_sha256(SCHEMA), P.schema_sha256(SCHEMA[::-1]))
        self.assertTrue(set(SCHEMA) <= set(HEADER))
        self.assertEqual(len(NUMERIC), 69)

    def test_artifact_reuse_bitwise_identical_and_tamper_detection(self):
        tr = synth(500, seed=3); ho = synth(50, seed=4); ho[0, POS["Flow Bytes/s"]] = np.inf
        for br in ("LR", "TREE"):
            art = P.fit(tr, SCHEMA, br, "P1:x")
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "a.json"); P.save_artifact(art, p)
                a2 = P.load_artifact(p)
                o1, _ = P.transform(ho, art); o2, _ = P.transform(ho, a2); o3, _ = P.transform(ho, a2)
                self.assertEqual(o1.tobytes(), o2.tobytes()); self.assertEqual(o2.tobytes(), o3.tobytes())
                j = json.load(open(p)); j["r3"]["Flow Bytes/s"][1] = (1.0).hex(); json.dump(j, open(p, "w"))
                with self.assertRaises(P.PreprocessingError):
                    P.load_artifact(p)

    def test_branch_mapping_from_frozen_grid(self):
        rows = [r for r in csv.DictReader(l for l in open(GRID_PATH) if not l.startswith("#"))]
        self.assertEqual(len(rows), 12)
        for r in rows:
            br = P.branch_for_variant(r["preprocessing_variant"])
            self.assertEqual(br, "LR" if r["family"] == "LR" else "TREE")
        with self.assertRaises(P.PreprocessingError):
            P.branch_for_variant("P-X")

    def test_tree_branch_identity_vs_lr_branch(self):
        tr = synth(300); ho = synth(20)
        o_t, c_t = P.transform(ho, P.fit(tr, SCHEMA, "TREE", "P1:x"))
        o_l, c_l = P.transform(ho, P.fit(tr, SCHEMA, "LR", "P1:x"))
        self.assertEqual(c_t, c_l)
        self.assertTrue(np.array_equal(o_t[:, :69], ho[:, [POS[c] for c in NUMERIC]]))   # identity on numeric
        self.assertFalse(np.allclose(o_l[:, :69], o_t[:, :69]))
        self.assertTrue(np.array_equal(o_t[:, 69:], o_l[:, 69:]))                           # one-hot identical


if __name__ == "__main__":
    import platform
    print(f"python {platform.python_version()} numpy {np.__version__}")
    unittest.main(verbosity=2)
