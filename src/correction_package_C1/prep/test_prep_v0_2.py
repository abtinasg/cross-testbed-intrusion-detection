#!/usr/bin/env python3
"""Tests for preprocessing v0.2 hardening + provenance (synthetic data only; the real allow-list file is
read for structure, but no dataset file is opened). Also re-runs the v10 reference suite unchanged.
Run: python3 -I -B test_prep_v0_2.py
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prep_v0_2 as V  # noqa: E402

REF = V.REF
BRIEF = os.path.abspath(os.path.join(HERE, "..", ".."))
SCHEMA_PATH = os.path.join(BRIEF, "FEATURE_SCHEMA_PRIMARY_72_DRAFT.csv")
HEADER = open(os.path.join(BRIEF, "preprocessing_review", "EXPECTED_CSV_HEADER_91.txt")).read().splitlines()
CON = V.load_contract(SCHEMA_PATH)
SCHEMA = CON["schema"]
POS = {c: j for j, c in enumerate(SCHEMA)}
REAL_ALLOWLIST = os.path.join(HERE, "..", "provenance", "P1_SOURCE_ALLOWLIST.json")
RNG = np.random.default_rng(13)


def synth(n, seed=None):
    r = np.random.default_rng(seed) if seed is not None else RNG
    X = np.abs(r.lognormal(3, 2, size=(n, 72)))
    for c in CON["integer"]:
        X[:, POS[c]] = np.floor(X[:, POS[c]])
    X[:, POS["Protocol"]] = r.choice([0, 1, 6, 17], n)
    X[:, POS["ICMP Code"]] = r.choice([-1, 0, 3], n)
    X[:, POS["ICMP Type"]] = r.choice([-1, 0, 8], n)
    return X


def row_tokens(overrides=None):
    row = ["0"] * 91
    for c in ("Protocol",):
        row[HEADER.index(c)] = "6"
    row[HEADER.index("ICMP Code")] = "-1"; row[HEADER.index("ICMP Type")] = "-1"
    for k, v in (overrides or {}).items():
        row[HEADER.index(k)] = v
    return row


def test_allowlist(tmpdir, n_days=5):
    files, al = {}, {"P1": {}, "forbidden": {}}
    for i, d in enumerate(V.DAYS[:n_days]):
        p = os.path.join(tmpdir, f"{d}.csv"); open(p, "w").write(f"synthetic-{d}\n")
        h = V.sha256_file(p); files[d] = p; al["P1"][h] = dict(day=d, file=f"{d}.csv")
    fp = os.path.join(tmpdir, "P2_fake.csv"); open(fp, "w").write("synthetic-P2\n")
    al["forbidden"][V.sha256_file(fp)] = dict(partition="P2", file="P2_fake.csv")
    ap = os.path.join(tmpdir, "allow.json"); json.dump(al, open(ap, "w"))
    return V.load_allowlist(ap), files, fp


class TestParsingHardening(unittest.TestCase):
    def test_valid_row_unaltered(self):
        X = V.parse_rows_strict(HEADER, [row_tokens({"Flow Duration": "12345", "Flow Bytes/s": "123.5"})], CON, HEADER)
        self.assertEqual(X[0, POS["Flow Duration"]], 12345.0); self.assertEqual(X[0, POS["Flow Bytes/s"]], 123.5)

    def test_overflow_vs_explicit_infinity(self):
        X = V.parse_rows_strict(HEADER, [row_tokens({"Flow Bytes/s": "inf"})], CON, HEADER)
        self.assertTrue(np.isposinf(X[0, POS["Flow Bytes/s"]]))
        for col in ("Flow Bytes/s", "Flow Duration"):
            with self.assertRaisesRegex(REF.PreprocessingError, "C-OVF"):
                V.parse_rows_strict(HEADER, [row_tokens({col: "1e309"})], CON, HEADER)

    def test_underflow_rejected_but_zero_accepted(self):
        with self.assertRaisesRegex(REF.PreprocessingError, "C-UF"):
            V.parse_rows_strict(HEADER, [row_tokens({"Flow IAT Mean": "1e-400"})], CON, HEADER)
        for z in ("0", "0.0", "0.000e5"):
            V.parse_rows_strict(HEADER, [row_tokens({"Flow IAT Mean": z})], CON, HEADER)

    def test_non_integer_categorical_rejected(self):
        for col, v in (("Protocol", "6.5"), ("ICMP Code", "0.5"), ("ICMP Type", "3.25")):
            with self.assertRaises(REF.PreprocessingError):
                V.parse_rows_strict(HEADER, [row_tokens({col: v})], CON, HEADER)
        V.parse_rows_strict(HEADER, [row_tokens({"Protocol": "6.0"})], CON, HEADER)  # integer-valued float ok

    def test_out_of_domain_categorical_rejected(self):
        for col, v in (("Protocol", "-1"), ("Protocol", "256"), ("ICMP Code", "-2"), ("ICMP Type", "300"),
                       ("Protocol", "inf")):
            with self.assertRaises(REF.PreprocessingError):
                V.parse_rows_strict(HEADER, [row_tokens({col: v})], CON, HEADER)
        V.parse_rows_strict(HEADER, [row_tokens({"Protocol": "255", "ICMP Code": "255", "ICMP Type": "-1"})], CON, HEADER)

    def test_integer_and_nonnegative_contract(self):
        ic = sorted(CON["integer"] - set(REF.CATEGORICAL))[0]
        with self.assertRaisesRegex(REF.PreprocessingError, "C-INT"):
            V.parse_rows_strict(HEADER, [row_tokens({ic: "2.5"})], CON, HEADER)
        with self.assertRaisesRegex(REF.PreprocessingError, "C-NN"):
            V.parse_rows_strict(HEADER, [row_tokens({"Flow Duration": "-5"})], CON, HEADER)
        X = synth(5); X[0, POS["Flow Packets/s"]] = -np.inf
        with self.assertRaisesRegex(REF.PreprocessingError, "C-NN"):
            V.check_contract(X, CON)

    def test_float32_representability(self):
        al_dir = tempfile.mkdtemp(); al, files, _ = test_allowlist(al_dir)
        prov = V.provenance_from_hashes([V.sha256_file(files[d]) for d in ("Tue", "Wed")], al)
        tr = synth(100); art = V.fit_v2(tr, CON, "TREE", prov, al)
        ho = synth(3); ho[0, POS["Flow Duration"]] = 1e39
        with self.assertRaisesRegex(REF.PreprocessingError, "C-F32"):
            V.transform_v2(ho, art, CON)

    def test_contract_counts(self):
        self.assertEqual(len(CON["integer"]), 47); self.assertEqual(len(CON["numeric"]), 69)


class TestProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(); self.al, self.files, self.p2 = test_allowlist(self.tmp)

    def test_allowlisted_files_produce_provenance(self):
        prov = V.provenance_from_files([self.files[d] for d in ("Fri", "Tue", "Thu", "Wed")], self.al)
        self.assertEqual(prov.days, ("Tue", "Wed", "Thu", "Fri")); self.assertEqual(prov.dataset, "P1")
        art = V.fit_v2(synth(200), CON, "LR", prov, self.al, heldout_day="Mon")
        self.assertEqual(art["provenance"]["days"], ["Tue", "Wed", "Thu", "Fri"] if isinstance(art["provenance"]["days"], list) else ("Tue", "Wed", "Thu", "Fri"))

    def test_unknown_hash_rejected(self):
        p = os.path.join(self.tmp, "x.csv"); open(p, "w").write("not allow-listed")
        with self.assertRaisesRegex(REF.PreprocessingError, "unrecognised"):
            V.provenance_from_files([p], self.al)

    def test_forbidden_p2_rejected(self):
        with self.assertRaisesRegex(REF.PreprocessingError, "forbidden source.*P2"):
            V.provenance_from_files([self.files["Tue"], self.p2], self.al)

    def test_string_tag_rejected_and_heldout_overlap(self):
        with self.assertRaisesRegex(REF.PreprocessingError, "TrainingProvenance"):
            V.fit_v2(synth(10), CON, "TREE", "P1:Tue,Wed", self.al)
        prov = V.provenance_from_files([self.files["Tue"], self.files["Wed"]], self.al)
        with self.assertRaisesRegex(REF.PreprocessingError, "held-out"):
            V.fit_v2(synth(10), CON, "TREE", prov, self.al, heldout_day="Wed")

    def test_forged_provenance_rejected(self):
        forged = V.TrainingProvenance("P1", ("Tue",), ("0" * 64,), self.al["_sha256"])
        with self.assertRaises(REF.PreprocessingError):
            V.fit_v2(synth(10), CON, "TREE", forged, self.al)
        other = V.TrainingProvenance("P1", ("Tue",), (V.sha256_file(self.files["Tue"]),), "f" * 64)
        with self.assertRaisesRegex(REF.PreprocessingError, "allow-list"):
            V.fit_v2(synth(10), CON, "TREE", other, self.al)
        swapped = V.TrainingProvenance("P1", ("Wed",), (V.sha256_file(self.files["Tue"]),), self.al["_sha256"])
        with self.assertRaises(REF.PreprocessingError):
            V.fit_v2(synth(10), CON, "TREE", swapped, self.al)

    def test_duplicate_day_rejected(self):
        with self.assertRaises(REF.PreprocessingError):
            V.provenance_from_files([self.files["Tue"], self.files["Tue"]], self.al)

    def test_real_allowlist_structure(self):
        al = V.load_allowlist(REAL_ALLOWLIST)
        self.assertEqual(sorted(v["day"] for v in al["P1"].values()), sorted(V.DAYS))
        self.assertEqual(len(al["forbidden"]), 2)
        self.assertTrue(all(v["partition"] == "P2" for v in al["forbidden"].values()))
        self.assertIn("51fe5dc962626efb4ae70dce0303072fb780da0932822b651202ee9c2fbc1aff", al["P1"])  # monday.csv
        self.assertIn("4ceed4ba26973004e3a86ae0c57d6a4ddd9742f1b222f44209cd7c1c48618fdb", al["forbidden"])  # P2 14-02
        with self.assertRaisesRegex(REF.PreprocessingError, "forbidden"):
            V.provenance_from_hashes(["a16b76a454a25fa4fd84c320f32e3164b22c09f537be3e77450ddeb2168a7622"], al)


class TestR3R5Unchanged(unittest.TestCase):
    def test_v10_reference_suite_still_passes(self):
        t = os.path.join(BRIEF, "preprocessing_review", "test_prep_reference.py")
        r = subprocess.run([sys.executable, "-I", "-B", t], capture_output=True, text=True)
        self.assertIn("OK", r.stderr.splitlines()[-1], r.stderr[-500:])

    def test_wrapper_output_equals_reference_output(self):
        al, files, _ = test_allowlist(tempfile.mkdtemp())
        prov = V.provenance_from_hashes([V.sha256_file(files[d]) for d in ("Tue", "Wed", "Thu", "Fri")], al)
        tr, ho = synth(300, seed=1), synth(40, seed=2)
        for br in ("LR", "TREE"):
            a_ref = REF.fit(tr, SCHEMA, br, "P1:Tue,Wed,Thu,Fri"); a_v2 = V.fit_v2(tr, CON, br, prov, al, "Mon")
            o1, _ = REF.transform(ho, a_ref); o2, _ = V.transform_v2(ho, a_v2, CON)
            self.assertEqual(o1.tobytes(), o2.tobytes())


if __name__ == "__main__":
    import platform
    print(f"python {platform.python_version()} numpy {np.__version__}")
    unittest.main(verbosity=2)
