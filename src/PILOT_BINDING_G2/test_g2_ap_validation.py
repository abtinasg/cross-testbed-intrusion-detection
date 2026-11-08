#!/usr/bin/env python3
"""G2 tests: Average Precision validity (unit rule + end to end through the guarded runner). Synthetic only.
Run: python3 -I -B test_g2_ap_validation.py
"""
import math
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "PILOT_INTEGRATION_G1"))
import g2_fixtures as X  # noqa: E402
import pilot_runner_g2 as P2  # noqa: E402

AP = P2.AP


class T1_Rule(unittest.TestCase):
    def test_valid_numeric_ap(self):
        for s in (0.0, 1.0, 0.5, np.float64(0.25), np.float32(0.75), 1e-300):
            self.assertEqual(AP.validate_ap(s, 5, 10), (True, "AP_VALID"), s)
        self.assertTrue(AP.validate_ap(1.0, 7, 0)[0])                                 # all-positive day: AP defined

    def test_invalid_numeric_ap(self):
        for s in (-0.1, -1e-300, -0.0 - 1e-17, 1.0000000000000002, 1.5, 2.0, math.nan, math.inf, -math.inf,
                  np.float64("nan"), np.float64("inf")):
            ok, why = AP.validate_ap(s, 5, 10)
            self.assertFalse(ok, s)

    def test_invalid_types(self):
        for s in (True, False, 1, 0, np.int64(1), "0.5", None, [0.5], {"ap": 0.5}, AP.AP_UNDEFINED):
            self.assertFalse(AP.validate_ap(s, 5, 10)[0], repr(s))

    def test_benign_only_day(self):
        self.assertEqual(AP.validate_ap(dict(AP.AP_UNDEFINED), 0, 50), (True, "AP_UNDEFINED_NO_POSITIVES"))
        for s in (0.0, 0.5, 1.0, math.nan, None, {"ap": None}, dict(AP.AP_UNDEFINED, note="x"),
                  {"ap": None, "reason": "AP_UNDEFINED"}, {"ap": 0.0, "reason": "AP_UNDEFINED_NO_POSITIVES"}):
            self.assertFalse(AP.validate_ap(s, 0, 50)[0], repr(s))

    def test_invalid_counts(self):
        for n_pos, n_neg in ((-1, 5), (5, -1), (0, 0), (True, 5), (5.0, 5), (None, 5), ("5", 5)):
            self.assertFalse(AP.validate_ap(0.5, n_pos, n_neg)[0], (n_pos, n_neg))


class T2_EndToEnd(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory(); self.d = self._td.name
        self.fx = X.fixture(os.path.join(self.d, "src"), mon_benign=True)
        self.n = 0

    def tearDown(self):
        self._td.cleanup()

    def _run(self, job):
        self.n += 1
        return X.runner(self.fx, os.path.join(self.d, f"w{self.n}")).run(job)

    def test_valid_ap_and_benign_only_day_accepted(self):
        rec = self._run(X.job("LR-01", "Thu"))
        self.assertEqual(rec["status"], "OK", rec.get("error")); self.assertTrue(0.0 <= rec["score"] <= 1.0)
        self.assertGreater(rec["detail"]["n_pos"], 0)
        rec = self._run(X.job("HGB-01", "Mon"))
        self.assertEqual(rec["status"], "OK", rec.get("error"))
        self.assertEqual(rec["score"], AP.AP_UNDEFINED); self.assertEqual(rec["detail"]["n_pos"], 0)

    def test_every_invalid_result_is_a_failure_without_score(self):
        cases = [(("score_value", "neg"), "Thu"), (("score_value", "gt1"), "Thu"), (("score_value", "nan"), "Thu"),
                 (("score_value", "inf"), "Thu"), (("score_value", "bool"), "Thu"), (("score_value", "int"), "Thu"),
                 (("score_value", "str"), "Thu"), (("score_value", "undef"), "Thu"), (("score_value", "undef_extra"), "Mon"),
                 (("score_value", "gt1"), "Mon"), (("payload", "extra_key"), "Thu"), (("payload", "no_counts"), "Thu")]
        for fault, day in cases:
            rec = self._run(X.job("RF-01", day, faults=(fault,)))
            self.assertEqual((rec["status"], rec["failure_class"]), ("FIT_ERROR", "INVALID_SCORE"), (fault, day, rec.get("error")))
            self.assertIsNone(rec["score"]); self.assertIsNone(rec["detail"])


if __name__ == "__main__":
    import platform
    print(f"python {platform.python_version()} machine {platform.machine()} runner {P2.VERSION}")
    unittest.main(verbosity=2)
