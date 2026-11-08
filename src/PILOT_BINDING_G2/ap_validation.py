#!/usr/bin/env python3
"""G2: validity rule for a fold's Average Precision result (CV v2: AP per held-out day; Monday AP undefined).

validate_ap(score, n_pos, n_neg) -> (ok, reason)
  * n_pos, n_neg: exact integer counts of positives and negatives in the validation day (primary or sensitivity
    target as configured), computed in the guarded child next to the score.
  * n_pos == 0 (benign-only day): the ONLY valid result is exactly {"ap": None, "reason": "AP_UNDEFINED_NO_POSITIVES"}.
  * n_pos  > 0: the ONLY valid result is a real float (Python float or NumPy floating; not bool, int, str, dict)
    that is finite and in [0, 1].
  Anything else (negative, > 1, NaN, +-inf, wrong type, the undefined marker on a day with positives, a number on a
  benign-only day, extra keys in the marker, invalid counts) is rejected.
"""
import math
import numbers

AP_UNDEFINED = {"ap": None, "reason": "AP_UNDEFINED_NO_POSITIVES"}


def _count_ok(n):
    return isinstance(n, numbers.Integral) and not isinstance(n, bool) and n >= 0


def validate_ap(score, n_pos, n_neg):
    if not (_count_ok(n_pos) and _count_ok(n_neg)) or n_pos + n_neg == 0:
        return False, f"invalid class counts n_pos={n_pos!r} n_neg={n_neg!r}"
    if n_pos == 0:
        if type(score) is dict and score == AP_UNDEFINED:
            return True, "AP_UNDEFINED_NO_POSITIVES"
        return False, "benign-only validation day must report AP_UNDEFINED_NO_POSITIVES"
    if isinstance(score, bool) or not isinstance(score, numbers.Real) or isinstance(score, numbers.Integral):
        return False, f"AP must be a real float, got {type(score).__name__}"
    s = float(score)
    if not math.isfinite(s):
        return False, f"AP not finite ({s!r})"
    if s < 0.0 or s > 1.0:
        return False, f"AP outside [0, 1] ({s!r})"
    return True, "AP_VALID"
