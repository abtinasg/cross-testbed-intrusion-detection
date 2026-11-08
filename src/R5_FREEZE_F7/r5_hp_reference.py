#!/usr/bin/env python3
"""Independent high-precision reference for R5 numerical review and golden tolerance checks (F7).

Used ONLY to assess numbers: after a kappa STOP (spec section 4), and when a golden hash differs across hosts
(spec section 6). It never produces preprocessing statistics for a fit, so it is not a substitute estimator.

Method, the same as C4 compare_cic_like.py and independent of NumPy reductions and of math.fsum:
every float64 z_i is an exact rational p_i/q_i (float.as_integer_ratio) with q_i a power of two. With a common
denominator D, Python big integers give the exact sums S1 = sum p_i' and S2 = sum p_i'^2. Then
mean* = S1/(nD) and var* = (n S2 - S1^2)/(n^2 D^2) exactly (Fraction), and sd* = sqrt(var*) to 80 digits
(Decimal), rounded once to float64.

Tolerances. These were declared in advance in C4 and are defined against the exact reference, not against any
host's values:
  SD_REL_TOL   = 1e-10  relative error of sd vs sd*
  OUT_ABS_TOL  = 1e-9   max |(z - mu_hat)/sd_hat - (z - mu*)/sd*| over the training column
A column is NUMERICALLY MATERIAL if either tolerance is exceeded.
"""
import math
from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np

getcontext().prec = 80
SD_REL_TOL = 1e-10
OUT_ABS_TOL = 1e-9


def exact_moments(z):
    """Returns (mean* as Fraction, var* as Fraction, sd* as float rounded once from an 80-digit sqrt)."""
    vals = [float(v) for v in np.asarray(z, dtype=np.float64).ravel().tolist()]
    if not vals or not all(math.isfinite(v) for v in vals):
        raise ValueError("empty or non-finite column")
    pq = [v.as_integer_ratio() for v in vals]
    D = max(q for _, q in pq)
    ints = [p * (D // q) for p, q in pq]
    n = len(ints)
    S1 = sum(ints)
    S2 = sum(i * i for i in ints)
    mean = Fraction(S1, n * D)
    var = Fraction(n * S2 - S1 * S1, n * n * D * D)
    sd = float((Decimal(var.numerator) / Decimal(var.denominator)).sqrt()) if var else 0.0
    return mean, var, sd


def assess_column(z, mu_hat, sd_hat):
    """Compare saved statistics (mu_hat, sd_hat) with the exact reference on the training column z."""
    z = np.ascontiguousarray(z, dtype=np.float64)
    mean, var, sd = exact_moments(z)
    mu_e = float(mean)
    constant = var == 0
    if constant:
        out_ref = np.zeros_like(z)
        sd_rel = abs(sd_hat - 1.0)                      # ZV-1: the scale must be exactly 1
    else:
        out_ref = (z - mu_e) / sd
        sd_rel = abs(sd_hat - sd) / sd
    out_err = float(np.max(np.abs((z - mu_hat) / sd_hat - out_ref)))
    ulp = math.ulp(mu_e) if mu_e != 0 else math.ulp(0.0)
    return dict(n=int(z.size), exact_constant=bool(constant), mean_exact=repr(mu_e), sd_exact=repr(sd),
                kappa_exact=(None if constant else repr(abs(mu_e) / sd if sd else math.inf)),
                mean_abs_err=repr(abs(mu_hat - mu_e)), mean_err_ulp=repr(abs(mu_hat - mu_e) / ulp),
                sd_rel_err=repr(sd_rel), out_max_abs_err=repr(out_err),
                material=bool(sd_rel > SD_REL_TOL or out_err > OUT_ABS_TOL))
