#!/usr/bin/env python3
"""E-R5: exactly-rounded-sum computation of the R5 standardisation statistics (PROPOSED; not frozen).

Definition (for one numeric column, after the unchanged R3 cap and signed log1p z_i = sign(x_i) log1p(|x_i|)):
    n   = number of training rows                                   (exact integer, n < 2**53)
    S   = RN( sum_i z_i )            exact real sum, rounded once to float64 (math.fsum)
    mu  = RN( S / n )                one IEEE division
    d_i = RN( z_i - mu )             IEEE subtraction (identical to NumPy's broadcast subtract)
    q_i = RN( d_i * d_i )            IEEE multiplication (identical to NumPy's x*x)
    Q   = RN( sum_i q_i )            exact real sum, rounded once (math.fsum)
    var = RN( Q / n )                population variance (ddof = 0), unchanged
    sd  = RN( sqrt(var) );  sd := 1.0 if sd == 0.0                  (zero-variance rule unchanged)
RN = IEEE-754 binary64 round-to-nearest-even. The ONLY difference from the v10 NumPy implementation
(np.mean / np.std: sum/n, subtract, square, sum/n, sqrt) is that the two sums S and Q are exact-then-
rounded-once instead of NumPy's blocked/pairwise float64 accumulation whose rounding depends on memory
layout, SIMD path and block width.

Overflow: |z_i| <= log1p(DBL_MAX) ~= 709.78 after signed log1p, so |S| <= 709.78 n and Q <= (2*709.78)**2 n;
for n <= 2**31 both are < 1e16 << DBL_MAX: overflow is impossible on valid inputs. Non-finite z, an
OverflowError/ValueError from fsum, or n == 0 -> PreprocessingError (fail closed).
Chunking: S and Q are exact sums over ALL values regardless of how the column is split into chunks
(fsum consumes one iterator spanning every chunk), so results are bitwise independent of chunk size,
row order and memory layout.
"""
import itertools
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BRIEF, "preprocessing_review"))
sys.path.insert(0, os.path.join(BRIEF, "correction_package_C1", "prep"))
sys.path.insert(0, os.path.join(BRIEF, "correction_package_C2", "lean"))
import prep_reference as REF  # noqa: E402  v10, unchanged
import prep_v0_2 as C1  # noqa: E402       contract + provenance, unchanged
import prep_exact as C2  # noqa: E402      lean + dense exact, unchanged

PreprocessingError = REF.PreprocessingError
VERSION = "E-R5-review-1"


# ----------------------------------------------------------------------------- core statistic
def _chunks(z, chunk_rows):
    z = np.asarray(z, dtype=np.float64)
    if chunk_rows is None or chunk_rows >= z.shape[0]:
        yield np.ascontiguousarray(z)
        return
    for i in range(0, z.shape[0], chunk_rows):
        yield np.ascontiguousarray(z[i:i + chunk_rows])


def er5_mu_sd(z, chunk_rows=None):
    """E-R5 statistics of one already-transformed (capped, signed-log1p) column z."""
    z = np.asarray(z, dtype=np.float64)
    n = int(z.shape[0])
    if n == 0:
        raise PreprocessingError("E-R5: empty column")
    if not np.isfinite(z).all():
        raise PreprocessingError("E-R5: non-finite value in standardisation input")
    try:
        S = math.fsum(itertools.chain.from_iterable(c.tolist() for c in _chunks(z, chunk_rows)))
        mu = S / n
        Q = math.fsum(itertools.chain.from_iterable(((c - mu) * (c - mu)).tolist() for c in _chunks(z, chunk_rows)))
    except (OverflowError, ValueError) as e:
        raise PreprocessingError(f"E-R5: summation failure {e}") from e
    var = Q / n
    sd = math.sqrt(var)
    return mu, (1.0 if sd == 0.0 else sd)


def v10_r5_block(Z):
    """The original (v10) R5 statistics exactly as prep_reference computes them: on the full (n x 69) block,
    mean/std along axis 0 (NumPy 2.4.6), zero std -> 1. Results depend on the block width (layout), so the
    comparator must be given the full 69-column block, as v10 does."""
    Z = np.asarray(Z, dtype=np.float64)
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return mu, sd


# ----------------------------------------------------------------------------- lean fit with chunking + P1 wrapper
def fit_lean_er5(X, schema, branch, partition, chunk_rows=None):
    """Identical to C2.fit_lean (same validation, R3, R4) but E-R5 statistics may be computed in row chunks."""
    art = C2.fit_lean(X, schema, "TREE", partition)          # R3/R4 + all checks (unchanged code)
    art["branch"] = branch
    if branch == "LR":
        pos = {c: j for j, c in enumerate(schema)}
        mus, sds = [], []
        for c in art["numeric"]:
            col = np.ascontiguousarray(X[:, pos[c]], dtype=np.float64)
            if c in art["r3"]:
                col = C2._cap(col, art["r3"][c])
            mu, sd = er5_mu_sd(REF.signed_log1p(col), chunk_rows)
            mus.append(mu); sds.append(sd)
        art["r5"] = dict(mean=[REF._hx(v) for v in mus], std=[REF._hx(v) for v in sds])
    art["version"] = C2.VERSION
    art["output_columns"] = REF._output_columns(art)
    return art


def fit_p1(X, contract, branch, provenance, allowlist, heldout_day=None, chunk_rows=None):
    """Authoritative entry point: P1 hash provenance (C1) + value contract (C1) + lean E-R5 fit."""
    C1._validate_provenance(provenance, allowlist, heldout_day)
    C1.check_contract(X, contract)
    art = fit_lean_er5(X, contract["schema"], branch, "P1:" + ",".join(provenance.days), chunk_rows)
    import dataclasses
    art["provenance"] = dataclasses.asdict(provenance)
    art["er5_version"] = VERSION
    return art
