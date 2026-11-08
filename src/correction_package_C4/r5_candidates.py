#!/usr/bin/env python3
"""C4: ZV-1 zero-variance repair + the two R5 candidates (for review; NOT frozen).

Shared (unchanged) semantics: R3 cap and R4 vocabulary fitted on the training fold (C2 lean code, which
reuses the v10 rules); z = signed log1p of the capped numeric column; LR output (z - mu) / sd; TREE branch has
no R5; output column order unchanged.

ZV-1 (approved for implementation):
  For each numeric column, on the TRANSFORMED training values z (after R3 cap and signed log1p), BEFORE any
  variance computation:  if min(z) == max(z)  (exactly constant)  ->  mu := z[0] (the exact constant), sd := 1.
  Training values then transform to exactly (z0 - z0) / 1 = 0. Held-out data use the saved mu/sd (no refit).
  No tolerance or variance threshold is used; near-constant columns go through the normal path.
  For non-constant columns the candidate's original statistic is used unchanged, including the original
  rule sd == 0 -> 1 (reachable only through underflow of a tiny genuine variance).

Candidate A (lean NumPy R5): mu = np.mean(z), sd = np.std(z) (ddof=0) on the WHOLE contiguous 1-D float64
  column (NumPy 2.4.6 pairwise summation). Not chunked. Deterministic for a fixed input row order on one host;
  NOT invariant to row order or chunking (rounding of the sums depends on order).
Candidate B (lean E-R5): er5.er5_mu_sd: the two sums sum(z) and sum((z-mu)^2) are each computed exactly and
  rounded once (math.fsum); mu, the differences, squares, the divisions and sqrt are ordinary IEEE operations,
  so the variance as a whole is NOT exact (it inherits the rounding of mu and of each d_i, q_i). Invariant to
  row order and chunking.
"""
import dataclasses
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BRIEF, "correction_package_C3_ER5"))
import er5 as E  # noqa: E402  (C3, unchanged; imports v10 REF, C1, C2 unchanged)

REF, C1, C2 = E.REF, E.C1, E.C2
PreprocessingError = REF.PreprocessingError
VERSION = "C4-r5-candidates-1"
CANDIDATES = ("A_numpy", "B_er5")


# ----------------------------------------------------------------------------- statistics
def stats_A(z):
    z = np.ascontiguousarray(z, dtype=np.float64)
    if z.size == 0 or not np.isfinite(z).all():
        raise PreprocessingError("R5-A: empty or non-finite input")
    mu = float(np.mean(z))
    sd = float(np.std(z))
    return mu, (1.0 if sd == 0.0 else sd)


def stats_B(z, chunk_rows=None):
    return E.er5_mu_sd(z, chunk_rows)


def zv1(z, stat_fn):
    """Returns (mu, sd, is_constant)."""
    z = np.ascontiguousarray(z, dtype=np.float64)
    if z.size == 0:
        raise PreprocessingError("ZV-1: empty column")
    if not np.isfinite(z).all():
        raise PreprocessingError("ZV-1: non-finite input")
    if z.min() == z.max():
        return float(z[0]), 1.0, True
    mu, sd = stat_fn(z)
    return mu, sd, False


# ----------------------------------------------------------------------------- lean fit
def fit_lean(X, schema, branch, partition, candidate, zv=True, chunk_rows=None):
    if candidate not in CANDIDATES:
        raise PreprocessingError(f"unknown candidate {candidate!r}")
    art = C2.fit_lean(X, schema, "TREE", partition)            # R3/R4 + all v10 checks (unchanged)
    art["branch"] = branch
    art["r5_method"] = candidate
    art["zv1"] = bool(zv)
    art["zv1_constant_columns"] = []
    if branch == "LR":
        pos = {c: j for j, c in enumerate(schema)}
        fn = stats_A if candidate == "A_numpy" else (lambda z: stats_B(z, chunk_rows))
        mus, sds = [], []
        for c in art["numeric"]:
            col = np.ascontiguousarray(X[:, pos[c]], dtype=np.float64)
            if c in art["r3"]:
                col = C2._cap(col, art["r3"][c])
            z = REF.signed_log1p(col)
            if zv:
                mu, sd, const = zv1(z, fn)
                if const:
                    art["zv1_constant_columns"].append(c)
            else:
                mu, sd = fn(z)
            mus.append(mu); sds.append(sd)
        art["r5"] = dict(mean=[REF._hx(v) for v in mus], std=[REF._hx(v) for v in sds])
    art["version"] = VERSION
    art["output_columns"] = REF._output_columns(art)
    return art


def transform(X, art, out_path=None):
    """Apply saved statistics; never refits. Identical elementwise semantics to C2/v10."""
    return C2.transform_lean(X, art, out_path)


def fit_p1(X, contract, branch, provenance, allowlist, candidate, heldout_day=None):
    """Authoritative entry point: P1 hash provenance + value contract (C1) + candidate lean fit with ZV-1."""
    C1._validate_provenance(provenance, allowlist, heldout_day)
    C1.check_contract(X, contract)
    art = fit_lean(X, contract["schema"], branch, "P1:" + ",".join(provenance.days), candidate)
    art["provenance"] = dataclasses.asdict(provenance)
    return art
