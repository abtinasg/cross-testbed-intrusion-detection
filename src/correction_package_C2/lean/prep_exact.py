#!/usr/bin/env python3
"""U1 preprocessing (correction package C2): exact-statistics dense reference + memory-lean implementation.

Rules R3, R4, R5 and the output layout are those of the v10 reference (preprocessing_review/prep_reference.py,
unchanged). The ONLY numerical specification added (proposal E-R5):
  R5 mean and population variance are computed with an exactly-rounded sum (math.fsum):
      mu  = fsum(z) / n          var = fsum((z - mu)**2) / n          sd = sqrt(var), or 1.0 if var == 0
  This makes the fitted parameters independent of summation order, memory layout, chunking, SIMD width and
  thread count, so the dense reference and the lean implementation are bitwise identical by construction.
  (NumPy's mean/std are order/layout dependent: column-block widths 1 and 4 changed results by up to
  1.6e-13 in a check on NumPy 2.4.6.)

fit_dense / transform_dense : straightforward dense implementation (exact-stat reference)
fit_lean  / transform_lean  : column-at-a-time; output written into a pre-allocated (optionally disk-backed)
                              Fortran-order array; no full-size intermediates.
Both return artefacts in the v10 format (floats as hex), so v10 transform semantics apply.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "preprocessing_review"))
import prep_reference as REF  # noqa: E402  (unchanged)

PreprocessingError = REF.PreprocessingError
VERSION = "prep-0.3-C2-exact-proposed"


def _check_inputs(X, schema, branch, partition):
    if not isinstance(partition, str) or not partition.startswith(REF.ALLOWED_PARTITION_PREFIX):
        raise PreprocessingError(f"fitting on partition {partition!r} is forbidden (source P1 only)")
    if branch not in ("LR", "TREE"):
        raise PreprocessingError("branch must be LR or TREE")
    if X.ndim != 2 or X.shape[1] != len(schema) or X.shape[0] == 0:
        raise PreprocessingError("bad training matrix shape")


def _col_checks(col, name):
    if np.isnan(col).any():
        raise PreprocessingError("NaN present (no imputation; abort)")
    if name not in REF.INF_ALLOWED and np.isinf(col).any():
        raise PreprocessingError(f"infinity in non-permitted columns ['{name}']")


def _exact_mu_sd(z):
    n = z.shape[0]
    mu = math.fsum(z.tolist()) / n
    d = z - mu
    var = math.fsum((d * d).tolist()) / n
    sd = math.sqrt(var)
    return mu, (1.0 if sd == 0.0 else sd)


def _cap(col, r3):
    lo, hi = REF._unhx(r3[0]), REF._unhx(r3[1])
    c = col.copy()
    c[np.isposinf(c)] = hi
    c[np.isneginf(c)] = lo
    return c


def _finalize(art):
    art["output_columns"] = REF._output_columns(art)
    art["version"] = VERSION
    return art


# ----------------------------------------------------------------------------- dense exact reference
def fit_dense(X, schema, branch, partition):
    X = np.asarray(X, dtype=np.float64)
    _check_inputs(X, schema, branch, partition)
    art = REF.fit(X, schema, "TREE", partition)          # R3/R4 + all v10 checks, unchanged
    art["branch"] = branch
    if branch == "LR":
        pos = {c: j for j, c in enumerate(schema)}
        Xc = REF._apply_r3(X, art)
        Z = REF.signed_log1p(Xc[:, [pos[c] for c in art["numeric"]]])
        mus, sds = zip(*(_exact_mu_sd(np.ascontiguousarray(Z[:, j])) for j in range(Z.shape[1])))
        art["r5"] = dict(mean=[REF._hx(v) for v in mus], std=[REF._hx(v) for v in sds])
    return _finalize(art)


def transform_dense(X, art):
    return REF.transform(X, art)


# ----------------------------------------------------------------------------- lean
def fit_lean(X, schema, branch, partition):
    """X: (n x 72) float64 array or np.memmap (any order). Reads one column at a time."""
    _check_inputs(X, schema, branch, partition)
    pos = {c: j for j, c in enumerate(schema)}
    r3, r4 = {}, {}
    for j, c in enumerate(schema):                           # validation pass, column by column
        _col_checks(np.asarray(X[:, j], dtype=np.float64), c)
    for c in REF.INF_ALLOWED:
        col = np.asarray(X[:, pos[c]], dtype=np.float64)
        fin = col[np.isfinite(col)]
        if fin.size == 0:
            raise PreprocessingError(f"R3: training column {c!r} has no finite values; cap undefined")
        r3[c] = [REF._hx(fin.min()), REF._hx(fin.max())]
    for c in REF.CATEGORICAL:
        r4[c] = [REF._hx(v) for v in np.unique(np.asarray(X[:, pos[c]], dtype=np.float64))]
    numeric = [c for c in schema if c not in REF.CATEGORICAL]
    art = dict(version=VERSION, schema_sha256=REF.schema_sha256(schema), schema=list(schema), branch=branch,
               partition=partition, n_train=int(X.shape[0]), r3=r3, r4=r4, numeric=numeric, r5=None)
    if branch == "LR":
        mus, sds = [], []
        for c in numeric:
            col = np.ascontiguousarray(X[:, pos[c]], dtype=np.float64)
            if c in r3:
                col = _cap(col, r3[c])
            mu, sd = _exact_mu_sd(REF.signed_log1p(col))
            mus.append(mu); sds.append(sd)
        art["r5"] = dict(mean=[REF._hx(v) for v in mus], std=[REF._hx(v) for v in sds])
    return _finalize(art)


def transform_lean(X, art, out_path=None):
    """Column-at-a-time transform into a pre-allocated Fortran-order float64 array (in RAM, or a disk-backed
    .npy memmap if out_path is given). Elementwise operations are identical to the v10/dense transform."""
    schema = art["schema"]
    pos = {c: j for j, c in enumerate(schema)}
    n = X.shape[0]
    cols = list(art["output_columns"])
    if out_path:
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float64, shape=(n, len(cols)), fortran_order=True)
    else:
        out = np.empty((n, len(cols)), dtype=np.float64, order="F")
    k = 0
    lr = art["branch"] == "LR"
    for i, c in enumerate(art["numeric"]):
        col = np.ascontiguousarray(X[:, pos[c]], dtype=np.float64)
        _col_checks(col, c)
        if c in art["r3"]:
            col = _cap(col, art["r3"][c])
        if lr:
            mu = REF._unhx(art["r5"]["mean"][i]); sd = REF._unhx(art["r5"]["std"][i])
            col = (REF.signed_log1p(col) - mu) / sd
        out[:, k] = col
        k += 1
    for c in REF.CATEGORICAL:
        col = np.asarray(X[:, pos[c]], dtype=np.float64)
        _col_checks(col, c)
        vocab = [REF._unhx(v) for v in art["r4"][c]]
        hit = np.zeros(n, dtype=bool)
        for v in vocab:
            m = col == v
            out[:, k] = m
            hit |= m
            k += 1
        out[:, k] = ~hit
        k += 1
    if not np.isfinite(out).all():
        raise PreprocessingError("non-finite output after preprocessing")
    return out, cols
