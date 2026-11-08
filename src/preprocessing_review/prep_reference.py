#!/usr/bin/env python3
"""Reference implementation of the PROPOSED preprocessing (NW2 / PREPROCESSING_CONFIG_PROPOSED.json).

Status: for review. Implements the proposed R3-R5 rules exactly. Places where the proposal was
silent are implemented FAIL-CLOSED and flagged `PROPOSED-CORRECTION` (see PREPROCESSING_AUDIT.md).
No dataset access here; the tests use synthetic data only.

API
  load_schema(path) -> list[str] (72 names, ordered)        schema_sha256(names) -> hex
  validate_header(header, expected_header)                   raises PreprocessingError
  parse_rows(header, rows, schema) -> float64 (n x 72)       strict token parsing
  fit(X, schema, branch, partition) -> artifact (dict)        training-fold data ONLY
  transform(X, artifact) -> (float64 matrix, output column names)
  artifact_sha256(artifact), save_artifact(a, path), load_artifact(path)
  branch_for_variant("P-LR"|"P-TREE") -> "LR"|"TREE"
"""
import csv
import hashlib
import json
import math
import re

import numpy as np

VERSION = "prep-ref-0.1-proposed"
CATEGORICAL = ("Protocol", "ICMP Code", "ICMP Type")
INF_ALLOWED = ("Flow Bytes/s", "Flow Packets/s")
NUM_RE = re.compile(r"^-?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?$")
INF_TOKENS = {"inf": math.inf, "+inf": math.inf, "infinity": math.inf, "+infinity": math.inf,
              "-inf": -math.inf, "-infinity": -math.inf}
ALLOWED_PARTITION_PREFIX = "P1:"  # source CIC-IDS2017 only


class PreprocessingError(Exception):
    pass


# ----------------------------------------------------------------------------- schema / header
def load_schema(path):
    names = [r["column"] for r in csv.DictReader(open(path))]
    if len(names) != 72 or len(set(names)) != 72:
        raise PreprocessingError(f"schema must have 72 unique names, got {len(names)} / {len(set(names))}")
    for c in CATEGORICAL:
        if c not in names:
            raise PreprocessingError(f"categorical column {c} missing from schema")
    return names


def schema_sha256(names):
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def validate_header(header, expected_header):
    """The CSV header must equal the frozen 91-column header exactly (names, count, order)."""
    if len(header) != len(set(header)):
        dup = sorted({h for h in header if header.count(h) > 1})
        raise PreprocessingError(f"duplicated columns: {dup}")
    missing = [h for h in expected_header if h not in header]
    if missing:
        raise PreprocessingError(f"missing columns: {missing}")
    extra = [h for h in header if h not in expected_header]
    if extra:
        raise PreprocessingError(f"unexpected columns: {extra}")
    if list(header) != list(expected_header):
        raise PreprocessingError("column order differs from the frozen header")


def _parse_token(tok, col):
    t = tok.strip()
    if NUM_RE.match(t):
        return float(t)
    low = t.lower()
    if low in INF_TOKENS:
        if col not in INF_ALLOWED:
            raise PreprocessingError(f"infinity token in non-permitted column {col!r}")
        return INF_TOKENS[low]
    if t == "" or low in ("nan", "-nan", "+nan", "null", "none", "na"):
        raise PreprocessingError(f"missing value in {col!r}")
    raise PreprocessingError(f"malformed token {tok!r} in {col!r}")


def parse_rows(header, rows, schema):
    idx = {h: i for i, h in enumerate(header)}
    for c in schema:
        if c not in idx:
            raise PreprocessingError(f"schema column {c!r} not in header")
    out = np.empty((len(rows), len(schema)), dtype=np.float64)
    for r, row in enumerate(rows):
        if len(row) != len(header):
            raise PreprocessingError(f"row {r}: {len(row)} fields, expected {len(header)}")
        for j, c in enumerate(schema):
            out[r, j] = _parse_token(row[idx[c]], c)
    return out


# ----------------------------------------------------------------------------- helpers
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _hx(v):
    return float(v).hex()


def _unhx(s):
    return float.fromhex(s)


def branch_for_variant(variant):
    m = {"P-LR": "LR", "P-TREE": "TREE"}
    if variant not in m:
        raise PreprocessingError(f"unknown preprocessing variant {variant!r}")
    return m[variant]


def _check_matrix(X, schema):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(schema):
        raise PreprocessingError(f"matrix must be n x {len(schema)}")
    if np.isnan(X).any():
        raise PreprocessingError("NaN present (no imputation; abort)")
    inf_cols = [schema[j] for j in np.flatnonzero(np.isinf(X).any(axis=0))]
    bad = [c for c in inf_cols if c not in INF_ALLOWED]
    if bad:
        raise PreprocessingError(f"infinity in non-permitted columns {bad}")
    return X


# ----------------------------------------------------------------------------- fit / transform
def fit(X, schema, branch, partition):
    """Fit R3/R4/(R5 if LR) on TRAINING-FOLD data only.
    partition: provenance tag, must start with 'P1:' (e.g. 'P1:Tue,Wed,Thu,Fri'). P2/P3 are rejected."""
    if not isinstance(partition, str) or not partition.startswith(ALLOWED_PARTITION_PREFIX):
        raise PreprocessingError(f"fitting on partition {partition!r} is forbidden (source P1 only)")
    if branch not in ("LR", "TREE"):
        raise PreprocessingError("branch must be LR or TREE")
    X = _check_matrix(X, schema)
    if X.shape[0] == 0:
        raise PreprocessingError("empty training matrix")
    pos = {c: j for j, c in enumerate(schema)}
    r3 = {}
    for c in INF_ALLOWED:
        col = X[:, pos[c]]
        fin = col[np.isfinite(col)]
        if fin.size == 0:  # PROPOSED-CORRECTION C-R3a: config silent -> fail closed
            raise PreprocessingError(f"R3: training column {c!r} has no finite values; cap undefined")
        r3[c] = [_hx(fin.min()), _hx(fin.max())]
    r4 = {c: [_hx(v) for v in np.unique(X[:, pos[c]])] for c in CATEGORICAL}
    numeric = [c for c in schema if c not in CATEGORICAL]
    art = dict(version=VERSION, schema_sha256=schema_sha256(schema), schema=list(schema), branch=branch,
               partition=partition, n_train=int(X.shape[0]), r3=r3, r4=r4, numeric=numeric, r5=None)
    if branch == "LR":
        Xc = _apply_r3(X, art)
        Z = signed_log1p(Xc[:, [pos[c] for c in numeric]])
        mu = Z.mean(axis=0)
        sd = Z.std(axis=0)  # population std (ddof=0)
        sd = np.where(sd == 0.0, 1.0, sd)  # zero variance -> divide by 1
        art["r5"] = dict(mean=[_hx(v) for v in mu], std=[_hx(v) for v in sd])
    art["output_columns"] = _output_columns(art)
    return art


def _apply_r3(X, art):
    Xc = X.copy()
    pos = {c: j for j, c in enumerate(art["schema"])}
    for c, (lo, hi) in art["r3"].items():
        j = pos[c]
        Xc[np.isposinf(Xc[:, j]), j] = _unhx(hi)
        Xc[np.isneginf(Xc[:, j]), j] = _unhx(lo)
    return Xc


def _output_columns(art):
    cols = list(art["numeric"])
    for c in CATEGORICAL:
        cols += [f"{c}={_unhx(v):g}" for v in art["r4"][c]] + [f"{c}=__other__"]
    return cols


def transform(X, art):
    schema = art["schema"]
    X = _check_matrix(X, schema)
    pos = {c: j for j, c in enumerate(schema)}
    Xc = _apply_r3(X, art)
    num = Xc[:, [pos[c] for c in art["numeric"]]]
    if art["branch"] == "LR":
        mu = np.array([_unhx(v) for v in art["r5"]["mean"]])
        sd = np.array([_unhx(v) for v in art["r5"]["std"]])
        num = (signed_log1p(num) - mu) / sd
    blocks = [num]
    for c in CATEGORICAL:
        vocab = np.array([_unhx(v) for v in art["r4"][c]])
        col = Xc[:, pos[c]]
        oh = (col[:, None] == vocab[None, :]).astype(np.float64)
        other = (oh.sum(axis=1) == 0).astype(np.float64)[:, None]
        blocks += [oh, other]
    out = np.hstack(blocks)
    if not np.isfinite(out).all():
        raise PreprocessingError("non-finite output after preprocessing")
    return out, list(art["output_columns"])


# ----------------------------------------------------------------------------- artefact persistence
def _canonical(art):
    a = {k: v for k, v in art.items() if k != "artifact_sha256"}
    return json.dumps(a, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def artifact_sha256(art):
    return hashlib.sha256(_canonical(art).encode()).hexdigest()


def save_artifact(art, path):
    a = dict(art)
    a["artifact_sha256"] = artifact_sha256(art)
    with open(path, "w") as fh:
        fh.write(json.dumps(a, sort_keys=True, indent=1))
    return a["artifact_sha256"]


def load_artifact(path):
    a = json.load(open(path))
    h = a.get("artifact_sha256")
    if h != artifact_sha256(a):
        raise PreprocessingError("artifact hash mismatch (tampered or corrupted)")
    return a
