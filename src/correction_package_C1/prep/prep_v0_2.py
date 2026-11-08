#!/usr/bin/env python3
"""Preprocessing v0.2 (correction package C1): hardened parsing, schema value contract, and hash-based
source provenance. WRAPS the v10 reference (preprocessing_review/prep_reference.py, unchanged); the
R3-R5 rules (caps, vocabulary, signed log1p, standardisation) are reused exactly and NOT modified.

Schema value contract (verified on complete P1+P2 files in Stage 02A-0):
  C-INT  the 47 columns typed `integer` in the frozen schema must hold integer values
  C-NN   the 69 non-categorical columns must be >= 0 (negatives occur only in ICMP Code/Type, = -1)
  C-PROTO Protocol: integer in [0, 255]
  C-ICMP ICMP Code / ICMP Type: integer, -1 (not ICMP) or in [0, 255]
  C-OVF  a numeric literal that overflows float64 (e.g. 1e309) is rejected; only explicit infinity tokens
         in the two R3 columns may produce +-inf
  C-UF   a non-zero literal that underflows to 0.0 (e.g. 1e-400) is rejected
  C-F32  every model-input value must be representable in float32 (scikit-learn RF casts X to float32)
Provenance: fitting requires a TrainingProvenance built from file SHA-256 values that are on the P1
allow-list; unknown or forbidden (P2/P3) hashes are rejected. String tags are not accepted by fit_v2.
"""
import csv
import dataclasses
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "..", "..", "preprocessing_review")
sys.path.insert(0, REF_DIR)
import prep_reference as REF  # noqa: E402  (v10 reference, unchanged)

PreprocessingError = REF.PreprocessingError
VERSION = "prep-0.2-C1-proposed"
F32_MAX = float(np.finfo(np.float32).max)
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri")


# ----------------------------------------------------------------------------- contract
def load_contract(schema_path):
    rows = list(csv.DictReader(open(schema_path)))
    names = [r["column"] for r in rows]
    integer = [r["column"] for r in rows if r["observed_type_P1"] == "integer" and r["observed_type_P2"] == "integer"]
    if len(names) != 72:
        raise PreprocessingError("schema must have 72 columns")
    return dict(schema=names, integer=set(integer), numeric=[c for c in names if c not in REF.CATEGORICAL])


def parse_token_strict(tok, col, contract):
    t = tok.strip()
    if REF.NUM_RE.match(t):
        v = float(t)
        if math.isinf(v):
            raise PreprocessingError(f"C-OVF numeric overflow {tok!r} in {col!r}")
        if v == 0.0 and any(ch in "123456789" for ch in t.split("e")[0].split("E")[0]):
            raise PreprocessingError(f"C-UF non-zero literal underflows to 0 {tok!r} in {col!r}")
        return v
    return REF._parse_token(tok, col)  # explicit inf tokens (R3 columns only) and all rejections


def check_contract(X, contract):
    """Fail closed on any value outside the documented schema contract. Never alters data."""
    X = np.asarray(X, dtype=np.float64)
    schema = contract["schema"]
    pos = {c: j for j, c in enumerate(schema)}
    for c in schema:
        col = X[:, pos[c]]
        fin = col[np.isfinite(col)]
        if c in contract["integer"] and fin.size and np.any(fin != np.floor(fin)):
            raise PreprocessingError(f"C-INT non-integer value in integer column {c!r}")
        if c == "Protocol":
            if np.any(~np.isfinite(col)) or np.any(col != np.floor(col)) or np.any((col < 0) | (col > 255)):
                raise PreprocessingError("C-PROTO Protocol must be an integer in [0, 255]")
        elif c in ("ICMP Code", "ICMP Type"):
            if np.any(~np.isfinite(col)) or np.any(col != np.floor(col)) or np.any((col < -1) | (col > 255)):
                raise PreprocessingError(f"C-ICMP {c} must be an integer, -1 or in [0, 255]")
        elif np.any(col == -np.inf) or (fin.size and np.any(fin < 0)):
            raise PreprocessingError(f"C-NN negative value in non-negative column {c!r}")
    return True


def parse_rows_strict(header, rows, contract, expected_header):
    REF.validate_header(header, expected_header)
    schema = contract["schema"]
    idx = {h: i for i, h in enumerate(header)}
    out = np.empty((len(rows), len(schema)), dtype=np.float64)
    for r, row in enumerate(rows):
        if len(row) != len(header):
            raise PreprocessingError(f"row {r}: {len(row)} fields, expected {len(header)}")
        for j, c in enumerate(schema):
            out[r, j] = parse_token_strict(row[idx[c]], c, contract)
    check_contract(out, contract)
    return out


# ----------------------------------------------------------------------------- provenance
@dataclasses.dataclass(frozen=True)
class TrainingProvenance:
    dataset: str
    days: tuple
    file_sha256: tuple
    allowlist_sha256: str


def load_allowlist(path):
    raw = open(path, "rb").read()
    a = json.loads(raw)
    if set(a["P1"]) & set(a.get("forbidden", {})):
        raise PreprocessingError("allow-list conflict: a hash is both permitted and forbidden")
    a["_sha256"] = hashlib.sha256(raw).hexdigest()
    return a


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def provenance_from_hashes(hashes, allowlist):
    days, hs = [], []
    for h in hashes:
        if h in allowlist.get("forbidden", {}):
            f = allowlist["forbidden"][h]
            raise PreprocessingError(f"forbidden source for fitting: {f.get('partition')} {f.get('file')}")
        if h not in allowlist["P1"]:
            raise PreprocessingError(f"unrecognised source hash {h[:16]}…: not on the P1 allow-list")
        days.append(allowlist["P1"][h]["day"])
        hs.append(h)
    if len(set(days)) != len(days):
        raise PreprocessingError("duplicate source day in provenance")
    order = sorted(range(len(days)), key=lambda i: DAYS.index(days[i]))
    return TrainingProvenance("P1", tuple(days[i] for i in order), tuple(hs[i] for i in order), allowlist["_sha256"])


def provenance_from_files(paths, allowlist):
    """Pilot integration: hash each actual input file, then map through the allow-list."""
    return provenance_from_hashes([sha256_file(p) for p in paths], allowlist)


def _validate_provenance(prov, allowlist, heldout_day):
    if not isinstance(prov, TrainingProvenance):
        raise PreprocessingError("fit_v2 requires a TrainingProvenance object (string tags are not accepted)")
    if prov.dataset != "P1" or prov.allowlist_sha256 != allowlist["_sha256"]:
        raise PreprocessingError("provenance not issued under the current P1 allow-list")
    for h, d in zip(prov.file_sha256, prov.days):
        if h not in allowlist["P1"] or allowlist["P1"][h]["day"] != d:
            raise PreprocessingError("provenance hash/day pair not on the allow-list")
    if not prov.days:
        raise PreprocessingError("empty provenance")
    if heldout_day is not None and heldout_day in prov.days:
        raise PreprocessingError(f"held-out day {heldout_day} is among the training days")


# ----------------------------------------------------------------------------- fit / transform wrappers
def fit_v2(X, contract, branch, provenance, allowlist, heldout_day=None):
    _validate_provenance(provenance, allowlist, heldout_day)
    check_contract(X, contract)
    tag = "P1:" + ",".join(provenance.days)
    art = REF.fit(X, contract["schema"], branch, tag)        # R3-R5 unchanged
    art["version_wrapper"] = VERSION
    art["provenance"] = dataclasses.asdict(provenance)
    return art


def transform_v2(X, art, contract):
    check_contract(X, contract)
    out, cols = REF.transform(X, art)
    if np.any(np.abs(out) > F32_MAX):
        raise PreprocessingError("C-F32 value not representable in float32")
    return out, cols
