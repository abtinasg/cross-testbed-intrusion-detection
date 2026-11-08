#!/usr/bin/env python3
"""Synthetic CIC-like data for 8 GiB resource feasibility (NO dataset rows, NO project labels).

Feature values: for each of the 69 numeric primary features, a zero/lognormal mixture matched to Stage 02A-0
AGGREGATE statistics (P1 zero share, median, p99 from the 2% sample; clipped to the complete-file P1 [min, max];
floored for integer columns). This is the C4 generator idea (compare_cic_like.py), re-implemented here per column
and per chunk. Categorical columns: Protocol in {0, 1, 6, 17}; ICMP Code/Type in {-1, 0, 3, 8}. A few +inf
values in Flow Bytes/s and Flow Packets/s (R3 path).
Labels: synthetic Bernoulli(0.24) "SYN-ATTACK"/"BENIGN", INDEPENDENT of the features (no signal). For trees this
is close to a worst case for tree size (no early purity). Attempted Category: -1, except a 0.5% random share.
"""
import csv
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
MATRIX = os.path.join(BRIEF, "..", "STAGE02A_0_DATASET_AUDIT", "07_CROSS_DATASET_FEATURE_MATRIX.csv")
P1_ROWS = (("Mon", 371624), ("Tue", 322078), ("Wed", 496641), ("Thu", 362076), ("Fri", 547557))


def _stats():
    with open(MATRIX) as fh:
        return {r["column"]: r for r in csv.DictReader(fh)}


def synth_matrix(n, seed, schema, integer_cols):
    st = _stats()
    r = np.random.default_rng(seed)
    X = np.empty((n, len(schema)), dtype=np.float64)
    for j, c in enumerate(schema):
        if c == "Protocol":
            X[:, j] = r.choice([0, 1, 6, 17], n, p=[0.001, 0.01, 0.79, 0.199]); continue
        if c in ("ICMP Code", "ICMP Type"):
            X[:, j] = r.choice([-1, 0, 3, 8], n, p=[0.985, 0.005, 0.005, 0.005]); continue
        m = st[c]
        lo, hi = float(m["P1_min"]), float(m["P1_max"])
        zs = float(m["sample_zero_share_P1"]) if lo <= 0.0 else 0.0
        med, p99 = float(m["sample_median_P1"]), float(m["sample_p99_P1"])
        base = med if med > 0 else max(p99 / 20.0, 1e-3)
        sig = max(math.log(p99 / base) / 2.326, 0.5) if p99 > base else 1.0
        col = np.zeros(n)
        nz = r.random(n) >= zs
        v = np.clip(r.lognormal(math.log(base), sig, int(nz.sum())), max(lo, 1e-12 if lo <= 0 else lo), hi)
        if c in integer_cols:
            v = np.maximum(np.floor(v), max(lo, 0.0))
        col[nz] = v
        X[:, j] = col
    for c in ("Flow Bytes/s", "Flow Packets/s"):
        X[r.choice(n, size=max(1, n // 150000), replace=False), schema.index(c)] = np.inf
    return X


def synth_labels(n, seed):
    r = np.random.default_rng(seed + 7)
    lab = np.where(r.random(n) < 0.24, "SYN-ATTACK", "BENIGN")
    att = np.where(r.random(n) < 0.005, 0, -1)
    return lab, att


def _fmt_col(v, integer):
    if integer:
        return [("inf" if x == np.inf else str(int(x))) for x in v.tolist()]
    return [("inf" if x == np.inf else repr(x)) for x in v.tolist()]


def write_day_csv(path, n, seed, schema, integer_cols, header, chunk=50000):
    """Chunked writer in the frozen 91-column header layout. Memory O(chunk). Returns (bytes, n_attempted)."""
    pos = {c: j for j, c in enumerate(schema)}
    n_att = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for a in range(0, n, chunk):
            m = min(chunk, n - a)
            X = synth_matrix(m, seed * 1000 + a // chunk, schema, integer_cols)
            lab, att = synth_labels(m, seed * 1000 + a // chunk)
            n_att += int((att != -1).sum())
            cols = {}
            for h in header:
                if h in pos:
                    cols[h] = _fmt_col(X[:, pos[h]], h in integer_cols or h in ("Protocol", "ICMP Code", "ICMP Type"))
                elif h == "id":
                    cols[h] = [str(i) for i in range(a + 1, a + m + 1)]
                elif h == "Label":
                    cols[h] = lab.tolist()
                elif h == "Attempted Category":
                    cols[h] = [f"{float(x):.1f}" for x in att.tolist()]
                elif h in ("Flow ID", "Src IP", "Dst IP", "Timestamp"):
                    cols[h] = ["s"] * m
                else:
                    cols[h] = ["0"] * m
            w.writerows(zip(*(cols[h] for h in header)))
    return os.path.getsize(path), n_att
