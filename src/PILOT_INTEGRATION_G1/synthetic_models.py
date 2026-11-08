#!/usr/bin/env python3
"""Synthetic workloads for G1 integration tests ONLY. Not a model family of the F3 grid. No real model is fitted.

SYNTH-CENTROID: score = X @ (mean(X | y=1) - mean(X | y=0)) on preprocessed features (pure NumPy, deterministic).
Fault injection (test profiles only; the runner refuses faults under the P1 profile):
  faults = {stage: kind}, stage in {load, assemble, preprocess, fit, score}
  kind: "hog"   allocate and touch memory until the guard kills the child
        "sleep" sleep for 3600 s (timeout test)
        "raise" raise RuntimeError (fit failure)
        "nan"   (stage "score" only) return a NaN score
"""
import time

import numpy as np

SYNTHETIC_FAMILIES = ("SYNTH-CENTROID",)
STAGES = ("load", "assemble", "preprocess", "fit", "score")
_HOLD = []


def inject(faults, stage):
    kind = (faults or {}).get(stage)
    if kind is None or kind == "nan":
        return
    if kind == "hog":
        for _ in range(256):                      # up to 16 GiB in 64 MiB steps; the guard kills first
            _HOLD.append(np.ones(8 * 1024 * 1024))
            time.sleep(0.02)
        time.sleep(3600)
    elif kind == "sleep":
        time.sleep(3600)
    elif kind == "raise":
        raise RuntimeError(f"synthetic fit failure injected at stage {stage}")
    else:
        raise ValueError(f"unknown fault {kind!r}")


def train(family, X, y):
    if family not in SYNTHETIC_FAMILIES:
        raise ValueError(f"model family {family!r} is not a synthetic family")
    if y.sum() == 0 or y.sum() == y.size:
        raise ValueError("synthetic model needs both classes in training")
    return dict(w=X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0))


def score(model, X):
    return X @ model["w"]
