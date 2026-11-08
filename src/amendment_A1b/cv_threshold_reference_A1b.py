#!/usr/bin/env python3
"""Amendment A1b reference semantics. Extends the FROZEN A1 module (imported unchanged).

New:
  * ta_select_hardcap   - hard alert cap: never exceed the quota; an indistinguishable boundary group
                          that does not fit is omitted (no backfill); unused capacity reported.
  * ta_select_overshoot - A1's all-group rule, retained ONLY as a budget-overshooting sensitivity policy.
  * quota_relative / quota_absolute - relative floor(q*N_h) vs absolute hourly alert capacity K.
  * check_t2c_precondition - T2c requires every Flow Duration <= 120 s; validated, not assumed.
  * ta_prime_versioned  - T-a' with reset to the source-calibrated T-b threshold whenever the model
                          version differs between batch h-1 and batch h.
Synthetic use only; no dataset access, no model.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "amendment_A1"))
import cv_threshold_reference as A1  # noqa: E402  (frozen F4a, not modified)

TIMEOUT_US = A1.TIMEOUT_US


# ----------------------------------------------------------------------------- quotas
def quota_relative(q, n_h):
    """Relative quota: floor(q * N_h). A fraction of flows, NOT a number of analyst-review slots."""
    if n_h < 0 or not (0.0 <= q <= 1.0):
        raise ValueError("bad q or N")
    return math.floor(q * n_h)


def quota_absolute(K, n_h):
    """Absolute hourly alert capacity: min(K, N_h) alerts may be surfaced in the hour."""
    if K < 0 or n_h < 0 or int(K) != K:
        raise ValueError("K must be a non-negative integer")
    return min(int(K), int(n_h))


# ----------------------------------------------------------------------------- selection
def _groups(scores, keys):
    """Rank (score desc, key asc) and return the ordered list of indistinguishable groups
    (equal score AND equal key), each as an index array."""
    s = np.asarray(scores, dtype=np.float64)
    k = np.asarray(keys, dtype=np.uint64)
    order = np.lexsort((k, -s))
    groups, start = [], 0
    for i in range(1, len(order) + 1):
        if i == len(order) or s[order[i]] != s[order[i - 1]] or k[order[i]] != k[order[i - 1]]:
            groups.append(order[start:i])
            start = i
    return groups


def ta_select_hardcap(scores, keys, quota):
    """PRIMARY hard-budget selection within one batch.
    Walk groups in rank order; alert a whole group only if it fits in the remaining quota.
    The first group that does not fit is OMITTED and selection STOPS (no backfill with lower-ranked
    flows), so the alerted set is always a score-upper set and never exceeds the quota."""
    n = len(scores)
    alerts = np.zeros(n, bool)
    info = dict(N=n, quota=int(quota), alerted=0, unused_capacity=int(quota), omitted_boundary_group_size=0,
                empty=n == 0)
    if n == 0 or quota <= 0:
        return alerts, info
    used = 0
    for g in _groups(scores, keys):
        if used + len(g) <= quota:
            alerts[g] = True
            used += len(g)
            if used == quota:
                break
        else:
            info["omitted_boundary_group_size"] = int(len(g))
            break
    info["alerted"] = used
    info["unused_capacity"] = int(quota) - used
    return alerts, info


def ta_select_overshoot(scores, keys, quota):
    """SENSITIVITY ONLY (budget-overshooting): A1 rule. Alerts the whole boundary group even if the
    quota is exceeded. Implemented here with an explicit quota (A1's version derives it from q)."""
    n = len(scores)
    alerts = np.zeros(n, bool)
    info = dict(N=n, quota=int(quota), alerted=0, overshoot=0, empty=n == 0)
    if n == 0 or quota <= 0:
        return alerts, info
    used = 0
    for g in _groups(scores, keys):
        if used >= quota:
            break
        alerts[g] = True
        used += len(g)
    info["alerted"] = used
    info["overshoot"] = max(0, used - int(quota))
    return alerts, info


def recall_loss(alerts_hard, alerts_over, is_attack):
    """Recall(overshoot) - Recall(hardcap) over the given flows; nan if no attacks."""
    y = np.asarray(is_attack, bool)
    if not y.any():
        return float("nan")
    return float(np.asarray(alerts_over, bool)[y].mean() - np.asarray(alerts_hard, bool)[y].mean())


# ----------------------------------------------------------------------------- T2c precondition
def check_t2c_precondition(duration_us):
    """T2c (a = s + 120 s) is a valid upper bound on completion only if every duration <= 120 s.
    Returns the number of violating rows; raises if any."""
    d = np.asarray(duration_us, dtype=np.int64)
    bad = int((d > TIMEOUT_US).sum())
    if bad:
        raise ValueError(f"T2c precondition violated: {bad} flows with duration > 120 s")
    return 0


def availability_time_checked(start_us, duration_us, rule):
    if rule == "T2c":
        check_t2c_precondition(duration_us)
    return A1.availability_time(start_us, duration_us, rule)


# ----------------------------------------------------------------------------- T-a' with model versions
def ta_prime_versioned(scores, keys, batch_ids, model_version, quota_fn, tb_tau_by_version):
    """Previous-hour threshold carried ONLY when batch h-1 and batch h were scored by the SAME model
    version. Otherwise (version change), or if batch h-1 is empty / has quota 0 / h is first:
    reset to the source-calibrated T-b threshold of the CURRENT model version (score > tau).
    quota_fn(N) gives the reference quota (relative or absolute). model_version: one value per flow,
    constant within a batch."""
    s = np.asarray(scores, dtype=np.float64)
    kk = np.asarray(keys, dtype=np.uint64)
    b = np.asarray(batch_ids, dtype=np.int64)
    v = np.asarray(model_version)
    alerts = np.zeros(s.size, bool)
    info = {}
    if s.size == 0:
        return alerts, info
    ver_of = {}
    for h in np.unique(b):
        vs = np.unique(v[b == h])
        if len(vs) != 1:
            raise ValueError(f"batch {h} scored by more than one model version")
        ver_of[int(h)] = vs[0]
    for h in range(int(b.min()), int(b.max()) + 1):
        m = b == h
        if not m.any():
            info[h] = dict(N=0, mode="empty")
            continue
        cur = ver_of[h]
        prev = b == h - 1
        k_prev = quota_fn(int(prev.sum())) if prev.any() else 0
        if prev.any() and ver_of[h - 1] != cur:
            mode = "reset_version_change"
        elif not prev.any() or k_prev == 0:
            mode = "fallback_tb"
        else:
            mode = "previous_hour"
        if mode == "previous_hour":
            sp, kp = s[prev], kk[prev]
            order = np.lexsort((kp, -sp))
            tau = float(sp[order[k_prev - 1]])
            alerts[m] = s[m] >= tau
        else:
            tau = float(tb_tau_by_version[cur])
            alerts[m] = s[m] > tau
        info[h] = dict(N=int(m.sum()), mode=mode, model_version=cur, tau=tau, achieved=int(alerts[m].sum()),
                       quota_reference=quota_fn(int(m.sum())))
    return alerts, info
