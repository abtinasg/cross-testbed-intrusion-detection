#!/usr/bin/env python3
"""Reference implementation of the threshold and timing rules of CV specification v2 + Amendment A1.

Pure functions on NumPy arrays. NO dataset access, NO model. Used only by the unit tests in this
folder, to demonstrate the rules. The Stage 02A-1 code must reproduce these semantics exactly.

Time unit: microseconds (int64), as in the CIC CSVs.
"""
import hashlib
import math

import numpy as np

US = 1_000_000
HOUR_US = 3600 * US
TIMEOUT_US = 120 * US
TIE_SEED = 20261104

# --------------------------------------------------------------------------- timing (A1-T)
AVAILABILITY_RULES = ("T2", "T2c", "T2+120")


def availability_time(start_us, duration_us, rule="T2"):
    """Feature-availability time a under a DOCUMENTED ASSUMPTION (not observed):
      T2      a = start + duration          (last packet; idealised online extractor closes at last packet)
      T2c     a = start + 120 s             (upper bound on last packet under the 120-s flow timeout)
      T2+120  a = start + duration + 120 s  (idealised extractor closing after a 120-s inactivity wait)
    None of these bounds the emission time of the CICFlowMeter fork that produced the data
    (it can emit at end of capture), hence T-a is retrospective only.
    """
    s = np.asarray(start_us, dtype=np.int64)
    d = np.asarray(duration_us, dtype=np.int64)
    if np.any(d < 0):
        raise ValueError("negative duration")
    if rule == "T2":
        return s + d
    if rule == "T2c":
        return s + TIMEOUT_US
    if rule == "T2+120":
        return s + d + TIMEOUT_US
    raise ValueError(rule)


def timing_table(start_us, duration_us, rule="T2", score_delay_us=0):
    """Return dict of the five times for each flow under batch membership by availability time.
      start s; availability a; score availability a + score_delay; batch h = floor(a / 1h);
      batch closure C_h = (h+1)*1h; decision time D = max(C_h, max score-availability in batch).
    """
    s = np.asarray(start_us, dtype=np.int64)
    a = availability_time(s, duration_us, rule)
    sa = a + int(score_delay_us)
    h = a // HOUR_US
    closure = (h + 1) * HOUR_US
    decision = closure.copy()
    for hh in np.unique(h):
        m = h == hh
        decision[m] = max(int(closure[m][0]), int(sa[m].max()))
    return dict(start=s, completion=s + np.asarray(duration_us, dtype=np.int64), availability=a,
                score_available=sa, batch=h, closure=closure, decision=decision)


# --------------------------------------------------------------------------- T-b
def tb_threshold(benign_calibration_scores, alpha):
    """tau = numpy.quantile(x, 1 - alpha, method='higher').
    Finite-sample: with x sorted ascending (0-based), tau = x[ceil((n-1)(1-alpha))]."""
    x = np.asarray(benign_calibration_scores, dtype=np.float64)
    if x.size == 0:
        raise ValueError("empty calibration set: T-b undefined")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha out of range")
    return float(np.quantile(x, 1.0 - alpha, method="higher"))


def tb_alerts(scores, tau):
    """Alert iff score > tau (strict). Ties at tau are NOT alerted."""
    return np.asarray(scores, dtype=np.float64) > tau


def fpr(alerts, is_attack):
    neg = ~np.asarray(is_attack, bool)
    return float(np.asarray(alerts, bool)[neg].mean()) if neg.any() else float("nan")


# --------------------------------------------------------------------------- T-a tie key
def tie_key(features, availability_us, seed=TIE_SEED):
    """Order-invariant, label-free pseudo-random key per flow:
    first 8 bytes of SHA-256(seed || availability time || float64 feature vector).
    Uses only information available at decision time; no id, IP, port, row order or label."""
    f = np.ascontiguousarray(np.asarray(features, dtype=np.float64))
    a = np.asarray(availability_us, dtype=np.int64)
    sb = int(seed).to_bytes(8, "little", signed=False)
    out = np.empty(len(a), dtype=np.uint64)
    for i in range(len(a)):
        hsh = hashlib.sha256(sb + int(a[i]).to_bytes(8, "little", signed=True) + f[i].tobytes()).digest()
        out[i] = int.from_bytes(hsh[:8], "little")
    return out


def ta_select(scores, keys, q):
    """Hourly batch-triage selection within ONE batch.
    k = floor(q*N). Rank by (score desc, key asc). If the k-th cut splits a group of rows that are
    indistinguishable (equal score AND equal key), the whole group is alerted (overshoot, reported).
    Returns (alert mask, info dict)."""
    s = np.asarray(scores, dtype=np.float64)
    kk = np.asarray(keys, dtype=np.uint64)
    n = s.size
    info = dict(N=n, quota=0, alerted=0, overshoot=0, ties_at_cut_score=0, empty=n == 0)
    alerts = np.zeros(n, bool)
    if n == 0:
        return alerts, info
    k = math.floor(q * n)
    info["quota"] = k
    if k == 0:
        return alerts, info
    order = np.lexsort((kk, -s))  # primary: -score asc (score desc); secondary: key asc
    alerts[order[:k]] = True
    cut = order[k - 1]
    info["ties_at_cut_score"] = int((s == s[cut]).sum())
    same = (s == s[cut]) & (kk == kk[cut]) & ~alerts
    alerts |= same
    info["alerted"] = int(alerts.sum())
    info["overshoot"] = int(alerts.sum()) - k
    return alerts, info


def ta_batches(scores, keys, batch_ids, q):
    """Apply ta_select per batch (membership by availability time). Returns alerts, per-batch info."""
    alerts = np.zeros(len(scores), bool)
    infos = {}
    for b in np.unique(batch_ids):
        m = batch_ids == b
        a, inf = ta_select(np.asarray(scores)[m], np.asarray(keys)[m], q)
        alerts[m] = a
        infos[int(b)] = inf
    return alerts, infos


# --------------------------------------------------------------------------- T-a' (sensitivity)
def ta_prime(scores, keys, batch_ids, q, fallback_tau, fallback="tb"):
    """Previous-hour threshold. For batch h, tau'_h = score at rank k_{h-1} (score desc, key asc) of
    batch h-1 (consecutive hour index). A flow of batch h is alerted iff score >= tau'_h.
    If batch h-1 is empty/absent or k_{h-1} == 0:
       fallback == 'tb'       -> use T-b rule (score > fallback_tau)   [primary T-a' variant]
       fallback == 'suppress' -> no alerts in batch h                   [secondary, cost reported]
    First batch of a capture has no previous hour -> same fallback.
    Batch ids must be hour indices of ONE capture day."""
    s = np.asarray(scores, dtype=np.float64)
    kk = np.asarray(keys, dtype=np.uint64)
    b = np.asarray(batch_ids, dtype=np.int64)
    alerts = np.zeros(s.size, bool)
    info = {}
    if s.size == 0:
        return alerts, info
    for h in range(int(b.min()), int(b.max()) + 1):
        m = b == h
        prev = b == h - 1
        n_prev = int(prev.sum())
        k_prev = math.floor(q * n_prev)
        if n_prev == 0 or k_prev == 0:
            if fallback == "tb":
                alerts[m] = s[m] > fallback_tau
                mode = "fallback_tb"
            elif fallback == "suppress":
                mode = "suppressed"
            else:
                raise ValueError(fallback)
            tau = None
        else:
            sp, kp = s[prev], kk[prev]
            order = np.lexsort((kp, -sp))
            tau = float(sp[order[k_prev - 1]])
            alerts[m] = s[m] >= tau
            mode = "previous_hour"
        info[h] = dict(N=int(m.sum()), mode=mode, tau=tau, quota_reference=math.floor(q * int(m.sum())),
                       achieved=int(alerts[m].sum()))
    return alerts, info


# --------------------------------------------------------------------------- recall aggregation
def pooled_recall(alerts, is_attack):
    y = np.asarray(is_attack, bool)
    return float(np.asarray(alerts, bool)[y].mean()) if y.any() else float("nan")


def grouped_recall(alerts, is_attack, groups):
    """Recall per group (e.g., documented episode or time block), plus the unweighted mean over groups
    that contain attacks. Flows inside a group are NOT treated as independent events."""
    y = np.asarray(is_attack, bool)
    a = np.asarray(alerts, bool)
    g = np.asarray(groups)
    per = {}
    for gg in np.unique(g[y]):
        m = (g == gg) & y
        per[gg] = float(a[m].mean())
    return per, (float(np.mean(list(per.values()))) if per else float("nan"))
