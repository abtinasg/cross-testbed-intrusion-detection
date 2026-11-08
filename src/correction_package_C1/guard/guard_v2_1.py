#!/usr/bin/env python3
"""Resource guard v2.1 (correction package C1). Supersedes resource_guard/guard_reference.py (kept unchanged).

Changes vs v2:
  * grid_decision() validates EXACT expected unique keys (model_id, fold_id) derived from frozen F3 and the
    CV protocol: duplicates, unexpected keys, missing keys, invalid OK records and failure records carrying a
    score are all rejected. COMPLETE only if every expected key has exactly one valid OK record.
  * preflight(): free disk >= 10 GiB, available memory >= 7.0 GiB, no concurrent guarded fit (exclusive lock
    file with stale-PID detection). Thresholds are NOT relaxed here.
run_guarded() semantics are unchanged (child process, 6.5 GiB RSS, 3 h, score=None on any failure).
"""
import csv
import errno
import math
import multiprocessing as mp
import numbers
import os
import shutil
import time

import psutil

GIB = 1024 ** 3
DEFAULT_LIMIT_BYTES = int(6.5 * GIB)
DEFAULT_TIME_LIMIT_S = 3 * 3600
DEFAULT_POLL_S = 2.0
MIN_FREE_DISK_BYTES = 10 * GIB
MIN_AVAILABLE_MEM_BYTES = int(7.0 * GIB)
OUTER_FOLDS = ("Mon", "Tue", "Wed", "Thu", "Fri")
STATUSES = ("OK", "RESOURCE_EXCEEDED", "TIME_EXCEEDED", "FIT_ERROR")


# ----------------------------------------------------------------------------- expected keys
def grid_config_ids(grid_path):
    rows = [r for r in csv.DictReader(l for l in open(grid_path) if not l.startswith("#"))]
    ids = [r["config_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate config_id in grid")
    return ids


def expected_keys(phase, config_ids, selected=None):
    """Exact set of required (model_id, fold_id) keys.
      flat   : every config x outer fold                                   -> 12 x 5 = 60
      tb     : selected config x (outer, inner) with inner != outer         -> 5 x 4 = 20
      nested : every config x (outer, inner) with inner != outer           -> 12 x 20 = 240
    fold_id is 'Wed' (outer) or 'Wed/Tue' (outer/inner held-out day)."""
    pairs = [(o, i) for o in OUTER_FOLDS for i in OUTER_FOLDS if i != o]
    if phase == "flat":
        return {(c, o) for c in config_ids for o in OUTER_FOLDS}
    if phase == "tb":
        if selected not in config_ids:
            raise ValueError("tb phase requires a selected config_id from the grid")
        return {(selected, f"{o}/{i}") for o, i in pairs}
    if phase == "nested":
        return {(c, f"{o}/{i}") for c in config_ids for o, i in pairs}
    raise ValueError(phase)


def _valid_ok(r):
    s = r.get("score")
    if s is None:
        return False
    if isinstance(s, bool):
        return False
    if isinstance(s, numbers.Real):
        return math.isfinite(float(s))
    return True  # non-scalar payload (e.g. score-array reference) is accepted as present


def grid_decision(records, expected):
    """Exact-key completion check. Returns dict(decision in {COMPLETE, CONTINUE, STOP, INVALID}, details)."""
    expected = set(expected)
    seen, dups, unexpected, bad_fail_score, bad_ok, bad_status = {}, [], [], [], [], []
    for r in records:
        k = (r.get("model_id"), r.get("fold"))
        if k not in expected:
            unexpected.append(k)
            continue
        if k in seen:
            dups.append(k)
            continue
        seen[k] = r
        st = r.get("status")
        if st not in STATUSES:
            bad_status.append(k)
        elif st == "OK" and not _valid_ok(r):
            bad_ok.append(k)
        elif st != "OK" and r.get("score") is not None:
            bad_fail_score.append(k)
    missing = sorted(expected - set(seen))
    failed = sorted(k for k, r in seen.items() if r.get("status") in STATUSES[1:])
    details = dict(expected=len(expected), unique_seen=len(seen), duplicates=sorted(set(dups)),
                   unexpected=sorted(set(map(str, unexpected))), missing=missing, failed=failed,
                   failure_records_with_score=sorted(bad_fail_score), invalid_ok_records=sorted(bad_ok),
                   unknown_status=sorted(bad_status))
    if dups or unexpected or bad_fail_score or bad_ok or bad_status:
        return dict(decision="INVALID", reason="record set violates the exact-key contract", **details)
    if failed:
        return dict(decision="STOP", reason="a fit failed: grid cannot be completed comparably; propose a uniform "
                    "alternative resource protocol for separate approval", **details)
    if missing:
        return dict(decision="CONTINUE", reason=f"{len(missing)} expected fits missing", **details)
    return dict(decision="COMPLETE", reason="every expected unique key has exactly one valid OK record", **details)


# ----------------------------------------------------------------------------- preflight
def _pid_alive(pid):
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def acquire_fit_lock(lock_path):
    """Exclusive lock for 'no concurrent guarded fit'. Stale locks (dead PID) are replaced; live ones refuse."""
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            try:
                pid = int(open(lock_path).read().strip() or "0")
            except (OSError, ValueError):
                pid = 0
            if pid and _pid_alive(pid):
                return False
            os.remove(lock_path)  # stale
    return False


def release_fit_lock(lock_path):
    try:
        if int(open(lock_path).read().strip() or "0") == os.getpid():
            os.remove(lock_path)
    except (OSError, ValueError):
        pass


def preflight(work_dir, lock_path, *, _disk_free=None, _available=None):
    """Return dict(ok, checks). Thresholds are fixed (10 GiB disk, 7.0 GiB available memory). The underscore
    arguments exist ONLY for synthetic tests to inject measurements; production calls pass none."""
    free = shutil.disk_usage(work_dir).free if _disk_free is None else _disk_free
    avail = psutil.virtual_memory().available if _available is None else _available
    locked = os.path.exists(lock_path)
    holder = None
    if locked:
        try:
            holder = int(open(lock_path).read().strip() or "0")
        except (OSError, ValueError):
            holder = 0
        locked = bool(holder) and _pid_alive(holder)
    checks = dict(free_disk_bytes=int(free), free_disk_ok=free >= MIN_FREE_DISK_BYTES,
                  available_mem_bytes=int(avail), available_mem_ok=avail >= MIN_AVAILABLE_MEM_BYTES,
                  concurrent_fit=locked, concurrent_ok=not locked, lock_holder=holder)
    return dict(ok=checks["free_disk_ok"] and checks["available_mem_ok"] and checks["concurrent_ok"], checks=checks)


# ----------------------------------------------------------------------------- guarded execution (unchanged semantics)
def _child(fn, args, conn):
    try:
        res = fn(*args)
        conn.send(("OK", res, None))
    except BaseException as e:  # noqa: BLE001
        conn.send(("FIT_ERROR", None, f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def _tree(ps):
    try:
        return [ps] + ps.children(recursive=True)
    except psutil.Error:
        return [ps]


def _tree_rss(ps):
    t = 0
    for p in _tree(ps):
        try:
            t += p.memory_info().rss
        except psutil.Error:
            pass
    return t


def run_guarded(fn, args, model_id, fold, limit_bytes=DEFAULT_LIMIT_BYTES,
                time_limit_s=DEFAULT_TIME_LIMIT_S, poll_s=DEFAULT_POLL_S):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(fn, args, child), daemon=True)
    t0 = time.monotonic()
    p.start()
    child.close()
    ps = psutil.Process(p.pid)
    peak, status, result, error = 0, None, None, None
    while True:
        rss = _tree_rss(ps)
        peak = max(peak, rss)
        if parent.poll():
            status, result, error = parent.recv()
            break
        if rss > limit_bytes:
            status = "RESOURCE_EXCEEDED"
            break
        if time.monotonic() - t0 > time_limit_s:
            status = "TIME_EXCEEDED"
            break
        if not p.is_alive() and not parent.poll():
            status, error = "FIT_ERROR", f"child exited with code {p.exitcode} without a result"
            break
        time.sleep(poll_s)
    if p.is_alive():
        for q in _tree(ps):
            try:
                q.kill()
            except psutil.Error:
                pass
        p.join(timeout=10)
    return dict(model_id=model_id, fold=fold, status=status, peak_rss_bytes=int(peak), limit_bytes=int(limit_bytes),
                elapsed_s=round(time.monotonic() - t0, 3), error=error, score=(result if status == "OK" else None))
