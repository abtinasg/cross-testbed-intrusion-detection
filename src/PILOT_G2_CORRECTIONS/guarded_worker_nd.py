#!/usr/bin/env python3
"""PROPOSED non-daemon guarded worker (replaces guard v2.1 run_guarded's daemon=True child; v2.1 file unchanged).

Why: inside a daemonic child, joblib reports effective_n_jobs(4) == 1, so RandomForest(n_jobs=4) silently runs with
one worker (M1 campaign + test_m1_equivalence). The frozen F3 setting must be honoured, not silently changed.

run_guarded_nd(fn, args, ...):
  * starts ONE non-daemon worker (spawn). The worker first calls os.setpgrp() (own process group, so every descendant,
    even one re-parented after its parent died, can be signalled), then takes an exclusive flock on child_lock_path
    (held for its whole life; the OS releases it when the worker dies). A second worker on the same lock refuses
    to start fitting -> no concurrent model fitting, including against an orphan from a crashed parent.
  * monitors the WHOLE descendant tree every poll: sum of RSS and sum of macOS physical footprint
    (proc_pid_rusage ri_phys_footprint: includes compressed and swapped anonymous memory, which RSS does not).
    Limit check uses max(tree RSS, tree footprint). Every PID ever seen is remembered with its create_time.
  * on memory breach, timeout, cancellation, parent-side exception or normal completion, it terminates the process
    group and every remembered descendant whose (pid, create_time) still matches, then verifies that none survives.
  * on_start(identity) is called with the worker's (pid, create_time) before monitoring starts (for write-ahead state).
Status: OK | RESOURCE_EXCEEDED | TIME_EXCEEDED | FIT_ERROR | CANCELLED. score is None unless OK.
"""
import ctypes
import fcntl
import multiprocessing as mp
import os
import signal
import time

import psutil

GIB = 1024 ** 3
DEFAULT_LIMIT_BYTES = int(6.5 * GIB)       # unchanged guard v2.1 value
DEFAULT_TIME_LIMIT_S = 3 * 3600
DEFAULT_POLL_S = 2.0


class ConcurrentFit(Exception):
    pass


# ----------------------------------------------------------------------------- macOS physical footprint
class _RUsageInfoV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
        "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
        "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups", "ri_child_pageins",
        "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten")]


try:
    _libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    _libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_RUsageInfoV2)]
except OSError:  # pragma: no cover  (non-macOS)
    _libproc = None


def phys_footprint(pid):
    """Physical footprint in bytes (macOS), or None if unavailable."""
    if _libproc is None:
        return None
    buf = _RUsageInfoV2()
    if _libproc.proc_pid_rusage(int(pid), 2, ctypes.byref(buf)) != 0:
        return None
    return int(buf.ri_phys_footprint)


# ----------------------------------------------------------------------------- worker side
def _worker_entry(fn, args, conn, child_lock_path):
    try:
        os.setpgrp()
        fd = os.open(child_lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ConcurrentFit("another guarded worker holds the fit lock (concurrent or orphaned fit)")
        os.ftruncate(fd, 0); os.write(fd, str(os.getpid()).encode())
        conn.send(("OK", fn(*args), None))
    except BaseException as e:  # noqa: BLE001
        try:
            conn.send(("FIT_ERROR", None, f"{type(e).__name__}: {e}"))
        except Exception:  # noqa: BLE001
            pass
    finally:
        conn.close()


def child_lock_busy(child_lock_path):
    """True if a live worker (possibly orphaned) holds the fit lock."""
    fd = os.open(child_lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


# ----------------------------------------------------------------------------- parent side
def _alive_same(pid, ctime):
    try:
        p = psutil.Process(pid)
        return p.status() != psutil.STATUS_ZOMBIE and abs(p.create_time() - ctime) < 0.01
    except psutil.Error:
        return False


def _terminate(root_pid, seen, reap=lambda: None):
    try:
        os.killpg(root_pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    for pid, ct in list(seen.items()):
        if _alive_same(pid, ct):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        reap()                                              # our own zombie worker still belongs to the group
        left = [pid for pid, ct in seen.items() if _alive_same(pid, ct)]
        if not left and not _group_exists(root_pid):        # also members never seen by a poll (re-parented)
            return []
        try:
            os.killpg(root_pid, signal.SIGKILL)
        except OSError:
            pass
        time.sleep(0.05)
    left = [pid for pid, ct in seen.items() if _alive_same(pid, ct)]
    return left + ([f"pgid:{root_pid}"] if _group_exists(root_pid) else [])


def _group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def run_guarded_nd(fn, args, model_id, fold, *, child_lock_path, limit_bytes=DEFAULT_LIMIT_BYTES,
                   time_limit_s=DEFAULT_TIME_LIMIT_S, poll_s=DEFAULT_POLL_S, on_start=None, cancel=None):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_worker_entry, args=(fn, args, child, child_lock_path), daemon=False)
    t0 = time.monotonic()
    p.start(); child.close()
    root = psutil.Process(p.pid)
    seen = {p.pid: root.create_time()}
    if on_start is not None:
        on_start(dict(pid=p.pid, create_time=seen[p.pid]))
    peak_rss = peak_fp = 0
    status = result = error = None
    try:
        while True:
            try:
                for q in root.children(recursive=True):
                    seen.setdefault(q.pid, q.create_time())
            except psutil.Error:
                pass
            rss = fp = 0
            for pid, ct in list(seen.items()):
                if not _alive_same(pid, ct):
                    continue
                try:
                    rss += psutil.Process(pid).memory_info().rss
                    fp += phys_footprint(pid) or 0
                except psutil.Error:
                    pass
            peak_rss, peak_fp = max(peak_rss, rss), max(peak_fp, fp)
            if parent.poll():
                status, result, error = parent.recv()
                break
            if max(rss, fp) > limit_bytes:
                status, error = "RESOURCE_EXCEEDED", f"tree max(RSS {rss}, footprint {fp}) > limit {limit_bytes}"
                break
            if time.monotonic() - t0 > time_limit_s:
                status, error = "TIME_EXCEEDED", f"elapsed > {time_limit_s} s"
                break
            if cancel is not None and cancel():
                status, error = "CANCELLED", "cancelled by the runner"
                break
            if not p.is_alive() and not parent.poll():
                status, error = "FIT_ERROR", f"worker exited with code {p.exitcode} without a result"
                break
            time.sleep(poll_s)
    finally:
        try:
            for q in root.children(recursive=True):
                seen.setdefault(q.pid, q.create_time())
        except psutil.Error:
            pass
        if status == "OK":
            p.join(timeout=30)                      # let the worker exit normally, then clean any leftovers
        survivors = _terminate(p.pid, seen, reap=lambda: p.join(timeout=0))
        p.join(timeout=10)
    return dict(model_id=model_id, fold=fold, status=status, error=error, score=(result if status == "OK" else None),
                peak_tree_rss_bytes=int(peak_rss), peak_tree_footprint_bytes=int(peak_fp), limit_bytes=int(limit_bytes),
                elapsed_s=round(time.monotonic() - t0, 3), worker_pid=p.pid, descendants_seen=len(seen) - 1,
                survivors=survivors, worker_daemon=False)
