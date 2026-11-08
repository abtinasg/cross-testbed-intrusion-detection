#!/usr/bin/env python3
"""RG8: unified 8 GiB execution guard (PROPOSED; NOT frozen; synthetic use only until independent review).

One implementation for every protected fit of all 12 frozen F3 configurations. Nothing here is typed per family or
per configuration: every limit is read from ONE reviewed policy file (RG8_POLICY_PROPOSED.json) whose SHA-256 is
pinned below. The guard refuses any other policy in production.

Rules (all fail closed)
  RG8-E  environment: the pinned interpreter (realpath), Python/NumPy/scikit-learn/psutil versions, isolated mode (-I).
         Checked in the parent before preflight AND again inside the worker before the job runs.
  RG8-P  host preflight before every fit: installed RAM >= 8 GiB; memory pressure readable and <= the policy level;
         available memory >= policy; free disk >= base + temporary file + swap-growth allowance; physical-footprint,
         swap and pressure sensors readable; no live parent or worker lock. Failure -> re-check every recheck_s for at
         most wait_s, then refuse (no fit is started, nothing is retried).
  RG8-W  ONE non-daemon worker (spawn, own process group, exclusive flock held for its whole life; G2-C
         guarded_worker_nd._worker_entry reused unchanged). The worker runs the WHOLE job function, so loading,
         preprocessing, model fitting and scoring are all inside the monitored process tree.
  RG8-M  monitor every poll_s: for EVERY live process of the tree, RSS and macOS physical footprint. A live process
         whose RSS or footprint cannot be read -> MEASUREMENT_UNAVAILABLE (stop). tree_mem = max(sum RSS, sum
         footprint) > tree_mem_limit -> stop. System swap growth since the worker started > limit -> stop. Pressure
         CRITICAL for >= critical_pressure_stop_s -> stop. Free disk < min -> stop. Elapsed > time limit -> stop.
         A pressure/swap/disk sensor failure -> MEASUREMENT_UNAVAILABLE. An independent watchdog thread kills the
         worker's process group if no complete sample has been taken for max_sample_gap_s (e.g. a hung sensor call)
         -> MONITOR_STALLED. Breaches are checked BEFORE a delivered result is accepted.
  RG8-K  on every exit path: kill the process group and every remembered (pid, create_time); verify that nothing
         survives; survivors turn an OK into a failure.
  RG8-L  durable logging: one JSON line per sample (flushed; fsync at least every fsync_s), plus the final record
         returned to the runner, which appends it with fsync (G2/G2-C ledger) BEFORE clearing its write-ahead intent.
Outcomes: (status, failure_class) with status in the guard v2.1 vocabulary {OK, RESOURCE_EXCEEDED, TIME_EXCEEDED,
FIT_ERROR}. score is None unless OK. There is no retry, resume, row-reduction or parameter argument anywhere.
"""
import ctypes
import hashlib
import json
import multiprocessing as mp
import os
import platform
import re
import shutil
import signal
import sys
import threading
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
if os.path.join(BRIEF, "PILOT_G2_CORRECTIONS") not in sys.path:
    sys.path.insert(0, os.path.join(BRIEF, "PILOT_G2_CORRECTIONS"))
import guarded_worker_nd as W  # noqa: E402  (G2-C, reused unchanged)

VERSION = "RG8-unified-guard-1.1 (V21: post-start cleanup repair)"
GIB = 1024 ** 3
POLICY_PATH = os.path.join(HERE, "RG8_POLICY_PROPOSED.json")
POLICY_SHA256 = "3ba9d56dfd26df5f1e0e196e6bba2b9e081c5158e5a993faef917bcde6891380"
W_SHA256 = "3087124c9f6bab419cd2508e275f08ebc260a52d13bbffc41fe0c0655265076f"
PRESSURE_NORMAL, PRESSURE_WARN, PRESSURE_CRITICAL = 1, 2, 4
POLICY_KEYS = {"policy_id", "status", "applies_to", "environment", "host", "preflight", "monitor", "worker", "on_breach",
               "derivation"}


class RG8Refusal(Exception):
    """Raised before any worker is started (bad policy, environment or code pins)."""


class SensorUnavailable(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- policy
def load_policy(path=POLICY_PATH, *, expected_sha256=None, _unpinned_test_policy=False):
    """Load the reviewed uniform policy. Production: the file must hash to POLICY_SHA256."""
    raw = open(path, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()
    if not _unpinned_test_policy and digest != (expected_sha256 or POLICY_SHA256):
        raise RG8Refusal(f"policy {path} sha256 {digest} != pinned {expected_sha256 or POLICY_SHA256}")
    pol = json.loads(raw)
    if set(pol) != POLICY_KEYS:
        raise RG8Refusal(f"policy keys {sorted(set(pol) ^ POLICY_KEYS)} differ from the reviewed schema")
    if pol["applies_to"] != "ALL_12_F3_CONFIGURATIONS_UNIFORMLY":
        raise RG8Refusal("policy must apply uniformly to all 12 frozen configurations")
    operative = json.dumps({k: v for k, v in pol.items() if k != "derivation"})   # derivation is documentation only
    hit = re.search(r"(?<![A-Za-z])(LR|HGB|RF)(?![A-Za-z])", operative)
    if hit:
        raise RG8Refusal(f"policy contains a family/configuration-specific entry ({hit.group(0)})")
    m, p, h = pol["monitor"], pol["preflight"], pol["host"]
    checks = [0 < m["tree_mem_limit_bytes"] <= int(6.5 * GIB), 0 < m["swap_growth_limit_bytes"] <= 8 * GIB,
              0 < m["time_limit_s"] <= 3 * 3600, 0 < m["critical_pressure_stop_s"] <= 60, 0 < m["poll_s"] <= 2.0,
              m["poll_s"] < m["max_sample_gap_s"] <= 30, m["min_free_disk_bytes"] >= 10 * GIB, 0 < m["fsync_s"] <= 60,
              p["min_available_bytes"] > 0, p["max_pressure_level"] in (PRESSURE_NORMAL, PRESSURE_WARN),
              p["min_free_disk_bytes"] >= m["min_free_disk_bytes"] + p["temp_reserve_bytes"] + m["swap_growth_limit_bytes"],
              0 <= p["wait_s"] <= 3600, 0 < p["recheck_s"] <= 60, h["min_installed_ram_bytes"] >= 8 * GIB,
              pol["worker"] == dict(daemon=False, own_process_group=True, exclusive_lock=True, start_method="spawn"),
              pol["on_breach"] == dict(retry=False, reduce_rows=False, change_parameters=False, score=None, stop_pilot=True)]
    if not all(checks):
        raise RG8Refusal(f"policy fails sanity checks {checks}")
    pol["_sha256"], pol["_pinned"] = digest, not _unpinned_test_policy
    return pol


# ----------------------------------------------------------------------------- sensors (macOS, fail closed)
class _XswUsage(ctypes.Structure):
    _fields_ = [("xsu_total", ctypes.c_uint64), ("xsu_avail", ctypes.c_uint64), ("xsu_used", ctypes.c_uint64),
                ("xsu_pagesize", ctypes.c_uint32), ("xsu_encrypted", ctypes.c_int32)]


try:
    _libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    _libc.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p,
                                   ctypes.c_size_t]
except OSError:  # pragma: no cover
    _libc = None


def _sysctl(name, ctype):
    if _libc is None:
        raise SensorUnavailable(f"sysctl {name}: libSystem unavailable")
    buf, size = ctype(), ctypes.c_size_t(ctypes.sizeof(ctype))
    if _libc.sysctlbyname(name.encode(), ctypes.byref(buf), ctypes.byref(size), None, 0) != 0 or size.value != ctypes.sizeof(ctype):
        raise SensorUnavailable(f"sysctl {name} failed")
    return buf


class Sensors:
    """Production sensors. Every method returns a number or raises SensorUnavailable (never a silent 0)."""

    def installed_ram(self):
        return int(_sysctl("hw.memsize", ctypes.c_uint64).value)

    def pressure(self):
        v = int(_sysctl("kern.memorystatus_vm_pressure_level", ctypes.c_int32).value)
        if v not in (PRESSURE_NORMAL, PRESSURE_WARN, PRESSURE_CRITICAL):
            raise SensorUnavailable(f"unexpected memory-pressure level {v}")
        return v

    def swap_used(self):
        return int(_sysctl("vm.swapusage", _XswUsage).xsu_used)

    def available(self):
        try:
            return int(psutil.virtual_memory().available)
        except Exception as e:  # noqa: BLE001
            raise SensorUnavailable(f"available memory: {e}")

    def disk_free(self, path):
        try:
            return int(shutil.disk_usage(path).free)
        except OSError as e:
            raise SensorUnavailable(f"disk: {e}")

    def proc_rss(self, pid):
        return int(psutil.Process(pid).memory_info().rss)          # psutil.Error propagates to the caller

    def proc_footprint(self, pid):
        fp = W.phys_footprint(pid)
        if fp is None:
            raise SensorUnavailable(f"physical footprint of pid {pid} unavailable")
        return int(fp)


# ----------------------------------------------------------------------------- environment
def check_environment(policy, *, _executable=None):
    e = policy["environment"]
    import numpy, sklearn
    got = dict(executable=os.path.realpath(_executable or sys.executable), python=platform.python_version(),
               numpy=numpy.__version__, sklearn=sklearn.__version__, psutil=psutil.__version__,
               isolated=bool(sys.flags.isolated))
    exp = dict(executable=os.path.realpath(e["executable"]), python=e["python"], numpy=e["numpy"], sklearn=e["sklearn"],
               psutil=e["psutil"], isolated=e["isolated"])
    checks = {k: got[k] == exp[k] for k in exp}
    return dict(ok=all(checks.values()), checks=checks, got=got)


# ----------------------------------------------------------------------------- preflight
def _lock_busy(path):
    return W.child_lock_busy(path)


def preflight(policy, work_dir, *, lock_paths=(), sensors=None):
    s = sensors or Sensors()
    p, m = policy["preflight"], policy["monitor"]
    meas, checks = {}, {}
    try:
        meas = dict(installed_ram=s.installed_ram(), pressure=s.pressure(), available=s.available(),
                    swap_used=s.swap_used(), disk_free=s.disk_free(work_dir), own_footprint=s.proc_footprint(os.getpid()),
                    own_rss=s.proc_rss(os.getpid()))
        checks = dict(sensors_ok=True,
                      installed_ram_ok=meas["installed_ram"] >= policy["host"]["min_installed_ram_bytes"],
                      pressure_ok=meas["pressure"] <= p["max_pressure_level"],
                      available_ok=meas["available"] >= p["min_available_bytes"],
                      disk_ok=meas["disk_free"] >= p["min_free_disk_bytes"],
                      locks_free=not any(_lock_busy(lp) for lp in lock_paths))
    except (SensorUnavailable, psutil.Error, OSError) as e:
        checks = dict(sensors_ok=False, error=f"{type(e).__name__}: {e}")
    ok = all(v for k, v in checks.items() if k != "error")
    return dict(ok=bool(ok), checks=checks, measurements=meas, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def preflight_with_wait(policy, work_dir, *, lock_paths=(), sensors=None, _sleep=time.sleep, _clock=time.monotonic):
    """Bounded wait for a transiently busy desktop. Nothing is started until a preflight passes."""
    t0, log = _clock(), []
    while True:
        pf = preflight(policy, work_dir, lock_paths=lock_paths, sensors=sensors)
        log.append(pf)
        if pf["ok"] or not pf["checks"].get("sensors_ok", False) or not pf["checks"].get("installed_ram_ok", False) \
                or not pf["checks"].get("locks_free", False) or _clock() - t0 >= policy["preflight"]["wait_s"]:
            return dict(pf, attempts=len(log), first=log[0])
        _sleep(policy["preflight"]["recheck_s"])


# ----------------------------------------------------------------------------- worker side
def stage(name):
    """Optional stage marker for job functions (attribution only; coverage does not depend on it)."""
    path = os.environ.get("RG8_STAGE_FILE")
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps(dict(stage=name, t=time.time())) + "\n"); fh.flush()


def _rg8_job(job_fn, job_args, env_pins, stage_file):
    """Runs inside the worker: re-check the pinned environment, then run the whole job (load..score)."""
    import numpy, sklearn
    got = (os.path.realpath(sys.executable), platform.python_version(), numpy.__version__, sklearn.__version__, psutil.__version__)
    exp = (os.path.realpath(env_pins["executable"]), env_pins["python"], env_pins["numpy"], env_pins["sklearn"], env_pins["psutil"])
    if got != exp:
        raise RG8Refusal(f"worker environment {got} != pinned {exp}")
    os.environ["RG8_STAGE_FILE"] = stage_file
    stage("worker_started")
    out = job_fn(*job_args)
    stage("job_returned")
    return out


def _last_stage(stage_file):
    try:
        with open(stage_file) as fh:
            lines = [l for l in fh if l.strip()]
        return json.loads(lines[-1])["stage"] if lines else None
    except (OSError, ValueError, KeyError):
        return None


# ----------------------------------------------------------------------------- parent side
def run_guarded_rg8(job_fn, job_args, *, policy, work_dir, child_lock_path, sample_log_path, stage_file, on_start=None,
                    sensors=None, _executable_for_env=None):
    """Run ONE protected job under RG8. Returns the guard record (never raises for job/resource failures)."""
    s = sensors or Sensors()
    m = policy["monitor"]
    pins = dict(policy["environment"])
    if _executable_for_env:
        pins["executable"] = _executable_for_env
    if W_SHA256 != "__PINNED_AT_PACKAGING__" and sha256_file(W.__file__) != W_SHA256:
        raise RG8Refusal("guarded_worker_nd.py differs from the pinned G2-C version")
    swap0 = s.swap_used()                                         # must be readable before anything starts
    ctx = mp.get_context(policy["worker"]["start_method"])
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=W._worker_entry, args=(_rg8_job, (job_fn, job_args, pins, stage_file), child, child_lock_path),
                    daemon=policy["worker"]["daemon"])
    t0 = time.monotonic()
    # V21 repair: every state used by the cleanup path exists BEFORE p.start(), and every operation after p.start()
    # (root lookup, on_start, opening the sample log, starting the watchdog, monitoring) is inside the try/finally
    # that kills and reaps the whole worker tree.
    peak = dict(rss=0, footprint=0, tree_mem=0, swap_growth=0, procs=0, threads=0)
    mins = dict(available=None, disk=None)
    max_pressure, crit_since, last_sample, last_fsync, n = 0, None, None, t0, 0
    status = failure_class = error = None
    result = None
    root, seen, log, survivors = None, {}, None, []
    wd = dict(last=t0, stalled=None, stop=False)

    def watchdog():                                    # independent of the sampling loop: a hung sensor cannot hide a fit
        while not wd["stop"]:
            gap = time.monotonic() - wd["last"]
            if gap > m["max_sample_gap_s"]:
                wd["stalled"] = gap
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
                return
            time.sleep(min(0.25, m["poll_s"]))
    p.start()
    try:
        child.close()
        try:
            root = psutil.Process(p.pid)
            seen[p.pid] = root.create_time()
        except psutil.Error as e:
            raise SensorUnavailable(f"worker {p.pid} unreadable right after start: {type(e).__name__}")
        if on_start is not None:
            on_start(dict(pid=p.pid, create_time=seen[p.pid]))
        log = open(sample_log_path, "a")
        wd["last"] = time.monotonic()
        wdt = threading.Thread(target=watchdog, daemon=True)
        wdt.start()
        while True:
            now = time.monotonic()
            el = now - t0
            if wd["stalled"] is not None:
                status, failure_class, error = "FIT_ERROR", "MONITOR_STALLED", f"no complete sample for {wd['stalled']:.1f} s > {m['max_sample_gap_s']} s"
                break
            try:
                try:
                    for q in root.children(recursive=True):
                        seen.setdefault(q.pid, q.create_time())
                except psutil.NoSuchProcess:
                    pass                                          # worker exited; its result (if any) is in the pipe
                except psutil.Error as e:
                    raise SensorUnavailable(f"process tree of {p.pid} unreadable: {type(e).__name__}")
                rss = fp = nproc = thr = 0
                for pid, ct in list(seen.items()):
                    if not W._alive_same(pid, ct):
                        continue
                    try:
                        r_, f_ = s.proc_rss(pid), s.proc_footprint(pid)
                        thr += psutil.Process(pid).num_threads()
                    except (psutil.Error, SensorUnavailable) as e:
                        if W._alive_same(pid, ct):                 # still alive and unreadable: fail closed
                            raise SensorUnavailable(f"pid {pid}: {type(e).__name__}: {e}")
                        continue                                  # exited between the checks
                    rss += r_; fp += f_; nproc += 1
                pr, sw, av, dk = s.pressure(), s.swap_used(), s.available(), s.disk_free(work_dir)
            except SensorUnavailable as e:
                status, failure_class, error = "FIT_ERROR", "MEASUREMENT_UNAVAILABLE", str(e)
                break
            last_sample = time.monotonic()
            if wd["stalled"] is not None:
                status, failure_class, error = "FIT_ERROR", "MONITOR_STALLED", f"no complete sample for {wd['stalled']:.1f} s > {m['max_sample_gap_s']} s"
                break
            wd["last"] = last_sample
            tree_mem, growth = max(rss, fp), sw - swap0
            peak.update(rss=max(peak["rss"], rss), footprint=max(peak["footprint"], fp), tree_mem=max(peak["tree_mem"], tree_mem),
                        swap_growth=max(peak["swap_growth"], growth), procs=max(peak["procs"], nproc), threads=max(peak["threads"], thr))
            mins.update(available=av if mins["available"] is None else min(mins["available"], av),
                        disk=dk if mins["disk"] is None else min(mins["disk"], dk))
            max_pressure = max(max_pressure, pr)
            log.write(json.dumps(dict(t=round(el, 2), rss=rss, footprint=fp, tree_mem=tree_mem, procs=nproc, threads=thr,
                                      swap_used=sw, swap_growth=growth, available=av, pressure=pr, disk_free=dk)) + "\n")
            log.flush(); n += 1
            if last_sample - last_fsync >= m["fsync_s"]:
                os.fsync(log.fileno()); last_fsync = last_sample
            # breaches first (a result delivered in the same poll is not accepted over a breach)
            if tree_mem > m["tree_mem_limit_bytes"]:
                status, failure_class, error = "RESOURCE_EXCEEDED", "TREE_MEMORY", f"tree max(RSS {rss}, footprint {fp}) > {m['tree_mem_limit_bytes']}"
            elif growth > m["swap_growth_limit_bytes"]:
                status, failure_class, error = "RESOURCE_EXCEEDED", "SWAP_GROWTH", f"swap growth {growth} > {m['swap_growth_limit_bytes']}"
            elif dk < m["min_free_disk_bytes"]:
                status, failure_class, error = "RESOURCE_EXCEEDED", "DISK", f"free disk {dk} < {m['min_free_disk_bytes']}"
            elif el > m["time_limit_s"]:
                status, failure_class, error = "TIME_EXCEEDED", "TIMEOUT", f"elapsed {el:.0f} s > {m['time_limit_s']} s"
            else:
                if pr == PRESSURE_CRITICAL:
                    crit_since = el if crit_since is None else crit_since
                    if el - crit_since >= m["critical_pressure_stop_s"]:
                        status, failure_class, error = "RESOURCE_EXCEEDED", "CRITICAL_PRESSURE", f"pressure CRITICAL for {el - crit_since:.1f} s"
                else:
                    crit_since = None
            if status is not None:
                break
            if parent.poll():
                kind, result, err = parent.recv()
                status, error = kind, err
                if kind != "OK":
                    failure_class = ("CONCURRENT_FIT" if "ConcurrentFit" in (err or "") else
                                     "ENVIRONMENT" if "RG8Refusal" in (err or "") else None)
                break
            if not p.is_alive() and not parent.poll():
                status, failure_class, error = "FIT_ERROR", "WORKER_DIED", f"worker exited with code {p.exitcode} without a result"
                break
            time.sleep(m["poll_s"])
    except BaseException as e:                                     # parent-side exception: still kill the tree
        status, failure_class, error = "FIT_ERROR", "GUARD_EXCEPTION", f"{type(e).__name__}: {e}"
        e.rg8_survivors = survivors                               # the list object is filled by the finally below
        raise
    finally:
        wd["stop"] = True
        if root is not None:
            try:
                for q in root.children(recursive=True):
                    seen.setdefault(q.pid, q.create_time())
            except psutil.Error:
                pass
        if status == "OK":
            p.join(timeout=30)
        survivors.extend(W._terminate(p.pid, seen, reap=lambda: p.join(timeout=0)))
        if p.is_alive():                                          # e.g. root unreadable, so not in `seen`: kill directly
            p.kill()
        p.join(timeout=10)
        if p.is_alive():
            survivors.append(p.pid)
        if log is not None:
            try:
                log.flush(); os.fsync(log.fileno())
            finally:
                log.close()
    if status == "OK" and survivors:
        status, failure_class, error = "FIT_ERROR", "PROCESS_TREE_NOT_TERMINATED", f"survivors {survivors}"
    return dict(status=status, failure_class=failure_class, error=error, payload=(result if status == "OK" else None),
                stage_at_end=_last_stage(stage_file), elapsed_s=round(time.monotonic() - t0, 3), worker_pid=p.pid,
                worker_daemon=policy["worker"]["daemon"], descendants_seen=len(seen) - 1, survivors=survivors,
                samples=n, swap_at_start=swap0, peak_tree_rss=peak["rss"], peak_tree_footprint=peak["footprint"],
                peak_tree_mem=peak["tree_mem"], peak_swap_growth=peak["swap_growth"], max_procs=peak["procs"],
                max_threads=peak["threads"], min_available=mins["available"], min_disk_free=mins["disk"],
                max_pressure=max_pressure, policy_id=policy["policy_id"], policy_sha256=policy["_sha256"],
                policy_pinned=policy["_pinned"], guard_version=VERSION)
