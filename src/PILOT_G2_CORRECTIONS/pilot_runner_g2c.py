#!/usr/bin/env python3
"""G2-C: corrected G2 runner (PROPOSED; synthetic only). G2 and G1 files are unchanged and reused.

Corrections
  1. Live-parent-aware recovery. Liveness of an execution is established by OS file locks, never by PIDs:
       .pilot_parent.flock  exclusive flock held by the runner for the whole of run() (record + intent cleanup
                            included). The kernel releases it when the holder dies, so a dead parent can never look
                            alive, and PID reuse cannot fake a live parent.
       .pilot_child.flock   exclusive flock held by the guarded worker for its whole life (guarded_worker_nd).
     check_recovery():
       * parent lock BUSY  -> an execution is ACTIVE in another live runner: report ACTIVE_EXECUTION; append nothing,
                              do not stop the pilot. run() is refused with CONCURRENT_RUNNER (nothing recorded).
       * parent lock FREE + INFLIGHT intent present -> the owner terminated without finishing: under the lock (so only
                              one recoverer acts), append ONE failure record INCOMPLETE_PARENT_TERMINATED (score None)
                              if the key has none, and STOP with RECOVERY_DECISION_REQUIRED. An orphan worker is
                              detected by the child lock (busy = still running) and by (pid, create_time) identity;
                              it is reported, never adopted, never killed automatically.
     Identity records {pid, create_time, boot_time, token} are written for parent and worker for audit; a live PID
     whose create_time differs is reported as PID reuse.
  2. Non-daemon guarded worker (guarded_worker_nd.run_guarded_nd): frozen RF n_jobs=4 honoured; whole descendant
     tree monitored by RSS and physical footprint; whole tree terminated on breach, timeout, cancel or exception.
     Limits unchanged (6.5 GiB, 3 h). The child lock also prevents a new fit while an orphan worker still runs.
  3. Durability: write-ahead intent written atomically (temp + fsync + rename + directory fsync); ledger append
     fsync'd before the intent is removed; the parent lock is released only after both.
"""
import fcntl
import json
import os
import sys
import time
import uuid

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
for _p in (os.path.join(BRIEF, "PILOT_BINDING_G2"), os.path.join(BRIEF, "PILOT_INTEGRATION_G1"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pilot_runner_g2 as P2  # noqa: E402
import guarded_worker_nd as W  # noqa: E402

G1, G, R, S, B = P2.G1, P2.G, P2.R, P2.S, P2.B
PilotStop = P2.PilotStop
VERSION = "G2C-pilot-runner-1"
PILOT_TRAINING_AUTHORIZED = False
G2_SHA256 = R.sha256_path(P2.__file__)   # recorded in every ledger record (G2 is reused unchanged)


def identity(pid=None):
    p = psutil.Process(pid or os.getpid())
    return dict(pid=p.pid, create_time=p.create_time(), boot_time=psutil.boot_time())


def identity_alive(ident):
    """'alive' (same process), 'pid_reused' (PID alive but another process), or 'dead'."""
    if not ident or not ident.get("pid"):
        return "unknown"
    try:
        p = psutil.Process(int(ident["pid"]))
        if p.status() == psutil.STATUS_ZOMBIE:
            return "dead"
        same = abs(p.create_time() - float(ident.get("create_time", -1))) < 0.01 and \
            abs(psutil.boot_time() - float(ident.get("boot_time", -1))) < 1.0
        return "alive" if same else "pid_reused"
    except psutil.Error:
        return "dead"


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class PilotRunnerG2C(P2.PilotRunnerG2):
    _held_fd = None

    # -- locks
    @property
    def parent_lock_path(self):
        return os.path.join(self.work_dir, ".pilot_parent.flock")

    @property
    def child_lock_path(self):
        return os.path.join(self.work_dir, ".pilot_child.flock")

    def _try_parent_lock(self):
        fd = os.open(self.parent_lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        os.ftruncate(fd, 0); os.write(fd, json.dumps(identity()).encode()); os.fsync(fd)
        return fd

    @staticmethod
    def _release(fd):
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    # -- intent
    def _read_intent(self):
        try:
            with open(self.inflight_path) as fh:
                return json.load(fh)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            return dict(unreadable=True)

    def _write_intent(self, intent):
        tmp = self.inflight_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(intent, fh); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, self.inflight_path)
        _fsync_dir(self.work_dir)

    def _clear_intent(self):
        for p in (self.child_pid_path, self.inflight_path):
            if os.path.exists(p):
                os.remove(p)
        _fsync_dir(self.work_dir)

    # -- recovery
    def check_recovery(self):
        if self._held_fd is not None:
            return self._recover_locked()
        fd = self._try_parent_lock()
        if fd is None:
            intent = self._read_intent()
            return dict(state="ACTIVE_EXECUTION", intent=intent,
                        parent=identity_alive((intent or {}).get("parent")) if intent and not intent.get("unreadable") else None)
        try:
            return self._recover_locked()
        finally:
            self._release(fd)

    def _recover_locked(self):
        recs = self.records()
        intent = self._read_intent()
        if intent is None:
            return dict(state="CLEAN" if not self.stopped else "STOPPED", reason=self.stop_reason())
        if intent.get("unreadable"):
            self._stop("RECOVERY_DECISION_REQUIRED", dict(intent="unreadable"))
            return dict(state="INCOMPLETE_FIT_DETECTED", key=None, record_existed=None, intent_unreadable=True)
        key = tuple(intent["key"])
        parent_state = identity_alive(intent.get("parent"))
        if parent_state == "alive" and intent["parent"]["pid"] == os.getpid():
            parent_state = "this_process_earlier_run"
        orphan = dict(child=intent.get("child"), child_identity=identity_alive(intent.get("child")),
                      child_lock_busy=W.child_lock_busy(self.child_lock_path))
        orphan["orphan_worker_running"] = orphan["child_lock_busy"]
        recorded = any((r["model_id"], r["fold"]) == key for r in recs)
        if not recorded:
            self._append(dict(model_id=key[0], fold=key[1], status="FIT_ERROR", failure_class="INCOMPLETE_PARENT_TERMINATED",
                              score=None, detail=None, error="execution owner terminated before the fit was recorded; result unknown",
                              parent_state=parent_state, orphan=orphan, intent=intent, version=VERSION, recovered_utc=R._utc(),
                              recovered_by=identity()))
        self._stop("RECOVERY_DECISION_REQUIRED", dict(intent=intent, parent_state=parent_state, orphan=orphan,
                                                      record_existed=recorded))
        return dict(state="INCOMPLETE_FIT_DETECTED", key=key, record_existed=recorded, parent_state=parent_state, orphan=orphan)

    # -- one protected fit
    def run(self, job):
        if not isinstance(job, P2.BoundJob):
            raise PilotStop("a BoundJob(config_id, fold) is required: model family and branch come only from frozen F3")
        fd = self._try_parent_lock()
        if fd is None:
            raise PilotStop("CONCURRENT_RUNNER: another runner is executing in this pilot directory; nothing recorded, pilot not stopped")
        self._held_fd = fd
        try:
            self._recover_locked()
            if self.stopped:
                raise PilotStop(f"PILOT_STOPPED ({self.stop_reason()}): a separate recorded decision is required")
            key = (job.config_id, job.fold_key)
            if key not in self.expected:
                raise PilotStop(f"UNEXPECTED_KEY {key}: not in the F3/{self.phase} expected key set")
            if any((r["model_id"], r["fold"]) == key for r in self.records()):
                raise PilotStop(f"DUPLICATE_KEY {key}: already recorded; no retry")
            if W.child_lock_busy(self.child_lock_path):
                raise PilotStop(self._stop("UNTRACKED_WORKER_ACTIVE", "a guarded worker holds the fit lock without an intent"))
            cfg = self.grid[job.config_id]
            ev = os.path.join(self.work_dir, "evidence", f"{job.config_id}__{job.fold_key.replace('/', '-')}")
            os.mkdir(ev)
            pf1 = self._preflight()
            if not pf1["ok"]:
                raise PilotStop(self._stop("PREFLIGHT_FAIL", pf1))
            pf2 = self._preflight()
            if not pf2["ok"]:
                raise PilotStop(self._stop("PREFLIGHT_UNDER_LOCK_FAIL", pf2))
            intent = dict(key=list(key), config_sha256=cfg.config_sha256, parent=identity(), token=uuid.uuid4().hex,
                          started_epoch=time.time(), utc=R._utc(), child=None)
            self._write_intent(intent)
            spec = dict(config_id=cfg.config_id, config_sha256=cfg.config_sha256, fold=(job.fold.outer, job.fold.inner),
                        faults=list(job.faults), target=self.target, source_paths=list(self.source_paths),
                        allowlist_path=self.allowlist_path, allowlist_sha256=self.allowlist_sha256,
                        profile=P2.dataclasses.asdict(self.profile), evidence_dir=ev, child_pid_path=self.child_pid_path)

            def on_start(ident):
                ident = dict(ident, boot_time=psutil.boot_time())
                self._write_intent(dict(intent, child=ident))
            rss0 = G1._parent_rss()
            g = W.run_guarded_nd(P2._protected_job_g2, (spec,), cfg.config_id, job.fold_key, child_lock_path=self.child_lock_path,
                                 limit_bytes=self.limit_bytes, time_limit_s=self.time_limit_s, poll_s=self.poll_s, on_start=on_start)
            rss1 = G1._parent_rss()
            payload = g.pop("score")
            rec = dict(g, version=VERSION, g2_sha256=G2_SHA256, config_sha256=cfg.config_sha256, family=cfg.family, branch=cfg.branch,
                       target=self.target, parent_rss_before=int(rss0), parent_rss_after=int(rss1), preflight_1=pf1["checks"],
                       preflight_2=pf2["checks"], score=None, detail=None, failure_class=None)
            if rec["status"] == "OK":
                pf3 = self._preflight()
                ok, why = P2.validate_payload(payload, cfg)
                if rec["survivors"]:
                    rec.update(status="FIT_ERROR", failure_class="PROCESS_TREE_NOT_TERMINATED", error=f"survivors {rec['survivors']}")
                elif not pf3["checks"]["free_disk_ok"]:
                    rec.update(status="RESOURCE_EXCEEDED", failure_class="DISK", error="free disk after fit below limit")
                elif rss1 - rss0 > G1.PARENT_MARGIN_BYTES:
                    rec.update(status="RESOURCE_EXCEEDED", failure_class="PARENT_MEMORY_UNACCOUNTED", error=f"parent RSS grew by {rss1 - rss0}")
                elif not ok:
                    rec.update(status="FIT_ERROR", failure_class="INVALID_SCORE", error=why)
                else:
                    rec["score"] = payload.pop("score"); rec["detail"] = payload
            else:
                err = rec.get("error") or ""
                rec["failure_class"] = ("BINDING" if err.startswith(("BindingError", "AdapterError"))
                                        else "CONCURRENT_FIT" if "ConcurrentFit" in err
                                        else "CANCELLED" if rec["status"] == "CANCELLED" else G1._failure_class(rec))
                if rec["status"] == "CANCELLED":
                    rec["status"] = "FIT_ERROR"                     # guard v2.1 status vocabulary
            if rec["status"] != "OK":
                rec["score"] = None
                self._stop(f"FIT_FAILURE_{rec['failure_class']}", dict(key=key, error=rec.get("error")))
            self._append(rec)
            self._clear_intent()
            return rec
        except PilotStop:
            raise
        except BaseException as e:
            self._stop("WRAPPER_ERROR", f"{type(e).__name__}: {e}")
            raise
        finally:
            self._held_fd = None
            self._release(fd)
