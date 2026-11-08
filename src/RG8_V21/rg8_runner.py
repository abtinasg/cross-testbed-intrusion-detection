#!/usr/bin/env python3
"""PilotRunnerRG8: the G2-C pilot runner with the unified RG8 guard integrated (PROPOSED; synthetic only).

Reused UNCHANGED (subclassed, not edited): G2 (F3 binding, AP validation, ledger, stop file) and G2-C (parent flock,
atomic write-ahead intent, live-parent-aware crash recovery, child flock). Replaced for every fit:
  * guard v2.1 preflight (7.0 GiB)            -> RG8 preflight (pinned uniform policy, bounded wait, fail closed)
  * run_guarded_nd (RSS/footprint only)       -> rg8_guard.run_guarded_rg8 (tree RSS + footprint + swap + pressure
                                                  + disk + time + sensor availability; fail closed)
Added checks (no parameter is tunable by the caller):
  * the policy, interpreter and limits come only from the pinned policy file; the constructor has no limit argument;
  * no hidden sampling: n_train / n_val in the payload must equal the profile's pinned row counts for the fold
    and target (fewer rows -> FIT_ERROR ROW_COUNT_MISMATCH);
  * no retry: duplicate keys, a stopped pilot and an incomplete earlier fit are refused exactly as in G2/G2-C;
  * every record carries the policy SHA-256, environment, guard version, peaks and the stage reached.
PILOT_TRAINING_AUTHORIZED stays False (inherited refusal of the P1 profile / P1 allow-list).
"""
import os
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
for _p in (os.path.join(BRIEF, "PILOT_G2_CORRECTIONS"), os.path.join(BRIEF, "PILOT_BINDING_G2"),
           os.path.join(BRIEF, "PILOT_INTEGRATION_G1"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pilot_runner_g2c as C  # noqa: E402
import rg8_guard as RG  # noqa: E402

P2, G1, R, S = C.P2, C.G1, C.R, C.S
PilotStop = C.PilotStop
VERSION = "RG8-pilot-runner-1 (PROPOSED)"
PILOT_TRAINING_AUTHORIZED = False


def expected_counts(profile, fold, target):
    val_day = fold.inner if fold.inner is not None else fold.outer
    return (sum(profile.expected_rows(d, target) for d in fold.training_days()), profile.expected_rows(val_day, target))


class PilotRunnerRG8(C.PilotRunnerG2C):
    def __init__(self, work_dir, allowlist_path, source_paths, profile, *, phase, selected=None, target="primary",
                 _unpinned_test_policy_path=None, _test_sensors=None, _test_executable=None):
        if _unpinned_test_policy_path is not None:
            self.policy = RG.load_policy(_unpinned_test_policy_path, _unpinned_test_policy=True)
        else:
            self.policy = RG.load_policy()
        self.sensors = _test_sensors or RG.Sensors()
        self._test_executable = _test_executable
        env = RG.check_environment(self.policy, _executable=_test_executable)
        if not env["ok"]:
            raise PilotStop(f"ENVIRONMENT_MISMATCH: {env['checks']} got {env['got']}")
        self.environment = env["got"]
        m = self.policy["monitor"]
        super().__init__(work_dir, allowlist_path, source_paths, profile, phase=phase, selected=selected, target=target,
                         limit_bytes=m["tree_mem_limit_bytes"], time_limit_s=m["time_limit_s"], poll_s=m["poll_s"])
        if not self.policy["_pinned"] and (profile == S.P1_PROFILE or self.allowlist_sha256 == S.P1_PROFILE.allowlist_sha256):
            raise PilotStop("an unpinned test policy can never be used with the P1 profile")

    # -- RG8 preflight in the G2 shape (ok, checks)
    def _preflight(self):
        pf = RG.preflight_with_wait(self.policy, self.work_dir, lock_paths=(self.child_lock_path,), sensors=self.sensors)
        checks = dict(pf["checks"], free_disk_ok=pf["checks"].get("disk_ok", False))
        return dict(ok=pf["ok"], checks=checks, measurements=pf.get("measurements"), attempts=pf.get("attempts"))

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
            if C.W.child_lock_busy(self.child_lock_path):
                raise PilotStop(self._stop("UNTRACKED_WORKER_ACTIVE", "a guarded worker holds the fit lock without an intent"))
            env = RG.check_environment(self.policy, _executable=self._test_executable)
            if not env["ok"]:
                raise PilotStop(self._stop("ENVIRONMENT_MISMATCH", env))
            cfg = self.grid[job.config_id]
            ev = os.path.join(self.work_dir, "evidence", f"{job.config_id}__{job.fold_key.replace('/', '-')}")
            os.mkdir(ev)
            pf1 = self._preflight()
            if not pf1["ok"]:
                raise PilotStop(self._stop("PREFLIGHT_FAIL", pf1))
            intent = dict(key=list(key), config_sha256=cfg.config_sha256, parent=C.identity(), token=uuid.uuid4().hex,
                          started_epoch=time.time(), utc=R._utc(), child=None, policy_sha256=self.policy["_sha256"])
            self._write_intent(intent)
            spec = dict(config_id=cfg.config_id, config_sha256=cfg.config_sha256, fold=(job.fold.outer, job.fold.inner),
                        faults=list(job.faults), target=self.target, source_paths=list(self.source_paths),
                        allowlist_path=self.allowlist_path, allowlist_sha256=self.allowlist_sha256,
                        profile=P2.dataclasses.asdict(self.profile), evidence_dir=ev, child_pid_path=self.child_pid_path)

            def on_start(ident):
                self._write_intent(dict(intent, child=dict(ident, boot_time=C.psutil.boot_time())))
            rss0 = G1._parent_rss()
            g = RG.run_guarded_rg8(P2._protected_job_g2, (spec,), policy=self.policy, work_dir=self.work_dir,
                                   child_lock_path=self.child_lock_path, sample_log_path=os.path.join(ev, "rg8_samples.jsonl"),
                                   stage_file=os.path.join(ev, "rg8_stages.jsonl"), on_start=on_start, sensors=self.sensors,
                                   _executable_for_env=self._test_executable)
            rss1 = G1._parent_rss()
            payload = g.pop("payload")
            rec = dict(g, model_id=cfg.config_id, fold=job.fold_key, version=VERSION, g2_sha256=C.G2_SHA256,
                       config_sha256=cfg.config_sha256, family=cfg.family, branch=cfg.branch, target=self.target,
                       environment=env["got"], parent_rss_before=int(rss0), parent_rss_after=int(rss1),
                       preflight_1=pf1["checks"], preflight_1_measurements=pf1["measurements"], preflight_attempts=pf1["attempts"],
                       score=None, detail=None)
            if rec["status"] == "OK":
                ok, why = P2.validate_payload(payload, cfg)
                want = expected_counts(self.profile, job.fold, self.target)
                if rss1 - rss0 > G1.PARENT_MARGIN_BYTES:
                    rec.update(status="RESOURCE_EXCEEDED", failure_class="PARENT_MEMORY_UNACCOUNTED", error=f"parent RSS grew by {rss1 - rss0}")
                elif not ok:
                    rec.update(status="FIT_ERROR", failure_class="INVALID_SCORE", error=why)
                elif (payload["n_train"], payload["n_val"]) != want:
                    rec.update(status="FIT_ERROR", failure_class="ROW_COUNT_MISMATCH",
                               error=f"(n_train, n_val) {(payload['n_train'], payload['n_val'])} != pinned {want}")
                else:
                    rec["score"] = payload.pop("score"); rec["detail"] = payload
            if rec["status"] != "OK":
                rec["score"] = None
                rec["failure_class"] = rec.get("failure_class") or G1._failure_class(rec)
                self._stop(f"FIT_FAILURE_{rec['failure_class']}", dict(key=key, error=rec.get("error")))
            self._append(rec)                       # durable (fsync) BEFORE the intent is cleared
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
