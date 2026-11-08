#!/usr/bin/env python3
"""G2 pilot runner (PROPOSED; synthetic only): G1 fail-closed execution, bound to frozen F3, strict AP validity,
and crash-safe execution state. G1 (`PILOT_INTEGRATION_G1/pilot_guarded_runner.py`) is reused, not modified.

Changes vs G1
  1. F3 binding: configurations and the expected (config_id, fold_id) set come ONLY from the SHA-256-pinned F3
     file (f3_binding.load_bound_grid / expected_plan). A job is a BoundJob(config_id, fold): it has no family
     or branch field. Family, estimator, preprocessing branch and hyperparameters are derived from F3 in the parent
     AND independently re-derived in the guarded child, checked by config_sha256. Mismatch -> failure.
  2. AP validity: every OK result must pass ap_validation.validate_ap with the class counts computed in the child
     (finite float in [0, 1] if the day has positives; exactly AP_UNDEFINED_NO_POSITIVES if benign-only).
     The child payload must have exactly the expected keys. grid_status re-validates every OK record.
  3. Execution-state recovery:
     write-ahead INFLIGHT.json (exclusive create, fsync) before the child starts; the child writes its PID first;
     the ledger record is appended (fsync) BEFORE INFLIGHT.json is removed. On construction and before every run,
     a leftover INFLIGHT.json means the parent was terminated abruptly (or crashed after recording): the key gets a
     failure record INCOMPLETE_PARENT_TERMINATED (score None) unless a record exists, the pilot is STOPPED with
     RECOVERY_DECISION_REQUIRED, and a still-running orphan child is reported (never adopted, never killed
     automatically). A corrupt ledger line also stops the pilot. There is no resume or retry method: continuing
     requires an explicit, recorded recovery decision.
Real data and real models remain refused (PILOT_TRAINING_AUTHORIZED = False).
"""
import dataclasses
import json
import math
import os
import sys
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
for _p in (os.path.join(BRIEF, "PILOT_INTEGRATION_G1"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pilot_guarded_runner as G1  # noqa: E402
import f3_binding as B  # noqa: E402
import ap_validation as AP  # noqa: E402
import model_adapter_g2 as A  # noqa: E402

G, S, R, M = G1.G, G1.S, G1.R, G1.M
PilotStop = G1.PilotStop
VERSION = "G2-pilot-runner-1"
PILOT_TRAINING_AUTHORIZED = False
G1_SHA256 = "7fb2b82306e624f7744019afcb6a2fe3a6a067bc9ad4f5794e20245428e6a078"
if R.sha256_path(G1.__file__) != G1_SHA256:
    raise R.DependencyError("G1 pilot_guarded_runner.py differs from the reviewed version")

SPEC_KEYS = frozenset({"config_id", "config_sha256", "fold", "faults", "target", "source_paths", "allowlist_path",
                       "allowlist_sha256", "profile", "evidence_dir", "child_pid_path"})
PAYLOAD_KEYS = frozenset({"score", "n_pos", "n_neg", "child_pid", "stages", "artifact_sha256", "n_train", "n_val",
                          "val_day", "kappa_max", "source_verification", "config_id", "config_sha256", "family",
                          "branch", "model_standin"})
SCORE_FAULTS = {"neg": -0.1, "gt1": 1.5, "nan": math.nan, "inf": math.inf, "bool": True, "int": 1, "str": "0.5",
                "undef": dict(AP.AP_UNDEFINED), "undef_extra": dict(AP.AP_UNDEFINED, note="x")}


@dataclasses.dataclass(frozen=True)
class BoundJob:
    config_id: str
    fold: S.FoldSpec
    faults: tuple = ()   # test profiles only

    @property
    def fold_key(self):
        return self.fold.outer if self.fold.inner is None else f"{self.fold.outer}/{self.fold.inner}"


# ----------------------------------------------------------------------------- guarded child
def _protected_job_g2(spec):
    with open(spec["child_pid_path"], "x") as fh:        # first action: make an orphan detectable
        fh.write(str(os.getpid()))
    import numpy as np
    from sklearn.metrics import average_precision_score
    if set(spec) != SPEC_KEYS:
        raise B.BindingError(f"unexpected job spec keys {sorted(set(spec) ^ SPEC_KEYS)}")
    grid = B.load_bound_grid()                            # independent re-derivation in the child
    cfg = grid.get(spec["config_id"])
    if cfg is None or cfg.config_sha256 != spec["config_sha256"]:
        raise B.BindingError("config_id / config_sha256 do not match frozen F3")
    A.build_estimator(cfg)                                # constructor validation only (no fit)
    faults = dict(spec["faults"])
    t0, stages = time.perf_counter(), []

    def mark(name):
        stages.append((name, round(time.perf_counter() - t0, 3)))
        M.inject({k: v for k, v in faults.items() if k != "score_value" and k != "payload"}, name)

    fold = S.FoldSpec(*spec["fold"])
    con = R.load_contract_verified()
    al = R.C1.load_allowlist(spec["allowlist_path"])
    if al["_sha256"] != spec["allowlist_sha256"]:
        raise R.PreprocessingError("allow-list file changed since the runner was created")
    profile = G1._profile_from(spec)
    if profile == S.P1_PROFILE:
        raise PilotStop("NOT_AUTHORIZED: real P1 data is refused in the child as well")
    val_day = fold.inner if fold.inner is not None else fold.outer
    need = set(fold.training_days()) | {val_day}
    mark("load")
    by_day = {}
    for p in spec["source_paths"]:
        d = R.C1.provenance_from_hashes([R.C1.sha256_file(p)], al).days[0]
        if d in by_day:
            raise R.PreprocessingError(f"two source files map to day {d}")
        by_day[d] = p
    if not need <= set(by_day):
        raise R.PreprocessingError(f"source set lacks days {sorted(need - set(by_day))}")
    blocks = {d: S.load_day_block(by_day[d], al, con, spec["target"], profile) for d in S.DAYS if d in need}
    mark("assemble")
    tm = S.assemble_training_matrix([blocks[d] for d in fold.training_days()], fold, al, con, spec["target"], profile)
    val = blocks[val_day]
    del blocks
    mark("preprocess")
    art = S.fit_p1_final(tm, con, cfg.branch, al, fold, spec["evidence_dir"], profile)   # branch from F3
    Xtr, _ = R.transform(tm.X, art, con)
    Xva, _ = R.transform(val.X, art, con)
    mark("fit")
    G1.require_verified_artifact(art, tm, fold, al, con, profile)
    ytr = G1.binary_target(tm.labels, tm.attempted, spec["target"])
    yva = G1.binary_target(val.labels, val.attempted, spec["target"])
    model = M.train("SYNTH-CENTROID", Xtr, ytr)           # synthetic stand-in; real training is not authorised
    mark("score")
    n_pos = int(yva.sum()); n_neg = int(yva.size - n_pos)
    if "score_value" in faults:
        ap = SCORE_FAULTS[faults["score_value"]]
    elif n_pos == 0:
        ap = dict(AP.AP_UNDEFINED)
    else:
        ap = float(average_precision_score(yva, M.score(model, Xva)))
    out = dict(score=ap, n_pos=n_pos, n_neg=n_neg, child_pid=os.getpid(), stages=stages,
               artifact_sha256=R.REF.artifact_sha256(art), n_train=int(tm.X.shape[0]), n_val=int(val.X.shape[0]),
               val_day=val_day, kappa_max=(R.kappa_record(art)["max_kappa"] if cfg.branch == "LR" else None),
               source_verification=art["source_verification"], config_id=cfg.config_id,
               config_sha256=cfg.config_sha256, family=cfg.family, branch=cfg.branch, model_standin="SYNTH-CENTROID")
    if faults.get("payload") == "extra_key":
        out["unexpected"] = 1
    elif faults.get("payload") == "branch_swap":
        out["branch"] = "LR" if cfg.branch == "TREE" else "TREE"
    elif faults.get("payload") == "no_counts":
        out.pop("n_pos")
    return out


# ----------------------------------------------------------------------------- parent
def validate_payload(payload, cfg):
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
        got = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        return False, f"unexpected score payload keys {got}"
    for k in ("config_id", "config_sha256", "family", "branch"):
        if payload[k] != getattr(cfg, k):
            return False, f"payload {k} {payload[k]!r} does not match F3 ({getattr(cfg, k)!r})"
    return AP.validate_ap(payload["score"], payload["n_pos"], payload["n_neg"])


class PilotRunnerG2:
    def __init__(self, work_dir, allowlist_path, source_paths, profile, *, phase, selected=None, target="primary",
                 probe=None, limit_bytes=G.DEFAULT_LIMIT_BYTES, time_limit_s=G.DEFAULT_TIME_LIMIT_S, poll_s=G.DEFAULT_POLL_S):
        al = R.C1.load_allowlist(allowlist_path)
        real = profile == S.P1_PROFILE or al["_sha256"] == S.P1_PROFILE.allowlist_sha256
        if real and not PILOT_TRAINING_AUTHORIZED:
            raise PilotStop("NOT_AUTHORIZED: Stage 02A-1 pilot training is not authorised (P1 profile / P1 allow-list refused)")
        S.check_profile(profile, al)
        if target not in S.TARGETS:
            raise PilotStop(f"unknown target {target!r}")
        self.grid = B.load_bound_grid()
        self.phase, self.selected = phase, selected
        self.expected = B.expected_plan(self.grid, phase, selected)
        self.work_dir = os.path.abspath(work_dir)
        self.allowlist_path, self.allowlist_sha256 = os.path.abspath(allowlist_path), al["_sha256"]
        self.source_paths = tuple(os.path.abspath(p) for p in source_paths)
        self.profile, self.target = profile, target
        self.probe, self.limit_bytes, self.time_limit_s, self.poll_s = probe, limit_bytes, time_limit_s, poll_s
        j = lambda n: os.path.join(self.work_dir, n)
        self.lock_path, self.ledger_path, self.stop_path = j(".pilot_fit.lock"), j("pilot_ledger.jsonl"), j("PILOT_STOPPED.json")
        self.inflight_path, self.child_pid_path = j("INFLIGHT.json"), j("INFLIGHT_CHILD.pid")
        os.makedirs(j("evidence"), exist_ok=True)
        self.recovery = self.check_recovery()

    # -- durable state
    def records(self):
        if not os.path.exists(self.ledger_path):
            return []
        out = []
        with open(self.ledger_path) as fh:
            for n, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    self._stop("LEDGER_CORRUPT", dict(line=n))
                    raise PilotStop(f"LEDGER_CORRUPT: ledger line {n} is not a complete record; recovery decision required")
        return out

    def _append(self, rec):
        with open(self.ledger_path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush(); os.fsync(fh.fileno())

    @property
    def stopped(self):
        return os.path.exists(self.stop_path)

    def stop_reason(self):
        if not self.stopped:
            return None
        with open(self.stop_path) as fh:
            return json.load(fh)["reason"]

    def _stop(self, reason, detail=None):
        if not self.stopped:
            with open(self.stop_path, "x") as fh:
                json.dump(dict(reason=reason, detail=detail, utc=R._utc()), fh, indent=1, default=str)
                fh.flush(); os.fsync(fh.fileno())
        return reason

    def _orphan(self, intent):
        if not os.path.exists(self.child_pid_path):
            return dict(child_pid=None, alive=False, note="child never started (no PID file)")
        try:
            pid = int(open(self.child_pid_path).read().strip() or "0")
        except (OSError, ValueError):
            return dict(child_pid=None, alive=None, note="unreadable child PID file")
        try:
            p = psutil.Process(pid)
            alive = p.status() != psutil.STATUS_ZOMBIE and p.create_time() >= intent["started_epoch"] - 2.0
        except psutil.Error:
            alive = False
        return dict(child_pid=pid, alive=bool(alive))

    def check_recovery(self):
        """Detect an incomplete fit left by an abrupt parent termination. Never resumes, retries or adopts."""
        recs = self.records()
        if not os.path.exists(self.inflight_path):
            return dict(state="CLEAN" if not self.stopped else "STOPPED", reason=self.stop_reason())
        try:
            with open(self.inflight_path) as fh:
                intent = json.load(fh)
        except (OSError, json.JSONDecodeError):
            intent = dict(key=None, started_epoch=0.0, note="unreadable INFLIGHT.json")
        orphan = self._orphan(intent)
        key = tuple(intent["key"]) if intent.get("key") else None
        recorded = key is not None and any((r["model_id"], r["fold"]) == key for r in recs)
        if key is not None and not recorded:
            self._append(dict(model_id=key[0], fold=key[1], status="FIT_ERROR", failure_class="INCOMPLETE_PARENT_TERMINATED",
                              score=None, detail=None, error="parent terminated before the fit was recorded; result unknown",
                              orphan=orphan, intent=intent, version=VERSION, recovered_utc=R._utc()))
        self._stop("RECOVERY_DECISION_REQUIRED", dict(intent=intent, orphan=orphan, record_existed=recorded))
        return dict(state="INCOMPLETE_FIT_DETECTED", key=key, record_existed=recorded, orphan=orphan)

    def _preflight(self):
        if self.probe is None:
            return G.preflight(self.work_dir, self.lock_path)
        disk, avail = self.probe.pop(0)
        return G.preflight(self.work_dir, self.lock_path, _disk_free=disk, _available=avail)

    def grid_status(self):
        recs = self.records()
        d = G.grid_decision(recs, self.expected)
        bad = []
        for r in recs:
            if r.get("status") == "OK":
                cfg = self.grid.get(r.get("model_id"))
                det = r.get("detail") or {}
                ok, why = (False, "unknown config") if cfg is None else validate_payload(dict(det, score=r.get("score")), cfg)
                if not ok:
                    bad.append(((r.get("model_id"), r.get("fold")), why))
        if bad and d["decision"] != "INVALID":
            d = dict(d, decision="INVALID", reason="OK record with an invalid AP or F3 binding", invalid_ap_records=bad)
        return d

    # -- one protected fit
    def run(self, job):
        if not isinstance(job, BoundJob):
            raise PilotStop("a BoundJob(config_id, fold) is required: model family and branch come only from frozen F3")
        self.check_recovery()
        if self.stopped:
            raise PilotStop(f"PILOT_STOPPED ({self.stop_reason()}): a separate recorded decision is required")
        key = (job.config_id, job.fold_key)
        if key not in self.expected:
            raise PilotStop(f"UNEXPECTED_KEY {key}: not in the F3/{self.phase} expected key set")
        if any((r["model_id"], r["fold"]) == key for r in self.records()):
            raise PilotStop(f"DUPLICATE_KEY {key}: already recorded; no retry")
        cfg = self.grid[job.config_id]
        ev = os.path.join(self.work_dir, "evidence", f"{job.config_id}__{job.fold_key.replace('/', '-')}")
        os.mkdir(ev)
        pf1 = self._preflight()
        if not pf1["ok"]:
            raise PilotStop(self._stop("PREFLIGHT_FAIL", pf1))
        if not G.acquire_fit_lock(self.lock_path):
            raise PilotStop(self._stop("CONCURRENT_FIT", dict(lock_path=self.lock_path)))
        rec = None
        try:
            pf2 = self._preflight()
            c = pf2["checks"]
            own = c["concurrent_fit"] and c["lock_holder"] == os.getpid()
            if not (c["free_disk_ok"] and c["available_mem_ok"] and (c["concurrent_ok"] or own)):
                raise PilotStop(self._stop("PREFLIGHT_UNDER_LOCK_FAIL", pf2))
            with open(self.inflight_path, "x") as fh:     # write-ahead intent
                json.dump(dict(key=list(key), config_sha256=cfg.config_sha256, parent_pid=os.getpid(),
                               started_epoch=time.time(), utc=R._utc()), fh)
                fh.flush(); os.fsync(fh.fileno())
            spec = dict(config_id=cfg.config_id, config_sha256=cfg.config_sha256, fold=(job.fold.outer, job.fold.inner),
                        faults=list(job.faults), target=self.target, source_paths=list(self.source_paths),
                        allowlist_path=self.allowlist_path, allowlist_sha256=self.allowlist_sha256,
                        profile=dataclasses.asdict(self.profile), evidence_dir=ev, child_pid_path=self.child_pid_path)
            rss0 = G1._parent_rss()
            g = G.run_guarded(_protected_job_g2, (spec,), cfg.config_id, job.fold_key, limit_bytes=self.limit_bytes,
                              time_limit_s=self.time_limit_s, poll_s=self.poll_s)
            rss1 = G1._parent_rss()
            payload = g.pop("score")
            rec = dict(g, version=VERSION, config_sha256=cfg.config_sha256, family=cfg.family, branch=cfg.branch,
                       target=self.target, parent_rss_before=int(rss0), parent_rss_after=int(rss1),
                       preflight_1=pf1["checks"], preflight_2=c, score=None, detail=None, failure_class=None)
            if rec["status"] == "OK":
                pf3 = self._preflight()
                ok, why = validate_payload(payload, cfg)
                if not pf3["checks"]["free_disk_ok"]:
                    rec.update(status="RESOURCE_EXCEEDED", failure_class="DISK", error=f"free disk after fit {pf3['checks']['free_disk_bytes']}")
                elif rss1 - rss0 > G1.PARENT_MARGIN_BYTES:
                    rec.update(status="RESOURCE_EXCEEDED", failure_class="PARENT_MEMORY_UNACCOUNTED",
                               error=f"parent RSS grew by {rss1 - rss0} bytes")
                elif not ok:
                    rec.update(status="FIT_ERROR", failure_class="INVALID_SCORE", error=why)
                else:
                    rec["score"] = payload.pop("score")
                    rec["detail"] = payload
            else:
                err = rec.get("error") or ""
                rec["failure_class"] = ("BINDING" if err.startswith("BindingError") or err.startswith("AdapterError")
                                        else G1._failure_class(rec))
        except PilotStop:
            raise
        except BaseException as e:
            self._stop("WRAPPER_ERROR", f"{type(e).__name__}: {e}")
            raise
        finally:
            G.release_fit_lock(self.lock_path)
        if rec["status"] != "OK":
            rec["score"] = None
            self._stop(f"FIT_FAILURE_{rec['failure_class']}", dict(key=key, error=rec.get("error")))
        self._append(rec)                                  # durable record first ...
        for p in (self.child_pid_path, self.inflight_path):  # ... then clear the intent
            if os.path.exists(p):
                os.remove(p)
        return rec
