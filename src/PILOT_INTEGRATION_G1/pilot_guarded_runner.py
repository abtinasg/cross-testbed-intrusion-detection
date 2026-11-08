#!/usr/bin/env python3
"""G1: single fail-closed Stage 02A-1 pilot execution wrapper (PROPOSED; synthetic integration only).

One protected fit = one call to PilotRunner.run(JobSpec). The order is fixed:
  0. refusals before any work: pilot already stopped; key not in the exact expected key set (RG-8'); key
     already recorded (no retry and no duplicate); model family not authorised; evidence directory exists
  1. preflight #1 (guard v2.1: >= 10 GiB disk, >= 7.0 GiB available, no live lock)        -> else STOP
  2. acquire_fit_lock (O_CREAT|O_EXCL, atomic; a live holder refuses)                        -> else STOP
  3. try:
       preflight #2 while holding the lock; our own PID in the lock is NOT a competing fit      -> else STOP
       run_guarded(_protected_job): ONE child process, monitored by RSS (6.5 GiB) and time (3 h),
         performs ALL data work: source hashing, P1 DayBlock loading, TrainingMatrix assembly,
         fit_p1_final (R3-R5), transforms, verified-artefact gate, model fit, scoring
       post-checks: disk >= 10 GiB; parent RSS growth <= 0.5 GiB (= 7.0 - 6.5 GiB, the margin the
         preflight leaves for the parent, which holds no data); the score must be valid
     finally: release_fit_lock (always)
  4. append the record to the ledger (fsync). On ANY failure the score is None and the pilot is STOPPED:
     PILOT_STOPPED.json is written and every later run() is refused. There is no retry path.
Grid completion: grid_status() runs guard v2.1 grid_decision on the ledger (exact unique keys).

Real data: the only data path is P1 hashes -> load_day_block -> FoldSpec -> assemble_training_matrix ->
fit_p1_final. The model step accepts only an artefact that re-verifies against the TrainingMatrix
(require_verified_artifact), so an artefact made by calling r5_final.fit_core directly is rejected.
audit_no_bypass() checks statically that this module never calls fit_core or the older fit functions.
Stage 02A-1 training is NOT authorised: the P1 profile and the pinned P1 allow-list are refused, and only
synthetic model families exist.
"""
import ast
import dataclasses
import hashlib
import json
import math
import os
import sys
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
for _p in (os.path.join(BRIEF, "R5_FREEZE_F7"), os.path.join(BRIEF, "correction_package_C1", "guard"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import guard_v2_1 as G  # noqa: E402
import r5_final as R  # noqa: E402  (verifies the F7/F8 pins at import)
import r5_p1_source as S  # noqa: E402
import synthetic_models as M  # noqa: E402

VERSION = "G1-pilot-runner-1"
PILOT_TRAINING_AUTHORIZED = False   # Stage 02A-1 NOT authorised. Changing this needs a recorded decision.
AUTHORIZED_MODEL_FAMILIES = ()      # no real family authorised; synthetic families only under test profiles
GUARD_SHA256 = "00a0ccdd2fee812764e04b2307fde5400a909087d4184133f7a60fcdd0878eb2"
PARENT_MARGIN_BYTES = G.MIN_AVAILABLE_MEM_BYTES - G.DEFAULT_LIMIT_BYTES   # 0.5 GiB, derived (no new limit)

if R.sha256_path(G.__file__) != GUARD_SHA256:
    raise R.DependencyError("guard_v2_1.py differs from the pinned version")


class PilotStop(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class JobSpec:
    config_id: str
    fold: S.FoldSpec
    branch: str
    model_family: str
    faults: tuple = ()  # ((stage, kind), ...) test profiles only

    @property
    def fold_key(self):  # guard v2.1 key format: 'Wed' or 'Wed/Tue'
        return self.fold.outer if self.fold.inner is None else f"{self.fold.outer}/{self.fold.inner}"


# ----------------------------------------------------------------------------- child side (monitored scope)
def binary_target(labels, attempted, target):
    import numpy as np
    lab = np.array([l != "BENIGN" for l in labels])
    return (lab & (np.asarray(attempted) == -1)).astype(np.int64) if target == "primary" else lab.astype(np.int64)


def require_verified_artifact(art, tm, fold, allowlist, contract, profile):
    """Model-step gate: the preprocessing artefact must come from fit_p1_final on THIS verified matrix."""
    if not isinstance(art, dict) or "source_verification" not in art or "provenance" not in art:
        raise R.PreprocessingError("artefact has no source verification: not produced by fit_p1_final (bypass rejected)")
    prov, ver = S.verify_training_matrix(tm, fold, allowlist, contract, profile)
    if art["source_verification"] != ver or art["provenance"] != R.provenance_dict(prov):
        raise R.PreprocessingError("artefact source verification does not match the training matrix")
    if (art.get("r5_spec") != R.SPEC_ID or art.get("fold_id") != fold.fold_id or art.get("n_train") != tm.X.shape[0]
            or art.get("partition") != "P1:" + ",".join(fold.training_days())):
        raise R.PreprocessingError("artefact identity (spec, fold, rows, partition) does not match")
    _require_statistics_from(art, tm)
    return True


def _require_statistics_from(art, tm):
    """Bind the artefact's CONTENT to the verified rows: R3 caps, R4 vocabularies and (LR) R5 statistics and
    ZV-1 set are recomputed from tm.X with the frozen F7 column statistic and must match bitwise. This rejects
    an artefact produced on another matrix with verification fields copied in."""
    import numpy as np
    REF = R.REF
    pos = {c: j for j, c in enumerate(art["schema"])}
    for c in REF.INF_ALLOWED:
        col = np.asarray(tm.X[:, pos[c]]); fin = col[np.isfinite(col)]
        if art["r3"][c] != [REF._hx(fin.min()), REF._hx(fin.max())]:
            raise R.PreprocessingError(f"artefact R3 cap for {c!r} not computed from the verified rows")
    for c in REF.CATEGORICAL:
        if art["r4"][c] != [REF._hx(v) for v in np.unique(np.asarray(tm.X[:, pos[c]]))]:
            raise R.PreprocessingError(f"artefact R4 vocabulary for {c!r} not computed from the verified rows")
    if art["branch"] == "LR":
        mus, sds, zv = [], [], []
        for c in art["numeric"]:
            col = np.ascontiguousarray(tm.X[:, pos[c]], dtype=np.float64)
            if c in art["r3"]:
                col = R.C2._cap(col, art["r3"][c])
            mu, sd, const, _ = R.r5_column_statistics(REF.signed_log1p(col))
            mus.append(REF._hx(mu)); sds.append(REF._hx(sd))
            if const:
                zv.append(c)
        if art["r5"] != dict(mean=mus, std=sds) or art["zv1_constant_columns"] != zv:
            raise R.PreprocessingError("artefact R5 statistics not computed from the verified rows")


def _profile_from(spec):
    p = spec["profile"]
    prof = S.SourceProfile(p["name"], p["allowlist_sha256"], tuple(map(tuple, p["file_rows"])), tuple(map(tuple, p["attempted_rows"])))
    return S.P1_PROFILE if prof == S.P1_PROFILE else prof


def _protected_job(spec):
    """Runs inside the guarded child. Everything that touches data happens here."""
    import numpy as np
    from sklearn.metrics import average_precision_score
    faults = dict(spec["faults"])
    t0, stages = time.perf_counter(), []

    def mark(name):
        stages.append((name, round(time.perf_counter() - t0, 3)))
        M.inject(faults, name)

    fold = S.FoldSpec(*spec["fold"])
    con = R.load_contract_verified()
    al = R.C1.load_allowlist(spec["allowlist_path"])
    if al["_sha256"] != spec["allowlist_sha256"]:
        raise R.PreprocessingError("allow-list file changed since the runner was created")
    profile = _profile_from(spec)
    val_day = fold.inner if fold.inner is not None else fold.outer
    need = set(fold.training_days()) | {val_day}
    mark("load")
    by_day = {}
    for p in spec["source_paths"]:
        d = R.C1.provenance_from_hashes([R.C1.sha256_file(p)], al).days[0]   # P2 / unknown rejected here
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
    art = S.fit_p1_final(tm, con, spec["branch"], al, fold, spec["evidence_dir"], profile)
    Xtr, _ = R.transform(tm.X, art, con)
    Xva, _ = R.transform(val.X, art, con)
    mark("fit")
    require_verified_artifact(art, tm, fold, al, con, profile)
    ytr = binary_target(tm.labels, tm.attempted, spec["target"])
    yva = binary_target(val.labels, val.attempted, spec["target"])
    model = M.train(spec["model_family"], Xtr, ytr)
    mark("score")
    if faults.get("score") == "nan":
        ap = float("nan")
    elif yva.sum() == 0:
        ap = dict(ap=None, reason="AP_UNDEFINED_NO_POSITIVES")
    else:
        ap = float(average_precision_score(yva, M.score(model, Xva)))
    return dict(score=ap, child_pid=os.getpid(), stages=stages, artifact_sha256=R.REF.artifact_sha256(art),
                n_train=int(tm.X.shape[0]), n_val=int(val.X.shape[0]), val_day=val_day,
                kappa_max=(R.kappa_record(art)["max_kappa"] if spec["branch"] == "LR" else None),
                source_verification=art["source_verification"])


# ----------------------------------------------------------------------------- parent side
def _parent_rss():
    return psutil.Process().memory_info().rss


def _valid_score(s):
    if isinstance(s, bool):
        return False
    if isinstance(s, (int, float)):
        return math.isfinite(float(s))
    return s == dict(ap=None, reason="AP_UNDEFINED_NO_POSITIVES")


def _failure_class(rec):
    st, err = rec["status"], rec.get("error") or ""
    if st == "RESOURCE_EXCEEDED":
        return "MEMORY"
    if st == "TIME_EXCEEDED":
        return "TIMEOUT"
    if err.startswith("NumericalReviewRequired"):
        return "NUMERICAL_REVIEW"
    if err.startswith("PreprocessingError"):
        return "PROVENANCE_OR_INPUT"
    return "FIT"


class PilotRunner:
    def __init__(self, work_dir, allowlist_path, source_paths, profile, *, phase, config_ids, selected=None,
                 target="primary", probe=None, limit_bytes=G.DEFAULT_LIMIT_BYTES,
                 time_limit_s=G.DEFAULT_TIME_LIMIT_S, poll_s=G.DEFAULT_POLL_S):
        al = R.C1.load_allowlist(allowlist_path)
        real = profile == S.P1_PROFILE or al["_sha256"] == S.P1_PROFILE.allowlist_sha256
        if real and not PILOT_TRAINING_AUTHORIZED:
            raise PilotStop("NOT_AUTHORIZED: Stage 02A-1 pilot training is not authorised (P1 profile / P1 allow-list refused)")
        if real and (probe is not None or limit_bytes != G.DEFAULT_LIMIT_BYTES or time_limit_s != G.DEFAULT_TIME_LIMIT_S
                     or poll_s != G.DEFAULT_POLL_S):
            raise PilotStop("test-only knobs (probe, limits, poll) are refused for real data")
        S.check_profile(profile, al)
        if target not in S.TARGETS:
            raise PilotStop(f"unknown target {target!r}")
        self.work_dir, self.real = os.path.abspath(work_dir), real
        self.allowlist_path, self.allowlist_sha256 = os.path.abspath(allowlist_path), al["_sha256"]
        self.source_paths = tuple(os.path.abspath(p) for p in source_paths)   # fixed for the whole pilot
        self.profile, self.target = profile, target
        self.expected = G.expected_keys(phase, list(config_ids), selected)
        self.probe, self.limit_bytes, self.time_limit_s, self.poll_s = probe, limit_bytes, time_limit_s, poll_s
        self.lock_path = os.path.join(self.work_dir, ".pilot_fit.lock")
        self.ledger_path = os.path.join(self.work_dir, "pilot_ledger.jsonl")
        self.stop_path = os.path.join(self.work_dir, "PILOT_STOPPED.json")
        os.makedirs(os.path.join(self.work_dir, "evidence"), exist_ok=True)

    # -- ledger / stop state
    def records(self):
        if not os.path.exists(self.ledger_path):
            return []
        with open(self.ledger_path) as fh:
            return [json.loads(l) for l in fh if l.strip()]

    def _append(self, rec):
        with open(self.ledger_path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush(); os.fsync(fh.fileno())

    @property
    def stopped(self):
        return os.path.exists(self.stop_path)

    def _stop(self, reason, detail=None):
        if not self.stopped:
            with open(self.stop_path, "x") as fh:
                json.dump(dict(reason=reason, detail=detail, utc=R._utc()), fh, indent=1, default=str)
        return reason

    def _preflight(self):
        if self.probe is None:
            return G.preflight(self.work_dir, self.lock_path)
        disk, avail = self.probe.pop(0)
        return G.preflight(self.work_dir, self.lock_path, _disk_free=disk, _available=avail)

    def grid_status(self):
        return G.grid_decision(self.records(), self.expected)

    # -- one protected fit
    def run(self, job):
        if not isinstance(job, JobSpec):
            raise PilotStop("a JobSpec is required")
        key = (job.config_id, job.fold_key)
        if self.stopped:
            raise PilotStop("PILOT_STOPPED: an earlier failure stopped the pilot; a separate decision is required")
        if key not in self.expected:
            raise PilotStop(f"UNEXPECTED_KEY {key}: not in the exact expected key set")
        if any((r["model_id"], r["fold"]) == key for r in self.records()):
            raise PilotStop(f"DUPLICATE_KEY {key}: already recorded; no retry")
        if job.branch not in ("LR", "TREE"):
            raise PilotStop("branch must be LR or TREE")
        if self.real:
            if job.model_family not in AUTHORIZED_MODEL_FAMILIES or job.faults:
                raise PilotStop(f"model family {job.model_family!r} not authorised / faults refused")
        elif job.model_family not in M.SYNTHETIC_FAMILIES:
            raise PilotStop(f"model family {job.model_family!r}: only synthetic families exist at this stage")
        ev = os.path.join(self.work_dir, "evidence", f"{job.config_id}__{job.fold_key.replace('/', '-')}")
        os.mkdir(ev)                                      # FileExistsError if this key ran before: no retry
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
            spec = dict(fold=(job.fold.outer, job.fold.inner), branch=job.branch, model_family=job.model_family,
                        faults=list(job.faults), target=self.target, source_paths=list(self.source_paths),
                        allowlist_path=self.allowlist_path, allowlist_sha256=self.allowlist_sha256,
                        profile=dataclasses.asdict(self.profile), evidence_dir=ev)
            rss0 = _parent_rss()
            g = G.run_guarded(_protected_job, (spec,), job.config_id, job.fold_key, limit_bytes=self.limit_bytes,
                              time_limit_s=self.time_limit_s, poll_s=self.poll_s)
            rss1 = _parent_rss()
            payload = g.pop("score")
            rec = dict(g, version=VERSION, branch=job.branch, model_family=job.model_family, target=self.target,
                       parent_rss_before=int(rss0), parent_rss_after=int(rss1), preflight_1=pf1["checks"],
                       preflight_2=c, score=None, detail=None, failure_class=None)
            if rec["status"] == "OK":
                pf3 = self._preflight()
                if not pf3["checks"]["free_disk_ok"]:
                    rec.update(status="RESOURCE_EXCEEDED", failure_class="DISK", error=f"free disk after fit {pf3['checks']['free_disk_bytes']}")
                elif rss1 - rss0 > PARENT_MARGIN_BYTES:
                    rec.update(status="RESOURCE_EXCEEDED", failure_class="PARENT_MEMORY_UNACCOUNTED",
                               error=f"parent RSS grew by {rss1 - rss0} bytes (> {PARENT_MARGIN_BYTES})")
                elif not isinstance(payload, dict) or not _valid_score(payload.get("score")):
                    rec.update(status="FIT_ERROR", failure_class="INVALID_SCORE", error="missing or non-finite score")
                else:
                    rec["score"] = payload.pop("score")
                    rec["detail"] = payload
            else:
                rec["failure_class"] = _failure_class(rec)
        except PilotStop:
            raise
        except BaseException as e:                        # wrapper-level error: stop, never continue silently
            self._stop("WRAPPER_ERROR", f"{type(e).__name__}: {e}")
            raise
        finally:
            G.release_fit_lock(self.lock_path)
        if rec["status"] != "OK":
            rec["score"] = None                           # never a numerical score for a failed fit
            self._stop(f"FIT_FAILURE_{rec['failure_class']}", dict(key=key, error=rec.get("error")))
        self._append(rec)
        return rec


# ----------------------------------------------------------------------------- static bypass audit
FORBIDDEN_CALLS = {"fit_core", "fit_lean", "fit_dense", "fit_v2", "fit_lean_er5", "er5_mu_sd"}
FORBIDDEN_IMPORTS = {"er5", "r5_candidates", "prep_exact", "prep_reference", "prep_v0_2"}


def audit_no_bypass(path):
    """Static check: the module never calls the unverified fit functions and never imports the older
    preprocessing modules directly; it must reach preprocessing only through S.fit_p1_final."""
    with open(path) as fh:
        tree = ast.parse(fh.read())
    bad, uses_entry = [], False
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            if n.attr in FORBIDDEN_CALLS or (n.attr == "fit" and isinstance(n.value, ast.Name) and n.value.id in ("REF", "C2", "C1")):
                bad.append(f"line {n.lineno}: .{n.attr}")
            uses_entry |= n.attr == "fit_p1_final"
        elif isinstance(n, ast.Name) and n.id in FORBIDDEN_CALLS:
            bad.append(f"line {n.lineno}: {n.id}")
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in n.names] + ([n.module] if isinstance(n, ast.ImportFrom) and n.module else [])
            bad += [f"line {n.lineno}: import {m}" for m in names if m.split(".")[0] in FORBIDDEN_IMPORTS]
    return dict(path=os.path.basename(path), sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(),
                violations=bad, uses_fit_p1_final=uses_entry, ok=not bad and uses_entry)
