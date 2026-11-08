#!/usr/bin/env python3
"""Flat leave-one-day-out CV on real P1: the 60 frozen keys (12 F3 configs x 5 outer days), sequential, headless.

  --check                        validate everything without training (guard pin, v20 policy pin, environment, F3,
                                 P1 allow-list + SHA-256 of the five CSVs, the 60-key plan, A1 import, ledger state,
                                 one informational preflight). Writes nothing.
  --execute --authorize TOKEN    run / resume. Token = GO-FLATCV-P1-<first 12 hex of the v20 policy SHA-256>.

Unchanged: F3 grid, CV v2 folds, R3/R4/R5/F7 preprocessing (same route and calls as the A1 job), rg8_guard V21, and the
v20 uniform policy RG8-UNIFORM-8GB-v1 (start >= 3.0 GiB available, tree 4.75 GiB, swap growth 2.75 GiB, 1 h per fit).

Checkpoints (cv_run/):
  LEDGER.jsonl   one fsync'd record per finished key; a key in the ledger is never fitted again.
  INTENT.json    written before a fit starts, removed after its record is in the ledger.
  STOPPED.json   written on any non-OK fit, interruption or wrapper error. Its presence refuses every later run
                 (no retry; a separate recorded decision is required).
A preflight failure (memory not available within the policy wait) starts no fit, consumes no key and does not stop
the CV: the run exits with code 2 and the same command resumes later at the first unfinished key.
LR-01/Tue is imported from the A1 smoke run (same route; SHA-256 verified) instead of being refitted.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
V21 = os.path.join(BRIEF, "RG8_V21")
A1 = os.path.join(BRIEF, "RG8_V22_A1_SMOKE")
for _p in (V21, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import psutil  # noqa: E402
import rg8_runner as RR  # noqa: E402  (also puts the frozen G1/G2/G2-C/R5 modules on sys.path)
import p1_cv_job as J  # noqa: E402

RG, S, P2 = RR.RG, RR.S, RR.P2
R, B = P2.R, P2.B
AP = P2.AP
GIB = 1024 ** 3

V21_GUARD_SHA256 = "2a2f320234171965f4b736356ead5c858cbecb77dc31bd4161024f3a9fdf0211"
V20_POLICY_SHA256 = "3ba9d56dfd26df5f1e0e196e6bba2b9e081c5158e5a993faef917bcde6891380"
P1_DIR = os.path.expanduser("~/Desktop/IDS_RAW_DATA_STAGE02A0/P1_extracted")
P1_ALLOWLIST = os.path.join(BRIEF, "correction_package_C1", "provenance", "P1_SOURCE_ALLOWLIST.json")
TMP_DIR = os.path.expanduser("~/Desktop/IDS_RAW_DATA_STAGE02A0/work/flat_cv_tmp")
RUN_DIR = os.path.join(HERE, "cv_run")
LEDGER, INTENT, STOPPED = (os.path.join(RUN_DIR, f) for f in ("LEDGER.jsonl", "INTENT.json", "STOPPED.json"))
SCORES, EVIDENCE = os.path.join(RUN_DIR, "scores"), os.path.join(RUN_DIR, "evidence")
LOCK = os.path.join(RUN_DIR, ".fit.lock")
DAY_ORDER = ("Tue", "Wed", "Thu", "Fri", "Mon")          # attack days first; Mon (no positives) last
A1_RESULT = os.path.join(A1, "smoke_run", "RESULT.json")
A1_KEY = ("LR-01", "Tue")
WATCH = {"Claude", "Safari", "com.apple.WebKit.WebContent", "Google Chrome", "firefox", "Firefox", "V2Box", "PacketTunnel"}
utc = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Refusal(Exception):
    pass


class Terminated(BaseException):
    """SIGTERM received: raised so that the guard's cleanup runs and the reason is recorded (never silent)."""


def _on_sigterm(signum, frame):
    raise Terminated(f"signal {signal.Signals(signum).name}")


def _event(kind, **kw):
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(os.path.join(RUN_DIR, "RUN_EVENTS.jsonl"), "a") as fh:
        fh.write(json.dumps(dict(utc=utc(), event=kind, pid=os.getpid(), **kw), default=str) + "\n")
        fh.flush(); os.fsync(fh.fileno())


def watched_running():
    return sorted({p.info["name"] for p in psutil.process_iter(["name"]) if p.info["name"] in WATCH})


def plan(grid):
    keys = B.expected_plan(grid, "flat")
    order = [(c, d) for d in DAY_ORDER for c in grid]           # F3 order within each day
    if set(order) != set(keys) or len(order) != 60:
        raise Refusal("execution order does not cover exactly the 60 frozen flat keys")
    return order


def validate():
    rep = {}
    if RG.sha256_file(RG.__file__) != V21_GUARD_SHA256:
        raise Refusal("rg8_guard.py differs from the V21 hash")
    pol = RG.load_policy(RG.POLICY_PATH, expected_sha256=V20_POLICY_SHA256)
    rep["policy"] = dict(id=pol["policy_id"], sha256=pol["_sha256"], pinned=pol["_pinned"])
    env = RG.check_environment(pol)
    rep["environment"] = env
    if not env["ok"]:
        raise Refusal(f"ENVIRONMENT_MISMATCH {env}")
    grid = B.load_bound_grid()
    order = plan(grid)
    al = R.C1.load_allowlist(P1_ALLOWLIST)
    if al["_sha256"] != S.P1_PROFILE.allowlist_sha256:
        raise Refusal("P1 allow-list hash differs from the frozen P1 profile")
    paths = {}
    for fn in sorted(os.listdir(P1_DIR)):
        if not fn.endswith(".csv"):
            continue
        p = os.path.join(P1_DIR, fn)
        day = R.C1.provenance_from_hashes([R.C1.sha256_file(p)], al).days[0]      # rejects P2 / unknown hashes
        if day in paths:
            raise Refusal(f"two P1 files map to {day}")
        paths[day] = p
    if set(paths) != set(S.DAYS):
        raise Refusal(f"P1 files cover {sorted(paths)}, expected {S.DAYS}")
    rep["p1_files"] = {d: os.path.basename(p) for d, p in paths.items()}
    base = dict(paths=paths, allowlist_path=P1_ALLOWLIST, profile=dataclasses.asdict(S.P1_PROFILE), policy_sha256=pol["_sha256"])
    return pol, grid, order, base, rep


def ledger():
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _append(rec):
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush(); os.fsync(fh.fileno())


def _write_x(path, obj):
    with open(path, "x") as fh:
        json.dump(obj, fh, indent=1, default=str)
        fh.flush(); os.fsync(fh.fileno())


def _stop(reason, detail):
    _write_x(STOPPED, dict(utc=utc(), reason=reason, detail=detail,
                           note="No retry. A separate recorded decision is required before any further Flat-CV fit."))


def ap_status(n_pos, ap):
    return dict(AP.AP_UNDEFINED) if n_pos == 0 else dict(ap=ap, reason=None)


def a1_compatibility(res, pol, base, grid):
    """Scientific + artefact + resource compatibility of the A1 fit with a Flat-CV fit of LR-01/Tue."""
    job, g, v = res["job"], res["guard"], res["validation"]
    want = RR.expected_counts(S.P1_PROFILE, S.FoldSpec("Tue"), "primary")
    m, p = pol["monitor"], pol["preflight"]
    checks = {
        "config_sha256_F3": job["config_sha256"] == grid["LR-01"].config_sha256,
        "profile_P1": job["profile"] == S.P1_PROFILE.name,
        "p1_files_same_allowlisted_days": v["p1_files"] == {d: os.path.basename(x) for d, x in base["paths"].items()},
        "environment_identical": v["environment"]["got"] == RG.check_environment(pol)["got"],
        "row_counts_pinned": (job["n_train"], job["n_val"]) == want,
        "route_identical": True,   # p1_cv_job differs from p1_smoke_job only in scope/token/file names (+classes_ assert)
        "classes_attack_is_1": job["n_pos"] == 6933,
        "v20_start_available": res["preflight"]["measurements"]["available"] >= p["min_available_bytes"],
        "v20_start_pressure": res["preflight"]["measurements"]["pressure"] <= p["max_pressure_level"],
        "v20_tree_mem": g["peak_tree_mem"] <= m["tree_mem_limit_bytes"],
        "v20_swap_growth": g["peak_swap_growth"] <= m["swap_growth_limit_bytes"],
        "v20_time": g["elapsed_s"] <= m["time_limit_s"],
        "v20_min_disk": g["min_disk_free"] >= m["min_free_disk_bytes"],
        "no_survivors": g["survivors"] == [],
    }
    return checks


def a1_import_record(grid, pol=None, base=None):
    """LR-01/Tue from the A1 smoke fit: same frozen route, status OK, all three vectors SHA-256-verified."""
    res = json.load(open(A1_RESULT))
    job = res.get("job") or {}
    cfg = grid["LR-01"]
    if res.get("status") != "OK" or job.get("config_id") != "LR-01" or job.get("outer") != "Tue":
        raise Refusal("A1 RESULT.json is not an OK LR-01/Tue fit")
    if job["config_sha256"] != cfg.config_sha256:
        raise Refusal("A1 config_sha256 differs from frozen F3")
    if pol is not None:
        res["compatibility"] = c = a1_compatibility(res, pol, base, grid)
        if not all(c.values()):
            raise Refusal(f"A1 fit is not compatible with the Flat CV: {[k for k, ok in c.items() if not ok]}")
    src = os.path.join(A1, "smoke_run", "scores")
    for meta in job["persisted"].values():
        if RG.sha256_file(os.path.join(src, meta["file"])) != meta["sha256"]:
            raise Refusal(f"A1 vector {meta['file']} fails its SHA-256")
    return res, src


def import_a1(grid, pol, base):
    res, src = a1_import_record(grid, pol, base)
    job = res["job"]
    fn = J.score_files("LR-01", "Tue")
    persisted = {}
    for k, new in (("scores", fn["scores"]), ("labels", fn["labels"]), ("ids", fn["row_ids"])):
        meta = job["persisted"][k]
        dst = os.path.join(SCORES, new)
        shutil.copyfile(os.path.join(src, meta["file"]), dst)
        os.chmod(dst, 0o444)
        if RG.sha256_file(dst) != meta["sha256"]:
            raise Refusal(f"copied A1 vector {new} fails its SHA-256")
        persisted[k] = dict(meta, file=new)
    _append(dict(key=list(A1_KEY), status="OK", source="A1_SMOKE_IMPORT", utc=utc(),
                 note="Imported from RG8_V22_A1_SMOKE/smoke_run (policy RG8-A1-SMOKE-8GB-v1); identical route; not refitted",
                 compatibility=res["compatibility"], ap_status=ap_status(job["n_pos"], job["ap_heldout"]),
                 guard=res["guard"], detail=dict(job, persisted=persisted)))


def check():
    pol, grid, order, base, rep = validate()
    a1_import_record(grid, pol, base)
    done = {tuple(r["key"]) for r in ledger()}
    pf = RG.preflight(pol, BRIEF)
    rep.update(n_keys=len(order), done=len(done), remaining=len(order) - len(done) - (A1_KEY not in done),
               stopped=os.path.exists(STOPPED), intent_pending=os.path.exists(INTENT),
               informational_preflight=pf, watched_processes_running=watched_running(),
               token_format=J.AUTH_PREFIX + "<first 12 hex of the v20 policy SHA-256>")
    print(json.dumps(rep, indent=1, default=str))
    av = pf["measurements"].get("available", 0) / GIB
    print(f"\nCHECK: all validations passed. Informational preflight now: {'PASS' if pf['ok'] else 'FAIL'} "
          f"(available {av:.2f} GiB, required {pol['preflight']['min_available_bytes'] / GIB:.2f}).")
    if rep["watched_processes_running"]:
        print(f"Quit before --execute: {', '.join(rep['watched_processes_running'])}")


def fit_one(pol, cfg, outer, base, token, suffix=""):
    ev = os.path.join(EVIDENCE, f"{cfg.config_id}__{outer}{suffix}")
    os.mkdir(ev)                                                  # fails if an earlier attempt left evidence: no retry
    os.makedirs(TMP_DIR, exist_ok=True)
    spec = dict(base, config_id=cfg.config_id, config_sha256=cfg.config_sha256, outer=outer, evidence_dir=ev,
                tmp_dir=TMP_DIR, score_dir=SCORES, authorization=token)
    g = RG.run_guarded_rg8(J.job_flat_cv, (spec,), policy=pol, work_dir=RUN_DIR, child_lock_path=LOCK,
                           sample_log_path=os.path.join(ev, "rg8_samples.jsonl"), stage_file=os.path.join(ev, "rg8_stages.jsonl"))
    payload = g.pop("payload")
    rec = dict(key=[cfg.config_id, outer], status=g["status"], source="FLAT_CV", guard=g, detail=None, checks_after_fit=[])
    if g["status"] == "OK":
        want = RR.expected_counts(S.P1_PROFILE, S.FoldSpec(outer), "primary")
        bad = []
        if (payload["n_train"], payload["n_val"]) != want:
            bad.append(f"ROW_COUNT_MISMATCH {(payload['n_train'], payload['n_val'])} != {want}")
        ap = payload["ap_heldout"]
        if payload["n_pos"] > 0 and (ap is None or not (math.isfinite(ap) and 0.0 <= ap <= 1.0)):
            bad.append(f"INVALID_AP {ap}")
        if payload["n_pos"] == 0 and ap is not None:
            bad.append("AP defined on a day without positives")
        for k, meta in payload["persisted"].items():
            if RG.sha256_file(os.path.join(SCORES, meta["file"])) != meta["sha256"]:
                bad.append(f"PERSISTED_HASH_MISMATCH {k}")
        rec.update(detail=payload, checks_after_fit=bad, ap_status=ap_status(payload["n_pos"], ap))
        if bad:
            rec["status"] = "FIT_ERROR"
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    return rec


RECOVERY_PREFIX = "GO-RECOVER-"


def recovery_path(key):
    return os.path.join(RUN_DIR, f"RECOVERY_{key[0]}__{key[1]}.json")


def recovery_token(key, stopped_bytes):
    return f"{RECOVERY_PREFIX}{key[0].replace('-', '')}-{key[1].upper()}-" + hashlib.sha256(stopped_bytes).hexdigest()[:12]


def _no_live_worker():
    """No fit process may survive: the fit lock must be free and the PID last written into it must be gone."""
    if RR.C.W.child_lock_busy(LOCK):
        raise Refusal("a guarded worker holds the fit lock; nothing started")
    try:
        pid = int(open(LOCK).read().strip() or 0)
    except (OSError, ValueError):
        pid = 0
    if pid and psutil.pid_exists(pid):
        raise Refusal(f"the last fit worker PID {pid} is still alive; nothing started")
    me = psutil.Process()
    # own tree = this runner, its ancestors (launcher) and its descendants: on macOS `caffeinate -dims <utility>` runs
    # the utility in the original process and forks caffeinate itself as a CHILD whose cmdline contains run_flat_cv.py
    own = {me.pid} | {x.pid for x in me.parents()} | {x.pid for x in me.children(recursive=True)}
    for q in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(q.info["cmdline"] or [])
        except (psutil.Error, TypeError):
            continue
        if q.info["pid"] not in own and ("run_flat_cv.py" in cl or "p1_cv_job" in cl) and "--check" not in cl:
            raise Refusal(f"another Flat-CV process is running (PID {q.info['pid']}: {cl[:120]}); nothing started")


def authorize_recovery(spec, token):
    """One-time, owner-authorised recovery of ONE failed key (attempt 2). Idempotent across low-memory pauses."""
    cid, outer = spec.split(":")
    key = (cid, outer)
    rp = recovery_path(key)
    if os.path.exists(rp):                                     # already authorised earlier in this campaign
        rec = json.load(open(rp))
        if rec["token"] != token:
            raise Refusal("recovery token differs from the recorded one-time recovery")
        return key
    if not os.path.exists(STOPPED):
        raise Refusal("no STOPPED record: nothing to recover")
    raw = open(STOPPED, "rb").read()
    st = json.loads(raw)
    if tuple(st["detail"]["key"]) != key or st["reason"] not in ("WRAPPER_ERROR", "INTERRUPTED"):
        raise Refusal(f"STOPPED is {st['reason']} for {st['detail'].get('key')}, not a recoverable interruption of {key}")
    if token != recovery_token(key, raw):
        raise Refusal("recovery token does not match this STOPPED record; nothing changed")
    lines = [l for l in open(LEDGER).read().splitlines() if l.strip()]
    hits = [(i, l) for i, l in enumerate(lines) if tuple(json.loads(l)["key"]) == key]
    if len(hits) != 1 or json.loads(hits[0][1])["status"] == "OK":
        raise Refusal(f"{key}: expected exactly one failed ledger record, found {len(hits)}")
    fn = J.score_files(cid, outer)
    if any(os.path.exists(os.path.join(SCORES, f)) for f in fn.values()):
        raise Refusal(f"{key}: attempt-1 score files exist; refusing to reuse or overwrite them")
    _no_live_worker()
    archived = os.path.join(RUN_DIR, f"STOPPED_{cid}__{outer}__attempt1.json")
    os.link(STOPPED, archived)                                # keep the original stop record byte-for-byte
    _write_x(rp, dict(utc=utc(), key=list(key), token=token, attempt=2, attempt_id=uuid.uuid4().hex,
                      authorised_by="owner: IDS FAST-TRACK ONE-TIME RECOVERY AUTHORIZATION",
                      replaces=dict(ledger_line=hits[0][0] + 1, ledger_line_sha256=hashlib.sha256(hits[0][1].encode()).hexdigest(),
                                    status=json.loads(hits[0][1])["status"], error=json.loads(hits[0][1]).get("error")),
                      stopped_archived_as=os.path.basename(archived), stopped_sha256=hashlib.sha256(raw).hexdigest(),
                      attempt1_evidence_dir_kept=f"evidence/{cid}__{outer}",
                      attempt2_evidence_dir=f"evidence/{cid}__{outer}__attempt2",
                      rule="exactly one recovery attempt; any failure of attempt 2 stops the CV with no further retry"))
    os.remove(STOPPED)                                        # the archived hard link preserves it
    _event("recovery_authorised", key=list(key), recovery_file=os.path.basename(rp))
    return key


def execute(token, recover=None, recovery_tok=None):
    pol, grid, order, base, rep = validate()
    if token != J.expected_token(pol["_sha256"]):                 # before anything is created
        raise Refusal("authorization token does not match the v20 policy; nothing started")
    os.makedirs(SCORES, exist_ok=True); os.makedirs(EVIDENCE, exist_ok=True)
    if recover:
        authorize_recovery(recover, recovery_tok)
    if os.path.exists(STOPPED):
        raise Refusal(f"Flat CV is STOPPED ({json.load(open(STOPPED))['reason']}); a recorded decision is required")
    if os.path.exists(INTENT):
        it = json.load(open(INTENT))
        if RR.C.W.child_lock_busy(LOCK):
            raise Refusal(f"a guarded worker still holds the fit lock for {it['key']}; nothing started")
        _append(dict(key=it["key"], status="INTERRUPTED", source="FLAT_CV", utc=utc(), intent=it))
        _stop("INTERRUPTED", it)
        os.remove(INTENT)
        raise Refusal(f"fit {it['key']} was interrupted earlier; recorded as INTERRUPTED and the CV is STOPPED (no retry)")
    if os.path.isdir(TMP_DIR) and os.listdir(TMP_DIR):
        raise Refusal(f"{TMP_DIR} is not empty; nothing started")
    _no_live_worker()
    if not os.path.exists(os.path.join(RUN_DIR, "AUTHORIZATION.json")):
        _write_x(os.path.join(RUN_DIR, "AUTHORIZATION.json"), dict(token=token, utc=utc(), host=platform.node(),
                 policy_sha256=pol["_sha256"], validation=rep, watched_processes_running=watched_running()))
    recs = ledger()
    n_rec = {}
    for r in recs:
        n_rec[tuple(r["key"])] = n_rec.get(tuple(r["key"]), 0) + 1
    # a key is done once it has a record; a key with a one-time recovery is done once attempt 2 is recorded
    done = {k for k, n in n_rec.items() if n >= (2 if os.path.exists(recovery_path(k)) else 1)}
    if A1_KEY not in done:
        import_a1(grid, pol, base); done.add(A1_KEY)
    todo = [k for k in order if k not in done]
    print(f"Flat CV: {len(order) - len(todo)}/60 done, {len(todo)} to run (policy {pol['policy_id']}).", flush=True)
    for i, (cid, outer) in enumerate(todo, 1):
        cfg = grid[cid]
        pf = RG.preflight_with_wait(pol, RUN_DIR, lock_paths=(LOCK,))
        if not pf["ok"]:
            with open(os.path.join(RUN_DIR, "PREFLIGHT_PAUSES.jsonl"), "a") as fh:
                fh.write(json.dumps(dict(utc=utc(), next_key=[cid, outer], preflight=pf), default=str) + "\n")
            print(f"PAUSED before {cid}/{outer}: preflight failed {pf['checks']} "
                  f"(available {pf['measurements'].get('available', 0) / GIB:.2f} GiB). No fit started; rerun the same "
                  f"command to resume.", flush=True)
            return 2
        rp = recovery_path((cid, outer))
        rinfo = json.load(open(rp)) if os.path.exists(rp) else None
        attempt = rinfo["attempt"] if rinfo else 1
        attempt_id = rinfo["attempt_id"] if rinfo else uuid.uuid4().hex
        _write_x(INTENT, dict(key=[cid, outer], utc=utc(), parent_pid=os.getpid(), config_sha256=cfg.config_sha256,
                              attempt=attempt, attempt_id=attempt_id))
        t = time.time()
        print(f"[{i}/{len(todo)}] {cid} outer={outer} start {utc()} (available {pf['measurements']['available'] / GIB:.2f} GiB)", flush=True)
        try:
            rec = fit_one(pol, cfg, outer, base, token, suffix="__attempt2" if rinfo else "")
        except BaseException as e:
            _append(dict(key=[cid, outer], status="WRAPPER_ERROR", source="FLAT_CV", utc=utc(), error=f"{type(e).__name__}: {e}",
                         attempt=attempt, attempt_id=attempt_id, recovery_of=(rinfo["replaces"] if rinfo else None)))
            _stop("WRAPPER_ERROR", dict(key=[cid, outer], error=f"{type(e).__name__}: {e}"))
            os.remove(INTENT)
            raise
        rec.update(utc=utc(), preflight=pf, attempt=attempt, attempt_id=attempt_id,
                   recovery_of=(rinfo["replaces"] if rinfo else None))
        _append(rec)                                              # durable BEFORE the intent is cleared
        os.remove(INTENT)
        d = rec.get("detail") or {}
        print(f"    -> {rec['status']} AP={d.get('ap_heldout')} peak={rec['guard']['peak_tree_mem'] / GIB:.2f} GiB "
              f"swap+={rec['guard']['peak_swap_growth'] / GIB:.2f} GiB {time.time() - t:.0f}s", flush=True)
        if rec["status"] != "OK":
            _stop(f"FIT_FAILURE_{rec['status']}", dict(key=[cid, outer], failure_class=rec["guard"].get("failure_class"),
                                                       error=rec["guard"].get("error"), checks=rec["checks_after_fit"]))
            print(f"STOPPED at {cid}/{outer}: {rec['status']} {rec['guard'].get('failure_class')}. No retry.", flush=True)
            return 3
    n_ok = sum(r["status"] == "OK" for r in ledger())
    print(f"Flat CV COMPLETE: {n_ok}/60 OK. Ledger: {LEDGER}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--authorize", default=None)
    ap.add_argument("--recover", default=None, help="CONFIG:DAY, one-time owner-authorised attempt 2")
    ap.add_argument("--recovery-token", default=None)
    a = ap.parse_args(argv)
    try:
        if a.check:
            check(); return 0
        if not a.authorize:
            raise Refusal("--execute requires --authorize TOKEN")
        signal.signal(signal.SIGTERM, _on_sigterm)
        _event("start")
        rc = execute(a.authorize, a.recover, a.recovery_token)
        _event("exit", rc=rc)
        return rc
    except Refusal as e:
        if a.execute:
            _event("refused", reason=str(e))
        print("REFUSED:", e); return 1
    except (Terminated, KeyboardInterrupt) as e:
        fit = os.path.exists(STOPPED)
        why = f"{type(e).__name__}: {e}"
        _event("terminated", reason=why, during_fit_recorded_as_stop=fit)
        print(f"TERMINATED ({why}). " + ("A fit was running: recorded as WRAPPER_ERROR and the CV is STOPPED (no retry)."
              if fit else "No fit was running: nothing consumed; the same command resumes."), flush=True)
        return 4
    except Exception as e:
        _event("crash", reason=f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
