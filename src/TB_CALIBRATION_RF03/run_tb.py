#!/usr/bin/env python3
"""T-b inner fits for the flat-selected RF-03 on real P1: the 20 frozen keys (outer k, inner j != k), sequential.

  --check                        validate (guard pin, v20 policy pin, environment, F3, P1 allow-list + CSV hashes,
                                 20-key plan, selection = RF-03 from the verified close-out). Writes nothing.
  --execute --authorize TOKEN    run / resume. Token = GO-TB-RF03-P1-<first 12 hex of the v20 policy SHA-256>.

Reuses the Flat-CV validation, guard (rg8_guard V21), v20 policy RG8-UNIFORM-8GB-v1 and checkpoint semantics unchanged:
LEDGER.jsonl (one record per finished key, never refitted), INTENT.json, STOPPED.json (no retry), exit 2 = low-memory
pause (no fit started; rerun resumes). No new fit is needed for the outer models: T-b applies tau_k to the stored
flat RF-03 outer scores (CV v2 section 4 step 3; TB_OOF_SCORE_EXPORT_SPEC section 1: identical fit, re-read bytes).
Order: outer Mon first (the Monday FPR deliverable), then Tue, Wed, Thu, Fri; inner days in D order.
"""
import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
FLAT = os.path.join(BRIEF, "FLAT_CV_P1")
for _p in (os.path.join(BRIEF, "RG8_V21"), FLAT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import psutil  # noqa: E402
import run_flat_cv as F  # noqa: E402  (validation and guard wiring; nothing in FLAT_CV_P1 is written)
import tb_job as J  # noqa: E402

RG, S, RR, B = F.RG, F.S, F.RR, F.B
GIB = F.GIB
SELECTED = J.SELECTED
CLOSEOUT = os.path.join(BRIEF, "P1_FLAT_CV_CLOSEOUT", "P1_FLAT_CV_VERIFICATION.json")
TMP_DIR = os.path.expanduser("~/Desktop/IDS_RAW_DATA_STAGE02A0/work/tb_rf03_tmp")
RUN_DIR = os.path.join(HERE, "tb_run")
LEDGER, INTENT, STOPPED = (os.path.join(RUN_DIR, f) for f in ("LEDGER.jsonl", "INTENT.json", "STOPPED.json"))
SCORES, EVIDENCE = os.path.join(RUN_DIR, "scores"), os.path.join(RUN_DIR, "evidence")
LOCK = os.path.join(RUN_DIR, ".fit.lock")
OUTER_ORDER = ("Mon", "Tue", "Wed", "Thu", "Fri")
utc = F.utc


class Refusal(Exception):
    pass


def _event(kind, **kw):
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(os.path.join(RUN_DIR, "RUN_EVENTS.jsonl"), "a") as fh:
        fh.write(json.dumps(dict(utc=utc(), event=kind, pid=os.getpid(), **kw), default=str) + "\n")
        fh.flush(); os.fsync(fh.fileno())


def plan(grid):
    keys = B.expected_plan(grid, "tb", SELECTED)
    order = [(SELECTED, o, i) for o in OUTER_ORDER for i in S.DAYS if i != o]
    if {(c, f"{o}/{i}") for c, o, i in order} != set(keys) or len(order) != 20:
        raise Refusal("execution order does not cover exactly the 20 frozen T-b keys")
    return order


def validate():
    pol, grid, _flat_order, base, rep = F.validate()
    v = json.load(open(CLOSEOUT))
    if v["status"] != "PASS" or v["selected"] != SELECTED or v["selected_config_sha256"] != grid[SELECTED].config_sha256:
        raise Refusal("P1 Flat-CV close-out does not certify RF-03 as the selected configuration")
    rep["selection"] = dict(selected=SELECTED, config_sha256=grid[SELECTED].config_sha256, closeout_ledger_sha256=v["ledger_sha256"])
    return pol, grid, plan(grid), base, rep


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
                           note="No retry. A separate recorded decision is required before any further T-b fit."))


def _no_live_worker():
    if RR.C.W.child_lock_busy(LOCK):
        raise Refusal("a guarded worker holds the T-b fit lock; nothing started")
    me = psutil.Process()
    own = {me.pid} | {x.pid for x in me.parents()} | {x.pid for x in me.children(recursive=True)}
    for q in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(q.info["cmdline"] or [])
        except (psutil.Error, TypeError):
            continue
        if q.info["pid"] not in own and any(s in cl for s in ("run_flat_cv.py", "p1_cv_job", "run_tb.py")) and "--check" not in cl:
            raise Refusal(f"another CV/T-b process is running (PID {q.info['pid']}); nothing started")


RECOVERY_PREFIX = "GO-RECOVER-TB-"


def recovery_path(key):
    return os.path.join(RUN_DIR, f"RECOVERY_{key[1]}__{key[2]}.json")


def recovery_token(key, stopped_bytes):
    return f"{RECOVERY_PREFIX}{key[1].upper()}-{key[2].upper()}-" + hashlib.sha256(stopped_bytes).hexdigest()[:12]


def authorize_recovery(token):
    """One-time owner-authorised attempt 2 of the single failed T-b key. Attempt-1 files are moved (not deleted) into
    scores/attempt1_orphans/ with their SHA-256 recorded; the original ledger line and STOPPED record are preserved."""
    done = [f for f in os.listdir(RUN_DIR) if f.startswith("RECOVERY_")]
    if done and not os.path.exists(STOPPED):                 # already authorised earlier (resume after a pause)
        rec = json.load(open(os.path.join(RUN_DIR, done[0])))
        if rec["token"] != token:
            raise Refusal("recovery token differs from the recorded one-time recovery")
        return tuple(rec["key"])
    if done:
        raise Refusal("a recovery was already used and T-b is STOPPED again; no further recovery is authorised")
    if not os.path.exists(STOPPED):
        raise Refusal("no STOPPED record: nothing to recover")
    raw = open(STOPPED, "rb").read()
    st = json.loads(raw)
    key = tuple(st["detail"]["key"])
    if st["reason"] != "FIT_FAILURE_FIT_ERROR" or st["detail"].get("failure_class") != "MEASUREMENT_UNAVAILABLE":
        raise Refusal(f"STOPPED is {st['reason']}/{st['detail'].get('failure_class')}: not the recoverable sampler race")
    if token != recovery_token(key, raw):
        raise Refusal("recovery token does not match this STOPPED record; nothing changed")
    lines = [l for l in open(LEDGER).read().splitlines() if l.strip()]
    hits = [(i, l) for i, l in enumerate(lines) if tuple(json.loads(l)["key"]) == key]
    if len(hits) != 1 or json.loads(hits[0][1])["status"] == "OK":
        raise Refusal(f"{key}: expected exactly one failed ledger record, found {len(hits)}")
    _no_live_worker()
    orph = os.path.join(SCORES, "attempt1_orphans")
    os.makedirs(orph, exist_ok=False)
    moved = {}
    for f in J.score_files(key[1], key[2]).values():
        src = os.path.join(SCORES, f)
        if os.path.exists(src):
            moved[f] = RG.sha256_file(src)
            os.rename(src, os.path.join(orph, f))
    archived = os.path.join(RUN_DIR, f"STOPPED_{key[1]}__{key[2]}__attempt1.json")
    os.link(STOPPED, archived)
    _write_x(recovery_path(key), dict(utc=utc(), key=list(key), token=token, attempt=2, attempt_id=uuid.uuid4().hex,
             authorised_by="owner: one-time T-b recovery after guard sampler race (MEASUREMENT_UNAVAILABLE at job_returned)",
             replaces=dict(ledger_line=hits[0][0] + 1, ledger_line_sha256=hashlib.sha256(hits[0][1].encode()).hexdigest()),
             stopped_archived_as=os.path.basename(archived), stopped_sha256=hashlib.sha256(raw).hexdigest(),
             attempt1_orphan_files_sha256=moved, attempt1_evidence_dir_kept=f"evidence/{key[0]}__outer={key[1]}__inner={key[2]}",
             rule="exactly one recovery attempt; any failure of attempt 2 stops T-b with no further retry"))
    os.remove(STOPPED)
    _event("recovery_authorised", key=list(key))
    return key


def check():
    pol, grid, order, base, rep = validate()
    done = {tuple(r["key"]) for r in ledger()}
    pf = RG.preflight(pol, BRIEF)
    rep.update(n_keys=len(order), done=len(done), stopped=os.path.exists(STOPPED), intent_pending=os.path.exists(INTENT),
               informational_preflight=pf, watched_processes_running=F.watched_running(),
               expected_counts={f"{o}/{i}": RR.expected_counts(S.P1_PROFILE, S.FoldSpec(o, i), "primary") for _, o, i in order},
               token_format=J.AUTH_PREFIX + "<first 12 hex of the v20 policy SHA-256>")
    print(json.dumps(rep, indent=1, default=str))
    print(f"\nCHECK: all validations passed. Informational preflight now: {'PASS' if pf['ok'] else 'FAIL'} "
          f"(available {pf['measurements'].get('available', 0) / GIB:.2f} GiB).")


def fit_one(pol, cfg, outer, inner, base, token, suffix=""):
    ev = os.path.join(EVIDENCE, f"{cfg.config_id}__outer={outer}__inner={inner}{suffix}")
    os.mkdir(ev)                                                  # fails if an earlier attempt left evidence: no retry
    os.makedirs(TMP_DIR, exist_ok=True)
    spec = dict(base, config_id=cfg.config_id, config_sha256=cfg.config_sha256, outer=outer, inner=inner, evidence_dir=ev,
                tmp_dir=TMP_DIR, score_dir=SCORES, authorization=token)
    g = RG.run_guarded_rg8(J.job_tb_inner, (spec,), policy=pol, work_dir=RUN_DIR, child_lock_path=LOCK,
                           sample_log_path=os.path.join(ev, "rg8_samples.jsonl"), stage_file=os.path.join(ev, "rg8_stages.jsonl"))
    payload = g.pop("payload")
    rec = dict(key=[cfg.config_id, outer, inner], status=g["status"], source="TB_INNER", guard=g, detail=None, checks_after_fit=[])
    if g["status"] == "OK":
        want = RR.expected_counts(S.P1_PROFILE, S.FoldSpec(outer, inner), "primary")
        bad = []
        if (payload["n_train"], payload["n_val"]) != want:
            bad.append(f"ROW_COUNT_MISMATCH {(payload['n_train'], payload['n_val'])} != {want}")
        if payload["n_neg"] == 0:
            bad.append("no benign rows on the inner day")
        for k, meta in payload["persisted"].items():
            if RG.sha256_file(os.path.join(SCORES, meta["file"])) != meta["sha256"]:
                bad.append(f"PERSISTED_HASH_MISMATCH {k}")
        rec.update(detail=payload, checks_after_fit=bad)
        if bad:
            rec["status"] = "FIT_ERROR"
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    return rec


def execute(token, recovery_tok=None):
    pol, grid, order, base, rep = validate()
    if token != J.expected_token(pol["_sha256"]):
        raise Refusal("authorization token does not match the v20 policy; nothing started")
    os.makedirs(SCORES, exist_ok=True); os.makedirs(EVIDENCE, exist_ok=True)
    if recovery_tok:
        authorize_recovery(recovery_tok)
    if os.path.exists(STOPPED):
        raise Refusal(f"T-b is STOPPED ({json.load(open(STOPPED))['reason']}); a recorded decision is required")
    if os.path.exists(INTENT):
        it = json.load(open(INTENT))
        if RR.C.W.child_lock_busy(LOCK):
            raise Refusal(f"a guarded worker still holds the fit lock for {it['key']}; nothing started")
        _append(dict(key=it["key"], status="INTERRUPTED", source="TB_INNER", utc=utc(), intent=it))
        _stop("INTERRUPTED", it)
        os.remove(INTENT)
        raise Refusal(f"fit {it['key']} was interrupted earlier; recorded as INTERRUPTED and T-b is STOPPED (no retry)")
    if os.path.isdir(TMP_DIR) and os.listdir(TMP_DIR):
        raise Refusal(f"{TMP_DIR} is not empty; nothing started")
    _no_live_worker()
    if not os.path.exists(os.path.join(RUN_DIR, "AUTHORIZATION.json")):
        _write_x(os.path.join(RUN_DIR, "AUTHORIZATION.json"), dict(token=token, utc=utc(), policy_sha256=pol["_sha256"],
                 authorised_by="owner: IDS CONFERENCE PAPER FAST-TRACK NEXT PHASE, Task 2 (RF-03 T-b source-only calibration)",
                 validation=rep, watched_processes_running=F.watched_running()))
    n_rec = {}
    for r in ledger():
        n_rec[tuple(r["key"])] = n_rec.get(tuple(r["key"]), 0) + 1
    done = {k for k, n in n_rec.items() if n >= (2 if os.path.exists(recovery_path(k)) else 1)}
    todo = [k for k in order if k not in done]
    print(f"T-b RF-03: {len(order) - len(todo)}/20 done, {len(todo)} to run (policy {pol['policy_id']}).", flush=True)
    cfg = grid[SELECTED]
    for n, (cid, outer, inner) in enumerate(todo, 1):
        pf = RG.preflight_with_wait(pol, RUN_DIR, lock_paths=(LOCK,))
        if not pf["ok"]:
            with open(os.path.join(RUN_DIR, "PREFLIGHT_PAUSES.jsonl"), "a") as fh:
                fh.write(json.dumps(dict(utc=utc(), next_key=[cid, outer, inner], preflight=pf), default=str) + "\n")
            print(f"PAUSED before {outer}/{inner}: preflight failed {pf['checks']} "
                  f"(available {pf['measurements'].get('available', 0) / GIB:.2f} GiB). No fit started.", flush=True)
            return 2
        rp = recovery_path((cid, outer, inner))
        rinfo = json.load(open(rp)) if os.path.exists(rp) else None
        attempt = rinfo["attempt"] if rinfo else 1
        attempt_id = rinfo["attempt_id"] if rinfo else uuid.uuid4().hex
        _write_x(INTENT, dict(key=[cid, outer, inner], utc=utc(), parent_pid=os.getpid(), config_sha256=cfg.config_sha256,
                              attempt=attempt, attempt_id=attempt_id))
        t = time.time()
        print(f"[{n}/{len(todo)}] {cid} outer={outer} inner={inner} start {utc()} "
              f"(available {pf['measurements']['available'] / GIB:.2f} GiB)", flush=True)
        try:
            rec = fit_one(pol, cfg, outer, inner, base, token, suffix="__attempt2" if rinfo else "")
        except BaseException as e:
            _append(dict(key=[cid, outer, inner], status="WRAPPER_ERROR", source="TB_INNER", utc=utc(),
                         error=f"{type(e).__name__}: {e}", attempt=attempt, attempt_id=attempt_id))
            _stop("WRAPPER_ERROR", dict(key=[cid, outer, inner], error=f"{type(e).__name__}: {e}"))
            os.remove(INTENT)
            raise
        rec.update(utc=utc(), preflight=pf, attempt=attempt, attempt_id=attempt_id,
                   recovery_of=(rinfo["replaces"] if rinfo else None))
        _append(rec)
        os.remove(INTENT)
        d = rec.get("detail") or {}
        print(f"    -> {rec['status']} inner AP={d.get('ap_inner')} peak={rec['guard']['peak_tree_mem'] / GIB:.2f} GiB "
              f"{time.time() - t:.0f}s", flush=True)
        if rec["status"] != "OK":
            _stop(f"FIT_FAILURE_{rec['status']}", dict(key=[cid, outer, inner], failure_class=rec["guard"].get("failure_class"),
                                                       error=rec["guard"].get("error"), checks=rec["checks_after_fit"]))
            print(f"STOPPED at {outer}/{inner}: {rec['status']}. No retry.", flush=True)
            return 3
    print(f"T-b inner fits COMPLETE: {sum(r['status'] == 'OK' for r in ledger())}/20 OK.", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--authorize", default=None)
    ap.add_argument("--recovery-token", default=None)
    a = ap.parse_args(argv)
    try:
        if a.check:
            check(); return 0
        if not a.authorize:
            raise Refusal("--execute requires --authorize TOKEN")
        signal.signal(signal.SIGTERM, F._on_sigterm)
        _event("start")
        rc = execute(a.authorize, a.recovery_token)
        _event("exit", rc=rc)
        return rc
    except (Refusal, F.Refusal) as e:
        if a.execute:
            _event("refused", reason=str(e))
        print("REFUSED:", e); return 1
    except (F.Terminated, KeyboardInterrupt) as e:
        _event("terminated", reason=f"{type(e).__name__}: {e}", during_fit_recorded_as_stop=os.path.exists(STOPPED))
        print(f"TERMINATED ({e}).", flush=True)
        return 4


if __name__ == "__main__":
    sys.exit(main())
