#!/usr/bin/env python3
"""P2 controlled execution: final RF-03 fit on P1, then label-blind scoring of the two P2 days, then the seal.

  --check                      validate (Flat-CV validation: guard pin, v20 policy, environment, F3, P1 files; P1 freeze
                               PASS; pre-registration; P2 CSV sizes). Writes nothing.
  --execute --authorize TOKEN  run / resume. TOKEN = GO-P2-RF03-d1a6585ac71b.
Keys in order: FINAL_FIT, SCORE/Wed14, SCORE/Fri16. Same checkpoint rules as the T-b runner (LEDGER, INTENT, STOPPED;
exit 2 = low-memory pause, nothing started). NO recovery path: any failure stops and is reported.
After all three keys are OK, P2_SCORES_SEALED.json (0444) hashes every output. Labels are read only by evaluate_p2.py,
which refuses to run without the seal.
"""
import argparse
import json
import os
import shutil
import signal
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.dirname(HERE)
BRIEF = os.path.dirname(FT)
for _p in (os.path.join(BRIEF, "RG8_V21"), os.path.join(BRIEF, "FLAT_CV_P1"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import psutil  # noqa: E402
import run_flat_cv as F  # noqa: E402
import p2_jobs as J  # noqa: E402
import hashlib  # noqa: E402
sys.path.insert(0, os.path.join(BRIEF, "RG8_V23_EXIT_RACE"))
import rg8_guard_v23 as G23  # noqa: E402  (V21 + verified-exit race rule only; used after an authorised recovery)
V23_GUARD_SHA256 = "0a57cf33dbbf5d90af2104bedbbf75ef0a08919f49e82aa01ec600faa0bf90a2"

RG, S, RR, B = F.RG, F.S, F.RR, F.B
GIB = F.GIB
P2_DIR = os.path.expanduser("~/Desktop/IDS_RAW_DATA_STAGE02A0/P2_extracted")
WORK = os.path.expanduser("~/Desktop/IDS_RAW_DATA_STAGE02A0/work/p2_run")
P2_FILES = {"Wed14": ("Wednesday-14-02-2018.csv", "4ceed4ba26973004e3a86ae0c57d6a4ddd9742f1b222f44209cd7c1c48618fdb", 5_898_350),
            "Fri16": ("Friday-16-02-2018.csv", "a16b76a454a25fa4fd84c320f32e3164b22c09f537be3e77450ddeb2168a7622", 7_390_266)}
RUN_DIR = os.path.join(HERE, "p2_run")
LEDGER, INTENT, STOPPED = (os.path.join(RUN_DIR, f) for f in ("LEDGER.jsonl", "INTENT.json", "STOPPED.json"))
MODEL_DIR, SCORES, EVIDENCE = (os.path.join(RUN_DIR, d) for d in ("final_model", "scores", "evidence"))
LOCK = os.path.join(RUN_DIR, ".fit.lock")
SEAL = os.path.join(HERE, "P2_SCORES_SEALED.json")
ORDER = [("FINAL_FIT",), ("SCORE", "Wed14"), ("SCORE", "Fri16")]
utc = F.utc


class Refusal(Exception):
    pass


def _event(kind, **kw):
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(os.path.join(RUN_DIR, "RUN_EVENTS.jsonl"), "a") as fh:
        fh.write(json.dumps(dict(utc=utc(), event=kind, pid=os.getpid(), **kw), default=str) + "\n")
        fh.flush(); os.fsync(fh.fileno())


def ledger():
    return [json.loads(l) for l in open(LEDGER) if l.strip()] if os.path.exists(LEDGER) else []


def _append(rec):
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n"); fh.flush(); os.fsync(fh.fileno())


def _write_x(path, obj):
    with open(path, "x") as fh:
        json.dump(obj, fh, indent=1, default=str); fh.flush(); os.fsync(fh.fileno())


def validate():
    pol, grid, _o, base, rep = F.validate()
    fz = json.load(open(os.path.join(FT, "P1_FREEZE_CHECK.json")))
    pre = json.load(open(os.path.join(FT, "P2_PREREGISTRATION.json")))
    if fz["status"] != "PASS" or pre["settings_sha256"] != J._sha(os.path.join(FT, "OPERATIONAL_SETTINGS_v1.json")) \
            or pre["final_thresholds_sha256"] != J._sha(os.path.join(FT, "FINAL_DETECTOR_TB_THRESHOLDS.json")):
        raise Refusal("P1 freeze / pre-registration / settings hashes are inconsistent")
    for d, (fn, h, n) in P2_FILES.items():
        p = os.path.join(P2_DIR, fn)
        if pre["p2_files"][d] != h or not os.path.exists(p):
            raise Refusal(f"P2 file {fn} missing or not pre-registered")
    rep["preregistration_sha256"] = J._sha(os.path.join(FT, "P2_PREREGISTRATION.json"))
    return pol, grid, base, rep


def _no_live_worker():
    if RR.C.W.child_lock_busy(LOCK):
        raise Refusal("a guarded worker holds the P2 lock; nothing started")
    me = psutil.Process()
    own = {me.pid} | {x.pid for x in me.parents()} | {x.pid for x in me.children(recursive=True)}
    for q in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(q.info["cmdline"] or [])
        except (psutil.Error, TypeError):
            continue
        if q.info["pid"] not in own and "python" in cl and any(s in cl for s in ("run_p2.py", "run_tb.py", "run_flat_cv.py")) and "--check" not in cl:
            raise Refusal(f"another runner is running (PID {q.info['pid']}); nothing started")


def _final_record():
    r = [x for x in ledger() if x["key"] == ["FINAL_FIT"] and x["status"] == "OK"]
    return r[0]["detail"] if r else None


RECOVERY_PREFIX = "GO-RECOVER-P2-"
RECOVERY = os.path.join(RUN_DIR, "RECOVERY_SCORE__Wed14.json")


def recovery_token(stopped_bytes):
    return RECOVERY_PREFIX + "WED14-" + hashlib.sha256(stopped_bytes).hexdigest()[:12]


def authorize_recovery(token):
    """One-time owner-authorised attempt 2 of SCORE/Wed14 only (MEASUREMENT_UNAVAILABLE at job_returned). Idempotent
    across low-memory pauses. Attempt-1 outputs (4 vectors + label-free parquet) are MOVED, never deleted or reused."""
    if os.path.exists(RECOVERY):
        rec = json.load(open(RECOVERY))
        if rec["token"] != token:
            raise Refusal("recovery token differs from the recorded one-time recovery")
        if os.path.exists(STOPPED):
            raise Refusal("the one-time recovery was used and P2 is STOPPED again; no further recovery is authorised")
        return
    if not os.path.exists(STOPPED):
        raise Refusal("no STOPPED record: nothing to recover")
    raw = open(STOPPED, "rb").read()
    st = json.loads(raw)
    det = st["detail"]
    if det["key"] != ["SCORE", "Wed14"] or det.get("failure_class") != "MEASUREMENT_UNAVAILABLE" or det.get("stage_at_end") != "job_returned":
        raise Refusal(f"STOPPED is {det}: not the recoverable SCORE/Wed14 exit race")
    if token != recovery_token(raw):
        raise Refusal("recovery token does not match this STOPPED record; nothing changed")
    if J._sha(G23.__file__) != V23_GUARD_SHA256 or J._sha(RG.__file__) != F.V21_GUARD_SHA256:
        raise Refusal("guard files differ from the pinned V21 / V23 hashes")
    lines = [l for l in open(LEDGER).read().splitlines() if l.strip()]
    hits = [(i, l) for i, l in enumerate(lines) if json.loads(l)["key"] == ["SCORE", "Wed14"]]
    fin = [json.loads(l) for l in lines if json.loads(l)["key"] == ["FINAL_FIT"] and json.loads(l)["status"] == "OK"]
    if len(hits) != 1 or json.loads(hits[0][1])["status"] == "OK" or len(fin) != 1:
        raise Refusal("ledger is not exactly: FINAL_FIT OK + one failed SCORE/Wed14")
    fd = fin[0]["detail"]
    if J._sha(os.path.join(MODEL_DIR, fd["model"]["file"])) != fd["model"]["sha256"] or \
            J._sha(os.path.join(MODEL_DIR, fd["artefact"]["file"])) != fd["artefact"]["sha256"]:
        raise Refusal("final model / artefact hash mismatch: not reusable")
    _no_live_worker()
    moved = {}
    od, ow = os.path.join(RUN_DIR, "attempt1_orphans"), os.path.join(WORK, "attempt1_orphans")
    os.makedirs(od, exist_ok=False); os.makedirs(ow, exist_ok=False)
    for f in sorted(os.listdir(SCORES)):
        if f.endswith("_P2_Wed14.npy"):
            moved[f] = J._sha(os.path.join(SCORES, f)); os.rename(os.path.join(SCORES, f), os.path.join(od, f))
    pqf = os.path.join(WORK, "P2_Wed14_label_free.parquet")
    if os.path.exists(pqf):
        moved[os.path.basename(pqf)] = J._sha(pqf); os.rename(pqf, os.path.join(ow, os.path.basename(pqf)))
    arch = os.path.join(RUN_DIR, "STOPPED_SCORE__Wed14__attempt1.json")
    os.link(STOPPED, arch)
    _write_x(RECOVERY, dict(utc=utc(), key=["SCORE", "Wed14"], token=token, attempt=2, attempt_id=uuid.uuid4().hex,
             authorised_by="owner: one-time P2 SCORE/Wed14 recovery (exit race at job_returned)",
             replaces=dict(ledger_line=hits[0][0] + 1, ledger_line_sha256=hashlib.sha256(hits[0][1].encode()).hexdigest()),
             stopped_archived_as=os.path.basename(arch), stopped_sha256=hashlib.sha256(raw).hexdigest(),
             attempt1_unverified_outputs_moved_sha256=moved, attempt1_evidence_dir_kept="evidence/SCORE__Wed14",
             final_fit_reused=dict(model=fd["model"], artefact=fd["artefact"]),
             guard_for_remaining_keys=dict(module="RG8_V23_EXIT_RACE/rg8_guard_v23.py", sha256=V23_GUARD_SHA256),
             rule="exactly one recovery attempt; any failure stops P2 with no further retry"))
    os.remove(STOPPED)
    _event("recovery_authorised", key=["SCORE", "Wed14"])


def run_key(pol, grid, base, key, suffix=""):
    tag = "__".join(key) + suffix
    ev = os.path.join(EVIDENCE, tag)
    os.mkdir(ev)
    tmp = os.path.join(WORK, "tmp")
    os.makedirs(tmp, exist_ok=True)
    if key == ("FINAL_FIT",):
        fn = J.job_final_fit
        spec = dict(base, config_sha256=grid[J.SEL].config_sha256, evidence_dir=ev, tmp_dir=tmp, model_dir=MODEL_DIR,
                    authorization=J.AUTH)
    else:
        fin = _final_record()
        d = key[1]
        fname, h, n = P2_FILES[d]
        fn = J.job_score_p2
        spec = dict(day=d, csv_path=os.path.join(P2_DIR, fname), csv_sha256=h, expected_rows=n,
                    model_path=os.path.join(MODEL_DIR, fin["model"]["file"]), model_sha256=fin["model"]["sha256"],
                    artefact_path=os.path.join(MODEL_DIR, fin["artefact"]["file"]), artefact_sha256=fin["artefact"]["sha256"],
                    parquet_path=os.path.join(WORK, f"P2_{d}_label_free.parquet"), score_dir=SCORES,
                    a1_dir=os.path.join(BRIEF, "amendment_A1"), authorization=J.AUTH)
    GM = G23 if os.path.exists(RECOVERY) else RG
    g = GM.run_guarded_rg8(fn, (spec,), policy=pol, work_dir=RUN_DIR, child_lock_path=LOCK,
                           sample_log_path=os.path.join(ev, "rg8_samples.jsonl"), stage_file=os.path.join(ev, "rg8_stages.jsonl"))
    payload = g.pop("payload")
    rec = dict(key=list(key), status=g["status"], guard=g, detail=payload if g["status"] == "OK" else None,
               guard_module=os.path.basename(GM.__file__))
    shutil.rmtree(tmp, ignore_errors=True)
    return rec


def seal():
    recs = {tuple(r["key"]): r for r in ledger() if r["status"] == "OK"}
    out = dict(record="P2 predictions sealed BEFORE any P2 label is read", utc=utc(),
               preregistration_sha256=J._sha(os.path.join(FT, "P2_PREREGISTRATION.json")),
               final_fit={k: recs[("FINAL_FIT",)]["detail"][k] for k in ("model", "artefact", "train_x_sha256", "n_train", "fold_id",
                                                                        "config_sha256", "model_columns")},
               days={}, ledger_sha256=J._sha(LEDGER),
               code_sha256={f: J._sha(os.path.join(HERE, f)) for f in ("p2_jobs.py", "run_p2.py")},
               recovery=(json.load(open(RECOVERY)) if os.path.exists(RECOVERY) else None),
               guard_by_key={"/".join(r["key"]): r.get("guard_module", "rg8_guard.py") for r in ledger() if r["status"] == "OK"})
    for d in P2_FILES:
        det = recs[("SCORE", d)]["detail"]
        for meta in det["persisted"].values():
            if J._sha(os.path.join(SCORES, meta["file"])) != meta["sha256"]:
                raise Refusal(f"{meta['file']} changed before sealing")
        out["days"][d] = dict(n_rows=det["n_rows"], csv_sha256=det["csv_sha256"], label_free_parquet=det["label_free_parquet"],
                              persisted=det["persisted"], label_columns_read=det["label_columns_read"])
    _write_x(SEAL, out)
    os.chmod(SEAL, 0o444)
    return J._sha(SEAL)


def execute(token, recovery_tok=None):
    pol, grid, base, rep = validate()
    if token != J.AUTH:
        raise Refusal("authorization token mismatch; nothing started")
    for d in (MODEL_DIR, SCORES, EVIDENCE):
        os.makedirs(d, exist_ok=True)
    if recovery_tok:
        authorize_recovery(recovery_tok)
    if os.path.exists(STOPPED):
        raise Refusal(f"P2 is STOPPED ({json.load(open(STOPPED))['reason']}); a recorded decision is required")
    if os.path.exists(INTENT):
        it = json.load(open(INTENT))
        _append(dict(key=it["key"], status="INTERRUPTED", utc=utc(), intent=it))
        _write_x(STOPPED, dict(utc=utc(), reason="INTERRUPTED", detail=it))
        os.remove(INTENT)
        raise Refusal(f"{it['key']} was interrupted earlier; recorded and STOPPED (no retry)")
    _no_live_worker()
    if not os.path.exists(os.path.join(RUN_DIR, "AUTHORIZATION.json")):
        _write_x(os.path.join(RUN_DIR, "AUTHORIZATION.json"), dict(token=token, utc=utc(), validation=rep,
                 authorised_by="owner: FINAL P1 CLOSEOUT AND P2 EXECUTION AUTHORIZATION (2026-11-07)"))
    n_rec = {}
    for r in ledger():
        n_rec[tuple(r["key"])] = n_rec.get(tuple(r["key"]), 0) + 1
    rk = ("SCORE", "Wed14")
    done = {k for k, c in n_rec.items() if c >= (2 if (k == rk and os.path.exists(RECOVERY)) else 1)}
    todo = [k for k in ORDER if k not in done]
    print(f"P2: {len(ORDER) - len(todo)}/3 done, {len(todo)} to run.", flush=True)
    for k in todo:
        pf = RG.preflight_with_wait(pol, RUN_DIR, lock_paths=(LOCK,))
        if not pf["ok"]:
            print(f"PAUSED before {k}: available {pf['measurements'].get('available', 0) / GIB:.2f} GiB. No job started.", flush=True)
            return 2
        _write_x(INTENT, dict(key=list(k), utc=utc(), attempt_id=uuid.uuid4().hex))
        t = time.time()
        print(f"{'/'.join(k)} start {utc()} (available {pf['measurements']['available'] / GIB:.2f} GiB)", flush=True)
        try:
            rec = run_key(pol, grid, base, k, suffix="__attempt2" if (k == rk and os.path.exists(RECOVERY)) else "")
        except BaseException as e:
            _append(dict(key=list(k), status="WRAPPER_ERROR", utc=utc(), error=f"{type(e).__name__}: {e}"))
            _write_x(STOPPED, dict(utc=utc(), reason="WRAPPER_ERROR", detail=dict(key=list(k), error=f"{type(e).__name__}: {e}")))
            os.remove(INTENT)
            raise
        rec.update(utc=utc(), preflight=pf, attempt=(2 if (k == rk and os.path.exists(RECOVERY)) else 1),
                   recovery_of=(json.load(open(RECOVERY))["replaces"] if (k == rk and os.path.exists(RECOVERY)) else None))
        _append(rec)
        os.remove(INTENT)
        g = rec["guard"]
        print(f"    -> {rec['status']} {g.get('failure_class') or ''} {g.get('error') or ''} peak={g['peak_tree_mem'] / GIB:.2f} GiB "
              f"{time.time() - t:.0f}s stage={g.get('stage_at_end')}", flush=True)
        if rec["status"] != "OK":
            _write_x(STOPPED, dict(utc=utc(), reason=f"JOB_FAILURE_{rec['status']}", detail=dict(key=list(k),
                     failure_class=g.get("failure_class"), error=g.get("error"), stage_at_end=g.get("stage_at_end"))))
            print("STOPPED. No retry and no recovery are authorised.", flush=True)
            return 3
    print("P2 scoring COMPLETE. Seal SHA-256:", seal(), flush=True)
    return 0


def check():
    pol, grid, base, rep = validate()
    pf = RG.preflight(pol, BRIEF)
    rep.update(done=[r["key"] for r in ledger()], stopped=os.path.exists(STOPPED), preflight_ok=pf["ok"],
               available_gib=pf["measurements"].get("available", 0) / GIB)
    print(json.dumps({k: rep[k] for k in ("preregistration_sha256", "done", "stopped", "preflight_ok", "available_gib")}, indent=1))
    print("CHECK: all validations passed.")


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
        _event("terminated", reason=str(e))
        print(f"TERMINATED ({e}).", flush=True); return 4


if __name__ == "__main__":
    sys.exit(main())
