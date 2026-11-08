#!/usr/bin/env python3
"""A1 guarded one-fit real-P1 smoke test (LR-01, outer Tue held out). See A1_AMENDMENT.md.

  --check                        validate everything without training (policy pin, guard pin, environment, F3 LR-01,
                                 P1 allow-list + SHA-256 of the five CSVs, expected row counts, one informational
                                 preflight reading). Reads P1 file bytes for hashing only; parses no rows. Writes nothing.
  --execute --authorize TOKEN    the single authorized run. Run from Terminal.app with the Claude desktop app, Safari,
                                 other browsers and V2Box QUIT:
      cd ".../STAGE02A_1_DECISION_BRIEF/RG8_V22_A1_SMOKE"
      <PINNED_INTERPRETER:anaconda3/bin/python3> -I -B run_p1_smoke.py --execute --authorize <TOKEN>

Write-once and no retry: --execute creates ./smoke_run/ with os.mkdir; if it already exists the run is refused. Every
outcome (PREFLIGHT_FAIL, any guard stop, FIT_ERROR, OK) is final and recorded in smoke_run/RESULT.json.
"""
import argparse
import dataclasses
import json
import math
import os
import platform
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
V21 = os.path.join(BRIEF, "RG8_V21")
for _p in (V21, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import psutil  # noqa: E402
import rg8_runner as RR  # noqa: E402  (also puts the frozen G1/G2/G2-C/R5 modules on sys.path)
import p1_smoke_job as J  # noqa: E402

RG, S, P2 = RR.RG, RR.S, RR.P2
R = P2.R
GIB = 1024 ** 3

A1_POLICY_PATH = os.path.join(HERE, "A1_SMOKE_POLICY.json")
A1_POLICY_SHA256 = "76406993e7d4d565837a5ed34785badabc4db926d4ac75e9508bcd80f49a21c3"
V21_GUARD_SHA256 = "2a2f320234171965f4b736356ead5c858cbecb77dc31bd4161024f3a9fdf0211"
V20_POLICY_SHA256 = "3ba9d56dfd26df5f1e0e196e6bba2b9e081c5158e5a993faef917bcde6891380"
P1_DIR = os.path.expanduser("~/Desktop/IDS_RAW_DATA_STAGE02A0/P1_extracted")
P1_ALLOWLIST = os.path.join(BRIEF, "correction_package_C1", "provenance", "P1_SOURCE_ALLOWLIST.json")
TMP_DIR = os.path.expanduser("~/Desktop/IDS_RAW_DATA_STAGE02A0/work/a1_smoke_tmp")
RUN_DIR = os.path.join(HERE, "smoke_run")
WATCH = {"Claude", "Safari", "com.apple.WebKit.WebContent", "Google Chrome", "firefox", "Firefox", "V2Box", "PacketTunnel"}


class Refusal(Exception):
    pass


def top_processes(n=15):
    rows = []
    for p in psutil.process_iter(["name", "memory_info"]):
        mi = p.info["memory_info"]
        rows.append((mi.rss if mi else 0, p.info["name"] or "?"))
    rows.sort(reverse=True)
    watch = sorted({nm for _, nm in rows if nm in WATCH})              # advisory record only; the memory gate decides
    return [dict(rss=r, name=nm) for r, nm in rows[:n]], watch


def validate():
    """Everything except training. Returns (policy, spec_base, report). Raises Refusal on any mismatch."""
    rep = {}
    if RG.sha256_file(RG.__file__) != V21_GUARD_SHA256:
        raise Refusal("rg8_guard.py differs from the V21 hash")
    if RG.sha256_file(RG.POLICY_PATH) != V20_POLICY_SHA256:
        raise Refusal("the v20 uniform policy file changed")
    pol = RG.load_policy(A1_POLICY_PATH, expected_sha256=A1_POLICY_SHA256)       # unchanged V21 loader + sanity checks
    rep["policy"] = dict(id=pol["policy_id"], sha256=pol["_sha256"], pinned=pol["_pinned"])
    env = RG.check_environment(pol)
    rep["environment"] = env
    if not env["ok"]:
        raise Refusal(f"ENVIRONMENT_MISMATCH {env}")
    cfg = P2.B.load_bound_grid()[J.CONFIG_ID]
    rep["f3"] = dict(config_id=cfg.config_id, config_sha256=cfg.config_sha256, branch=cfg.branch, family=cfg.family)
    if cfg.family != "LR":
        raise Refusal("LR-01 is not an LR configuration in frozen F3")
    al = R.C1.load_allowlist(P1_ALLOWLIST)
    if al["_sha256"] != S.P1_PROFILE.allowlist_sha256:
        raise Refusal("P1 allow-list hash differs from the frozen P1 profile")
    paths = {}
    for fn in sorted(os.listdir(P1_DIR)):
        if not fn.endswith(".csv"):
            continue
        p = os.path.join(P1_DIR, fn)
        h = R.C1.sha256_file(p)
        day = R.C1.provenance_from_hashes([h], al).days[0]          # rejects P2 / unknown hashes
        if day in paths:
            raise Refusal(f"two P1 files map to {day}")
        paths[day] = p
    if set(paths) != set(S.DAYS):
        raise Refusal(f"P1 files cover {sorted(paths)}, expected {S.DAYS}")
    rep["p1_files"] = {d: os.path.basename(p) for d, p in paths.items()}
    fold = S.FoldSpec(J.OUTER)
    rep["expected_rows"] = dict(zip(("n_train", "n_val"), RR.expected_counts(S.P1_PROFILE, fold, "primary")))
    spec = dict(config_id=cfg.config_id, config_sha256=cfg.config_sha256, outer=J.OUTER, paths=paths,
                allowlist_path=P1_ALLOWLIST, profile=dataclasses.asdict(S.P1_PROFILE), policy_sha256=pol["_sha256"])
    return pol, spec, rep


def check():
    pol, spec, rep = validate()
    pf = RG.preflight(pol, os.path.dirname(HERE))
    top, watch = top_processes()
    rep.update(informational_preflight=pf, watched_processes_running=watch, run_dir_exists=os.path.exists(RUN_DIR),
               token_format=J.AUTH_PREFIX + "<first 12 hex of the A1 policy SHA-256>")
    print(json.dumps(rep, indent=1, default=str))
    av = pf["measurements"].get("available", 0) / GIB
    print(f"\nCHECK: all validations passed. Informational preflight now: {'PASS' if pf['ok'] else 'FAIL'} "
          f"(available {av:.2f} GiB, required {pol['preflight']['min_available_bytes'] / GIB:.2f}).")
    if watch:
        print(f"Quit before --execute: {', '.join(watch)}")


def _write_x(path, obj):
    with open(path, "x") as fh:
        json.dump(obj, fh, indent=1, default=str)
        fh.flush(); os.fsync(fh.fileno())


def execute(token):
    if token != J.expected_token(A1_POLICY_SHA256):                      # before anything is read or created
        raise Refusal("authorization token does not match the A1 policy; nothing started")
    pol, spec, rep = validate()
    if os.path.isdir(TMP_DIR) and os.listdir(TMP_DIR):
        raise Refusal(f"{TMP_DIR} is not empty; nothing started")
    try:
        os.mkdir(RUN_DIR)                                              # write-once: any earlier attempt blocks this one
    except FileExistsError:
        raise Refusal(f"{RUN_DIR} exists: the single A1 run has already been attempted (no retry)")
    utc = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    top0, watch0 = top_processes()
    _write_x(os.path.join(RUN_DIR, "AUTHORIZATION.json"), dict(token=token, utc=utc(), host=platform.node(),
             policy_sha256=pol["_sha256"], validation=rep, top_processes_rss=top0, watched_processes_running=watch0))
    result = dict(protocol="A1 one-fit real-P1 smoke", utc_start=utc(), policy_id=pol["policy_id"], policy_sha256=pol["_sha256"],
                  validation=rep, watched_processes_running=watch0)
    lock = os.path.join(RUN_DIR, ".fit.lock")
    pf = RG.preflight_with_wait(pol, RUN_DIR, lock_paths=(lock,))
    result["preflight"] = pf
    if not pf["ok"]:
        result.update(status="PREFLIGHT_FAIL", decision="BLOCKED: no fit started; use the alternative in A1_AMENDMENT.md section 7")
        _write_x(os.path.join(RUN_DIR, "RESULT.json"), result)
        print(f"PREFLIGHT_FAIL after {pf.get('attempts')} attempts: {pf['checks']}  (no fit started; final)")
        return 2
    ev, sc = os.path.join(RUN_DIR, "evidence"), os.path.join(RUN_DIR, "scores")
    os.mkdir(ev); os.mkdir(sc)
    os.makedirs(TMP_DIR, exist_ok=True)
    job_spec = dict(spec, evidence_dir=ev, tmp_dir=TMP_DIR, score_dir=sc, authorization=token)
    print(f"preflight PASS (available {pf['measurements']['available'] / GIB:.2f} GiB); starting the single guarded fit ...", flush=True)
    g = RG.run_guarded_rg8(J.job_lr01_smoke, (job_spec,), policy=pol, work_dir=RUN_DIR, child_lock_path=lock,
                           sample_log_path=os.path.join(ev, "rg8_samples.jsonl"), stage_file=os.path.join(ev, "rg8_stages.jsonl"))
    payload = g.pop("payload")
    result["guard"] = g
    status = g["status"]
    if status == "OK":
        want = (rep["expected_rows"]["n_train"], rep["expected_rows"]["n_val"])
        bad = []
        if (payload["n_train"], payload["n_val"]) != want:
            bad.append(f"ROW_COUNT_MISMATCH {(payload['n_train'], payload['n_val'])} != {want}")
        ap = payload["ap_heldout"]
        if ap is None or not (math.isfinite(ap) and 0.0 <= ap <= 1.0):
            bad.append(f"INVALID_AP {ap}")
        for k, meta in payload["persisted"].items():
            if RG.sha256_file(os.path.join(sc, meta["file"])) != meta["sha256"]:
                bad.append(f"PERSISTED_HASH_MISMATCH {k}")
        if bad:
            status = "FIT_ERROR"
        result["checks_after_fit"] = bad
        result["job"] = payload
    result.update(status=status, utc_end=utc(),
                  decision=("SMOKE PASSED: resource evidence for the real-data route recorded" if status == "OK" else
                            f"SMOKE STOPPED ({status} {g.get('failure_class')}): no retry; see A1_AMENDMENT.md section 7"))
    _write_x(os.path.join(RUN_DIR, "RESULT.json"), result)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print(json.dumps(dict(status=status, failure_class=g.get("failure_class"), error=g.get("error"),
                          peak_tree_mem_gib=round(g["peak_tree_mem"] / GIB, 3), peak_swap_growth_gib=round(g["peak_swap_growth"] / GIB, 3),
                          min_available_gib=round((g["min_available"] or 0) / GIB, 3), max_pressure=g["max_pressure"],
                          elapsed_s=g["elapsed_s"], survivors=g["survivors"],
                          ap_heldout=(payload or {}).get("ap_heldout"), n_iter=(payload or {}).get("n_iter"),
                          converged=(payload or {}).get("converged")), indent=1))
    print("written:", os.path.join(RUN_DIR, "RESULT.json"))
    return 0 if status == "OK" else 3


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--authorize", default=None)
    a = ap.parse_args(argv)
    try:
        if a.check:
            check(); return 0
        if not a.authorize:
            raise Refusal("--execute requires --authorize TOKEN")
        return execute(a.authorize)
    except Refusal as e:
        print("REFUSED:", e); return 1


if __name__ == "__main__":
    sys.exit(main())
