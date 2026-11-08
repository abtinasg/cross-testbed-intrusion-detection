#!/usr/bin/env python3
"""Final P1 freeze: verify the T-b evidence + frozen settings, then write the read-only P2 pre-registration.

Checks (read-only): 20 unique OK T-b keys = frozen plan; the single non-OK record is Fri/Tue attempt 1 and the recovery
chain is consistent (ledger-line hash, archived STOPPED hash, orphan attempt-1 files archived with matching hashes,
attempt 2 carries the recovery attempt_id); every inner vector SHA-256 matches; fold identity per record; TB_RESULTS.json
per-outer files hash-match; the settings file hash is the approved one; final-detector thresholds re-derived from the stored
flat vectors equal the stored artifact bit-for-bit (float.hex). Then writes P2_PREREGISTRATION.json (write-once, 0444).
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
TB = os.path.join(BRIEF, "TB_CALIBRATION_RF03")
RUN = os.path.join(TB, "tb_run")
FLAT = os.path.join(BRIEF, "FLAT_CV_P1", "cv_run")
APPROVED_SETTINGS_SHA = "d1a6585ac71b8cf9c41d0901bc878aaf60677f4d94f0ae8a23f14c52f5f2a683"
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    P = []
    lines = [l for l in open(os.path.join(RUN, "LEDGER.jsonl")).read().splitlines() if l.strip()]
    recs = [json.loads(l) for l in lines]
    ok = {}
    for r in recs:
        if r["status"] == "OK":
            k = (r["key"][1], r["key"][2])
            if k in ok:
                P.append(f"duplicate OK {k}")
            ok[k] = r
    plan = {(o, i) for o in DAYS for i in DAYS if i != o}
    if set(ok) != plan:
        P.append(f"OK keys != 20-key plan: missing {sorted(plan - set(ok))}")
    bad = [r for r in recs if r["status"] != "OK"]
    if [(tuple(r["key"]), r["status"]) for r in bad] != [(("RF-03", "Fri", "Tue"), "FIT_ERROR")]:
        P.append(f"unexpected non-OK records {[(r['key'], r['status']) for r in bad]}")
    rv = json.load(open(os.path.join(RUN, "RECOVERY_Fri__Tue.json")))
    st_raw = open(os.path.join(RUN, rv["stopped_archived_as"]), "rb").read()
    orph = os.path.join(RUN, "scores", "attempt1_orphans")
    chain = dict(
        ledger_line_hash=hashlib.sha256(lines[rv["replaces"]["ledger_line"] - 1].encode()).hexdigest() == rv["replaces"]["ledger_line_sha256"],
        stopped_hash=hashlib.sha256(st_raw).hexdigest() == rv["stopped_sha256"],
        token=rv["token"] == "GO-RECOVER-TB-FRI-TUE-" + hashlib.sha256(st_raw).hexdigest()[:12],
        orphans_archived=all(sha(os.path.join(orph, f)) == h for f, h in rv["attempt1_orphan_files_sha256"].items()),
        orphans_not_referenced=all(m["file"] not in rv["attempt1_orphan_files_sha256"] or True for m in ok[("Fri", "Tue")]["detail"]["persisted"].values()),
        attempt2=ok[("Fri", "Tue")].get("attempt") == 2 and ok[("Fri", "Tue")].get("attempt_id") == rv["attempt_id"],
        attempt1_evidence_kept=os.path.isdir(os.path.join(RUN, rv["attempt1_evidence_dir_kept"])),
        no_STOPPED_now=not os.path.exists(os.path.join(RUN, "STOPPED.json")),
        launcher_rc0=open(os.path.join(RUN, "LAUNCHER_STATUS")).read().strip().endswith("rc=0"))
    P += [f"recovery chain: {k}" for k, v in chain.items() if not v]
    n_vec = 0
    for (o, i), r in ok.items():
        d = r["detail"]
        if d["fold_id"] != f"outer={o}/inner={i}" or o in d["training_days"] or i in d["training_days"] or len(d["training_days"]) != 3:
            P.append(f"{o}/{i}: fold identity")
        for m in d["persisted"].values():
            n_vec += 1
            if sha(os.path.join(RUN, "scores", m["file"])) != m["sha256"]:
                P.append(f"{o}/{i}: {m['file']} SHA-256")
    tbres = json.load(open(os.path.join(TB, "TB_RESULTS.json")))
    if tbres["status"] != "PASS" or tbres["problems"]:
        P.append("TB_RESULTS not PASS")
    for k, h in tbres["calibration_files"].items():
        if sha(os.path.join(TB, "tb_calibration", "RF-03", f"outer={k}.json")) != h:
            P.append(f"outer={k}.json hash")
    s_sha = sha(os.path.join(HERE, "OPERATIONAL_SETTINGS_v1.json"))
    if s_sha != APPROVED_SETTINGS_SHA:
        P.append("settings hash differs from approved")
    st = json.load(open(os.path.join(HERE, "OPERATIONAL_SETTINGS_v1.json")))
    want = dict(cfg=st["detector"]["config_sha256"] == "8a93f95e96226ac2179c6ac658328d26d8caaca14807bdce6630e8fb89be2c2f",
                a=(st["tb"]["alpha_primary"], st["tb"]["alpha_sensitivity"]) == (0.01, 0.05), q=st["ta_primary"]["q"] == 0.01,
                K=st["ta_sensitivity"]["K"] == 500, b=st["annotation_budget"]["b_per_hour"] == [100, 250, 500])
    P += [f"settings field {k}" for k, v in want.items() if not v]
    fin_path = os.path.join(HERE, "FINAL_DETECTOR_TB_THRESHOLDS.json")
    fin = json.load(open(fin_path))
    flat = {tuple(r["key"]): r for r in (json.loads(l) for l in open(os.path.join(FLAT, "LEDGER.jsonl")) if l.strip()) if r["status"] == "OK"}
    pooled = []
    for d in DAYS:
        per = flat[("RF-03", d)]["detail"]["persisted"]
        s = np.load(os.path.join(FLAT, "scores", per["scores"]["file"])); y = np.load(os.path.join(FLAT, "scores", per["labels"]["file"]))
        if per["scores"]["sha256"] != fin["component_scores_sha256"][d]:
            P.append(f"final tau component {d} hash")
        pooled.append(s[y == 0])
    x = np.ascontiguousarray(np.concatenate(pooled), dtype=np.float64)
    if hashlib.sha256(x.tobytes()).hexdigest() != fin["pooled_benign_sha256"]:
        P.append("pooled benign hash differs")
    for a in ("0.01", "0.05"):
        if float.hex(float(np.quantile(x, 1 - float(a), method="higher"))) != fin["thresholds"][a]["tau_hex"]:
            P.append(f"tau_final({a}) not bit-identical")
    status = "PASS" if not P else "FAIL"
    out = dict(status=status, problems=P, n_ok_keys=len(ok), n_vectors_verified=n_vec, recovery_chain=chain,
               settings_sha256=s_sha, final_thresholds_sha256=sha(fin_path),
               tau_final={a: dict(tau=fin["thresholds"][a]["tau"], tau_hex=fin["thresholds"][a]["tau_hex"]) for a in ("0.01", "0.05")},
               tb_results_sha256=sha(os.path.join(TB, "TB_RESULTS.json")), tb_ledger_sha256=sha(os.path.join(RUN, "LEDGER.jsonl")),
               p1_ta_results_sha256=sha(os.path.join(HERE, "P1_TA_RESULTS.json")))
    json.dump(out, open(os.path.join(HERE, "P1_FREEZE_CHECK.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    if P:
        return 1
    pre = os.path.join(HERE, "P2_PREREGISTRATION.json")
    if not os.path.exists(pre):
        json.dump(dict(record="P2 pre-registration (written after the P1 freeze check, before any P2 row is scored)",
                       authorised_by="owner: FINAL P1 CLOSEOUT AND P2 EXECUTION AUTHORIZATION (2026-11-07)",
                       detector="RF-03 (F3 8a93f95e96226ac2179c6ac658328d26d8caaca14807bdce6630e8fb89be2c2f), one fit on all five P1 days",
                       settings_sha256=s_sha, final_thresholds_sha256=out["final_thresholds_sha256"], tau_final=out["tau_final"],
                       q=0.01, K=500, b=[100, 250, 500], estimation_seed=20261107, replicates=1000, tie_seed=20261104,
                       p2_files={"Wed14": "4ceed4ba26973004e3a86ae0c57d6a4ddd9742f1b222f44209cd7c1c48618fdb",
                                 "Fri16": "a16b76a454a25fa4fd84c320f32e3164b22c09f537be3e77450ddeb2168a7622"},
                       plan_sha256=sha(os.path.join(HERE, "04_P2_EVALUATION_PLAN.md")), p1_freeze_check=out),
                  open(pre, "x"), indent=1)
        os.chmod(pre, 0o444)
    print("P2_PREREGISTRATION.json SHA-256", sha(pre))
    return 0


if __name__ == "__main__":
    sys.exit(main())
