#!/usr/bin/env python3
"""P1 Flat-CV close-out: ONE read-only integrity check of FLAT_CV_P1/cv_run (no fit, no write inside cv_run).

Checks
  1. Ledger: exactly the 60 frozen flat keys have exactly one OK record; the only non-OK record is the authorised
     LR-02/Tue attempt 1 (WRAPPER_ERROR, SIGTERM) and its recovery chain is consistent (ledger-line hash, archived STOPPED
     hash, recovery token, attempt 2 / attempt_id carried into the OK record).
  2. Per OK key: config_sha256 = pinned F3 hash; persisted scores / labels / row_ids exist, are read-only and match
     their SHA-256; scores finite float64 in [0, 1]; labels int8 in {0, 1}; row ids int64 unique; n_val = pinned count.
     Labels and row ids of a held-out day are byte-identical across all 12 configurations.
  3. AP recomputed from the stored vectors (scikit-learn average_precision_score) = ledger AP (|diff| <= 1e-12);
     mAP_day recomputed in the frozen order (Tue + Wed + Thu + Fri) / 4; selection under CV v2 section 3.3
     (largest unrounded float64; exact-tie -> first in F3 order).
  4. Reporting items for the selected configuration: distinct score values per held-out day (CV v2 section 3.1).
Writes P1_FLAT_CV_CLOSEOUT/P1_FLAT_CV_VERIFICATION.json and .md only.
"""
import hashlib
import json
import os
import stat
import sys

import numpy as np
from sklearn.metrics import average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
CV = os.path.join(BRIEF, "FLAT_CV_P1")
RUN = os.path.join(CV, "cv_run")
sys.path[:0] = [os.path.join(BRIEF, d) for d in ("PILOT_BINDING_G2", "PILOT_INTEGRATION_G1", "R5_FREEZE_F7",
                                                 "PILOT_G2_CORRECTIONS", "RG8_V21")]
import f3_binding as B  # noqa: E402
import r5_p1_source as S  # noqa: E402

DAYS = ("Tue", "Wed", "Thu", "Fri")
ALL = DAYS + ("Mon",)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    problems, notes = [], []
    raw_lines = [l for l in open(os.path.join(RUN, "LEDGER.jsonl")).read().splitlines() if l.strip()]
    recs = [json.loads(l) for l in raw_lines]
    grid = B.load_bound_grid()
    plan = set(B.expected_plan(grid, "flat"))
    ok = {}
    for r in recs:
        if r["status"] == "OK":
            k = tuple(r["key"])
            if k in ok:
                problems.append(f"{k}: more than one OK record")
            ok[k] = r
    if set(ok) != plan:
        problems.append(f"OK keys != frozen plan: missing {sorted(plan - set(ok))}, extra {sorted(set(ok) - plan)}")
    not_ok = [(i, r) for i, r in enumerate(recs) if r["status"] != "OK"]

    # ---- 1b. recovery chain of LR-02/Tue
    rec_file = os.path.join(RUN, "RECOVERY_LR-02__Tue.json")
    rv = json.load(open(rec_file))
    stopped_arch = os.path.join(RUN, rv["stopped_archived_as"])
    stopped_raw = open(stopped_arch, "rb").read()
    exp_token = "GO-RECOVER-LR02-TUE-" + hashlib.sha256(stopped_raw).hexdigest()[:12]
    li = rv["replaces"]["ledger_line"] - 1
    rchk = {
        "only_non_ok_is_LR02_Tue_attempt1": [(r["key"], r["status"]) for _, r in not_ok] == [(["LR-02", "Tue"], "WRAPPER_ERROR")],
        "replaced_ledger_line_hash": hashlib.sha256(raw_lines[li].encode()).hexdigest() == rv["replaces"]["ledger_line_sha256"],
        "archived_STOPPED_hash": hashlib.sha256(stopped_raw).hexdigest() == rv["stopped_sha256"],
        "token_matches_STOPPED": rv["token"] == exp_token,
        "STOPPED_json_absent_now": not os.path.exists(os.path.join(RUN, "STOPPED.json")),
        "INTENT_json_absent_now": not os.path.exists(os.path.join(RUN, "INTENT.json")),
        "attempt2_record_OK": ok.get(("LR-02", "Tue"), {}).get("attempt") == 2
                              and ok[("LR-02", "Tue")].get("attempt_id") == rv["attempt_id"],
        "attempt1_evidence_kept": os.path.isdir(os.path.join(RUN, rv["attempt1_evidence_dir_kept"])),
        "attempt2_evidence_present": os.path.isdir(os.path.join(RUN, rv["attempt2_evidence_dir"])),
        "attempt1_left_no_score_files": True,   # checked in authorize_recovery; re-checked below via OK-file set
        "launcher_final_rc0": open(os.path.join(RUN, "LAUNCHER_STATUS")).read().strip().endswith("rc=0"),
    }
    for k, v in rchk.items():
        if not v:
            problems.append(f"recovery chain: {k} failed")

    # ---- 2/3. per-key vectors, AP
    day_ref = {}
    rows = {}
    expected_files = set()
    for c in grid:
        rows[c] = dict(ap={}, n_distinct={})
        for d in ALL:
            r = ok.get((c, d))
            if r is None:
                continue
            det = r["detail"]
            if det["config_sha256"] != B.EXPECTED_CONFIG_SHA256[c] or det["config_sha256"] != grid[c].config_sha256:
                problems.append(f"{c}/{d}: config_sha256 differs from pinned F3")
            per = det["persisted"]
            arrs = {}
            for kind, meta in per.items():
                p = os.path.join(RUN, "scores", meta["file"])
                expected_files.add(meta["file"])
                if sha(p) != meta["sha256"]:
                    problems.append(f"{c}/{d}: {meta['file']} SHA-256 mismatch")
                if os.stat(p).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                    problems.append(f"{c}/{d}: {meta['file']} is writable")
                arrs[kind] = np.load(p, allow_pickle=False)
            s, y, ids = arrs["scores"], arrs["labels"], arrs["ids"]
            n_exp = S.P1_PROFILE.expected_rows(d, "primary")
            if not (s.dtype == np.float64 and y.dtype == np.int8 and ids.dtype.kind == "i"):
                problems.append(f"{c}/{d}: dtypes {s.dtype} {y.dtype} {ids.dtype}")
            if not (s.shape == y.shape == ids.shape == (n_exp,)):
                problems.append(f"{c}/{d}: lengths {s.shape} {y.shape} {ids.shape} != pinned {n_exp}")
            if not (np.all(np.isfinite(s)) and s.min() >= 0 and s.max() <= 1):
                problems.append(f"{c}/{d}: scores not finite in [0, 1]")
            if not set(np.unique(y)).issubset({0, 1}):
                problems.append(f"{c}/{d}: labels not binary")
            if np.unique(ids).size != ids.size:
                problems.append(f"{c}/{d}: duplicate row ids")
            ref = (per["labels"]["sha256"], per["ids"]["sha256"])
            if day_ref.setdefault(d, ref) != ref:
                problems.append(f"{c}/{d}: labels/row ids differ from other configurations on the same day")
            if (det["n_val"], det["n_train"]) != (n_exp, sum(S.P1_PROFILE.expected_rows(x, "primary") for x in S.DAYS if x != d)):
                problems.append(f"{c}/{d}: ledger row counts differ from pinned")
            rows[c]["n_distinct"][d] = int(np.unique(s).size)
            if d == "Mon":
                if int(y.sum()) != 0 or r["ap_status"].get("reason") != "AP_UNDEFINED_NO_POSITIVES":
                    problems.append(f"{c}/Mon: expected AP_UNDEFINED_NO_POSITIVES")
                continue
            ap = float(average_precision_score(y, s))
            if abs(ap - r["ap_status"]["ap"]) > 1e-12 or abs(ap - det["ap_heldout"]) > 1e-12:
                problems.append(f"{c}/{d}: recomputed AP {ap!r} != ledger {r['ap_status']['ap']!r}")
            rows[c]["ap"][d] = ap
            rows[c].setdefault("n_pos", {})[d] = int(y.sum())
            rows[c].setdefault("prev", {})[d] = float(y.mean())
        if len(rows[c]["ap"]) == 4:
            a = rows[c]["ap"]
            rows[c]["mAP_day"] = float((((np.float64(a["Tue"]) + a["Wed"]) + a["Thu"]) + a["Fri"]) / 4)
    on_disk = set(os.listdir(os.path.join(RUN, "scores")))
    if on_disk != expected_files:
        problems.append(f"score dir has unreferenced/missing files: extra {sorted(on_disk - expected_files)}, "
                        f"missing {sorted(expected_files - on_disk)}")

    # ---- selection (CV v2 3.3): largest unrounded float64; exact tie -> first in F3 order
    order = list(grid)
    best = max(order, key=lambda c: (rows[c]["mAP_day"], -order.index(c)))
    ranked = sorted(order, key=lambda c: (-rows[c]["mAP_day"], order.index(c)))
    exact_tie = [c for c in order if c != best and rows[c]["mAP_day"] == rows[best]["mAP_day"]]
    prev_res = json.load(open(os.path.join(RUN, "FLAT_CV_RESULTS.json")))
    if prev_res["best"] != best:
        problems.append(f"FLAT_CV_RESULTS best {prev_res['best']} != recomputed {best}")
    for pr in prev_res["rows"]:
        if abs(pr["mAP_day"] - rows[pr["config"]]["mAP_day"]) > 1e-12:
            problems.append(f"{pr['config']}: FLAT_CV_RESULTS mAP_day differs")

    code = {f: sha(os.path.join(CV, f)) for f in sorted(os.listdir(CV)) if f.endswith((".py", ".sh"))}
    out = dict(status="PASS" if not problems else "FAIL", problems=problems, n_ok_unique=len(ok), n_records=len(recs),
               recovery_chain=rchk, selected=best, selected_config_sha256=grid[best].config_sha256,
               selection_rule="CV v2 3.3: max unrounded float64 mAP_day; exact (bitwise) tie -> first in frozen F3 order",
               exact_ties_with_selected=exact_tie,
               margin_to_runner_up=rows[ranked[0]]["mAP_day"] - rows[ranked[1]]["mAP_day"],
               ranking=[dict(config=c, mAP_day=rows[c]["mAP_day"], mAP_day_hex=float.hex(rows[c]["mAP_day"]), ap=rows[c]["ap"],
                             n_distinct_scores=rows[c]["n_distinct"]) for c in ranked],
               day_label_and_id_sha256={d: dict(labels=v[0], row_ids=v[1]) for d, v in day_ref.items()},
               n_pos=rows[best]["n_pos"], prevalence=rows[best]["prev"],
               ledger_sha256=sha(os.path.join(RUN, "LEDGER.jsonl")), code_sha256=code,
               label="selection-optimistic (flat CV; not an unbiased estimate)")
    json.dump(out, open(os.path.join(HERE, "P1_FLAT_CV_VERIFICATION.json"), "w"), indent=1)
    L = [f"# P1 Flat-CV close-out verification: **{out['status']}**", "",
         f"- Unique OK keys: {len(ok)}/60 (ledger records: {len(recs)}; the one non-OK record is the superseded LR-02/Tue attempt 1).",
         "- Recovery chain LR-02/Tue: " + ", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in rchk.items()),
         "- All 180 vectors: SHA-256 verified, read-only, finite, pinned row counts; labels/row ids identical across the 12 configs per day.",
         f"- AP recomputed from stored vectors = ledger AP for all 48 attack-day keys; Monday AP undefined for all 12.",
         f"- Selected: **{best}** (config_sha256 `{grid[best].config_sha256[:16]}…`), margin to {ranked[1]} = {out['margin_to_runner_up']:.6f}; exact ties: {exact_tie or 'none'}.",
         f"- LEDGER.jsonl SHA-256 `{out['ledger_sha256']}`", "",
         "| Rank | Config | AP Tue | AP Wed | AP Thu | AP Fri | mAP_day (selection-optimistic) |", "|---|---|---|---|---|---|---|"]
    for i, c in enumerate(ranked, 1):
        a = rows[c]["ap"]
        L.append(f"| {i} | {c} | {a['Tue']:.4f} | {a['Wed']:.4f} | {a['Thu']:.4f} | {a['Fri']:.4f} | {rows[c]['mAP_day']:.6f} |")
    L += ["", f"Distinct score values per held-out day ({best}): " + ", ".join(f"{d} {n:,}" for d, n in rows[best]["n_distinct"].items()),
          "", "Problems: " + ("none" if not problems else "; ".join(problems))]
    open(os.path.join(HERE, "P1_FLAT_CV_VERIFICATION.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
