#!/usr/bin/env python3
"""P1-only operational evaluation of the flat-selected RF-03 (stored vectors only; no fit, no P2 access).

(a) T-a hourly alert budgets on the held-out attack days Tue-Fri, settings from OPERATIONAL_SETTINGS_v1.json
    (primary q = 0.01 relative quota; sensitivity K = 500 absolute), hard-cap selection and A1.2 tie key computed with the
    FROZEN reference functions (amendment_A1 / amendment_A1b). Scores = flat RF-03 outer scores (SHA-256 verified).
(b) Final-detector T-b threshold for the P2 plan: the final RF-03 is trained on all five P1 days; its inner
    leave-one-day-out over D is exactly the flat CV, so tau_final(alpha) = quantile(benign flat-OOF RF-03 scores of all five
    days, pooled Mon -> Fri, 1 - alpha, method="higher"). P1-only; computed before any P2 scoring.
Writes P1_TA_RESULTS.json/.md and FINAL_DETECTOR_TB_THRESHOLDS.json in this folder.
"""
import hashlib
import json
import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(BRIEF, "amendment_A1"), os.path.join(BRIEF, "amendment_A1b")]
import cv_threshold_reference as A1  # noqa: E402  (frozen)
import cv_threshold_reference_A1b as A1b  # noqa: E402  (frozen)

FLAT_RUN = os.path.join(BRIEF, "FLAT_CV_P1", "cv_run")
SETTINGS = os.path.join(HERE, "OPERATIONAL_SETTINGS_v1.json")
SCHEMA = os.path.join(BRIEF, "FEATURE_SCHEMA_PRIMARY_72_DRAFT.csv")
P1_DIR = os.path.expanduser("~/Desktop/IDS_RAW_DATA_STAGE02A0/P1_extracted")
P1_FILES = {"Mon": "monday.csv", "Tue": "tuesday.csv", "Wed": "wednesday.csv", "Thu": "thursday.csv", "Fri": "friday.csv"}
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri")
SEL = "RF-03"
HOUR = 3600 * 10 ** 6


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def flat_vectors():
    recs = [json.loads(l) for l in open(os.path.join(FLAT_RUN, "LEDGER.jsonl")) if l.strip()]
    out = {}
    for r in recs:
        if r["status"] == "OK" and r["key"][0] == SEL:
            v = {}
            for kind, meta in r["detail"]["persisted"].items():
                p = os.path.join(FLAT_RUN, "scores", meta["file"])
                if sha(p) != meta["sha256"]:
                    raise SystemExit(f"SHA-256 mismatch {p}")
                v[kind] = np.load(p, allow_pickle=False)
                v[kind + "_sha256"] = meta["sha256"]
            out[r["key"][1]] = v
    return out


def day_inputs(day, ids):
    cols = [c for c in pl.read_csv(SCHEMA)["column"].to_list()]
    df = pl.read_csv(os.path.join(P1_DIR, P1_FILES[day]), columns=["id", "Timestamp"] + cols,
                     schema_overrides={"id": pl.Int64, "Timestamp": pl.Utf8, **{c: pl.Float64 for c in cols}})
    if not np.array_equal(df["id"].to_numpy(), ids):
        raise SystemExit(f"{day}: stored row ids are not the file order")
    ts = df["Timestamp"].str.to_datetime("%Y-%m-%d %H:%M:%S%.f", time_unit="us", strict=True).cast(pl.Int64).to_numpy()
    dur = df["Flow Duration"].to_numpy().astype(np.int64)
    a = A1.availability_time(ts, dur, "T2")
    X = np.ascontiguousarray(df.select(cols).to_numpy(), dtype=np.float64)
    return a, X


def ta_day(scores, y, batch, keys, quota_fn):
    alerts = np.zeros(scores.size, bool)
    per = []
    for h in np.unique(batch):
        m = np.flatnonzero(batch == h)
        quota = quota_fn(m.size)
        al, inf = A1b.ta_select_hardcap(scores[m], keys[m], quota)
        alerts[m] = al
        per.append(dict(hour=int(h), N=int(m.size), quota=int(quota), alerted=inf["alerted"], unused=inf["unused_capacity"],
                        omitted_group=inf["omitted_boundary_group_size"], K_attacks=int(y[m].sum()), TP=int((al & y[m]).sum()),
                        ties_at_cut=int((scores[m] == np.sort(scores[m])[::-1][quota - 1]).sum()) if 0 < quota <= m.size else 0))
    K = sum(p["K_attacks"] for p in per)
    TP = sum(p["TP"] for p in per)
    att_hours = [p for p in per if p["K_attacks"] > 0]
    return dict(recall_flow=(TP / K if K else None), TP=TP, K_attacks=K, alerts=int(alerts.sum()),
                precision=(TP / alerts.sum() if alerts.sum() else None),
                benign_alerts=int((alerts & ~y).sum()), fpr=float(alerts[~y].mean()),
                recall_time_block_mean=(float(np.mean([p["TP"] / p["K_attacks"] for p in att_hours])) if att_hours else None),
                n_hours=len(per), n_attack_hours=len(att_hours), n_zero_quota_hours=sum(p["quota"] == 0 for p in per),
                unused_capacity=sum(p["unused"] for p in per), n_hours_with_omitted_group=sum(p["omitted_group"] > 0 for p in per),
                max_recall_given_quota=(sum(min(p["quota"], p["K_attacks"]) for p in per) / K if K else None),
                per_hour=per)


def main():
    st = json.load(open(SETTINGS))
    q, Kcap = st["ta_primary"]["q"], st["ta_sensitivity"]["K"]
    vec = flat_vectors()
    # ---- (b) final-detector thresholds (P1 only)
    pooled = np.ascontiguousarray(np.concatenate([vec[d]["scores"][vec[d]["labels"] == 0] for d in DAYS]), dtype=np.float64)
    fin = dict(record="Final-detector T-b thresholds (RF-03 trained on all five P1 days); P1-only; computed before any P2 scoring",
               rule="tau = numpy.quantile(benign flat leave-one-day-out RF-03 scores of Mon..Fri pooled in day order, 1 - alpha, method='higher'); alert iff score > tau",
               justification="CV v2 section 4 applied one level up: the inner leave-one-day-out of a model trained on D is the flat CV over D; the stored flat vectors are those inner OOF scores (no refit)",
               settings_sha256=sha(SETTINGS), n_benign_pooled=int(pooled.size),
               pooled_benign_sha256=hashlib.sha256(pooled.tobytes()).hexdigest(),
               component_scores_sha256={d: vec[d]["scores_sha256"] for d in DAYS}, thresholds={})
    for name in ("alpha_primary", "alpha_sensitivity"):
        a = st["tb"][name]
        tau = float(np.quantile(pooled, 1 - a, method="higher"))
        fin["thresholds"][str(a)] = dict(tau=tau, tau_hex=float.hex(tau), calibration_fpr=float((pooled > tau).mean()),
                                         per_day_benign_fpr_at_tau={d: float((vec[d]["scores"][vec[d]["labels"] == 0] > tau).mean()) for d in DAYS})
    json.dump(fin, open(os.path.join(HERE, "FINAL_DETECTOR_TB_THRESHOLDS.json"), "w"), indent=1)
    # ---- (a) T-a on P1 attack days
    res = dict(settings_sha256=sha(SETTINGS), config=SEL, q=q, K=Kcap, days={})
    for d in ("Tue", "Wed", "Thu", "Fri"):
        s, y, ids = vec[d]["scores"], vec[d]["labels"].astype(bool), vec[d]["ids"]
        a, X = day_inputs(d, ids)
        keys = A1.tie_key(X, a)
        del X
        batch = a // HOUR
        res["days"][d] = dict(
            n_flows=int(s.size), prevalence=float(y.mean()), keys_sha256=hashlib.sha256(keys.tobytes()).hexdigest(),
            primary_q=ta_day(s, y, batch, keys, lambda n: A1b.quota_relative(q, n)),
            sensitivity_K=ta_day(s, y, batch, keys, lambda n: A1b.quota_absolute(Kcap, n)))
    json.dump(res, open(os.path.join(HERE, "P1_TA_RESULTS.json"), "w"), indent=1)
    L = ["# P1 T-a alert-budget results, RF-03 (flat outer scores; retrospective hourly batches, T2 availability, hard cap)", "",
         f"Settings: OPERATIONAL_SETTINGS_v1.json SHA-256 `{res['settings_sha256']}` (written before this run).", "",
         "| Day | Budget | alerts | recall (flow) | max recall allowed by quota | recall (hour-block mean) | precision | benign alerts | FPR | attack hours | zero-quota hours |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:.4f}"
    for d, v in res["days"].items():
        for lab, k in ((f"q={q}", "primary_q"), (f"K={Kcap}", "sensitivity_K")):
            e = v[k]
            L.append(f"| {d} | {lab} | {e['alerts']:,} | {f(e['recall_flow'])} ({e['TP']:,}/{e['K_attacks']:,}) | {f(e['max_recall_given_quota'])} | "
                     f"{f(e['recall_time_block_mean'])} | {f(e['precision'])} | {e['benign_alerts']:,} | {e['fpr']:.5f} | {e['n_attack_hours']} | {e['n_zero_quota_hours']} |")
    L += ["", "## Final-detector T-b thresholds (for P2; P1-only)", "",
          f"Pooled benign flat-OOF scores: {fin['n_benign_pooled']:,}; " +
          "; ".join(f"alpha {a}: tau = {t['tau']:.6f} (calibration FPR {t['calibration_fpr']:.5f})" for a, t in fin["thresholds"].items())]
    open(os.path.join(HERE, "P1_TA_RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
