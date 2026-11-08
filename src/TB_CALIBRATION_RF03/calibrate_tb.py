#!/usr/bin/env python3
"""T-b calibration for RF-03 from the stored vectors only (CV v2 section 4 + A1.1; TB_OOF_SCORE_EXPORT_SPEC sections 4-5).

For each outer day k:
  benign_k  = benign (primary target) inner-OOF scores of RF-03 on days j != k, pooled in day order Mon -> Fri;
  tau_k(a)  = numpy.quantile(benign_k, 1 - a, method="higher"), a in {0.01 primary, 0.05 sensitivity} (RG8_V21 prespec);
  alert     iff score > tau_k (strict); applied to the STORED flat RF-03 outer scores of day k (identical fit; bytes re-read).
Labels of day k are used only to compute the reported metrics (FPR, recall), never for tau_k.
Reported: nominal a vs achieved FPR, recall (attack days), alerts per day and per hour (hour of T2 availability
a = Timestamp + Flow Duration, as recorded in the CSV), calibration-set FPR (<= a by construction), score-scale diagnostic
(inner-OOF benign vs outer held-out benign quantiles and KS distance; evaluation only, never fed back).
Writes tb_calibration/RF-03/outer=<k>.json, TB_RESULTS.json, TB_RESULTS.md. Read-only on tb_run and FLAT_CV_P1.
"""
import hashlib
import json
import os
import sys

import numpy as np
import polars as pl
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
TB_RUN = os.path.join(HERE, "tb_run")
FLAT_RUN = os.path.join(BRIEF, "FLAT_CV_P1", "cv_run")
OUT = os.path.join(HERE, "tb_calibration", "RF-03")
ALPHA_FILE = os.path.join(BRIEF, "RG8_V21", "TB_ALPHA_PRESPECIFICATION.json")
P1_DIR = os.path.expanduser("~/Desktop/IDS_RAW_DATA_STAGE02A0/P1_extracted")
P1_FILES = {"Mon": "monday.csv", "Tue": "tuesday.csv", "Wed": "wednesday.csv", "Thu": "thursday.csv", "Fri": "friday.csv"}
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri")
SEL = "RF-03"
Q = (0.5, 0.9, 0.99, 0.999)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_verified(run, per):
    out = {}
    for kind, meta in per.items():
        p = os.path.join(run, "scores", meta["file"])
        if sha(p) != meta["sha256"]:
            raise SystemExit(f"SHA-256 mismatch: {p}")
        out[kind] = np.load(p, allow_pickle=False)
    return out


def hours_of(day, ids):
    df = pl.read_csv(os.path.join(P1_DIR, P1_FILES[day]), columns=["id", "Timestamp", "Flow Duration"],
                     schema_overrides={"id": pl.Int64, "Timestamp": pl.Utf8, "Flow Duration": pl.Int64})
    ts = df["Timestamp"].str.to_datetime("%Y-%m-%d %H:%M:%S%.f", time_unit="us", strict=False)
    if ts.null_count():
        raise SystemExit(f"{day}: {ts.null_count()} unparsable timestamps")
    avail = ts.cast(pl.Int64).to_numpy() + df["Flow Duration"].to_numpy()       # T2: a = s + duration (us)
    pos = {int(v): i for i, v in enumerate(df["id"].to_numpy())} if not np.array_equal(df["id"].to_numpy(), ids) else None
    if pos is not None:
        avail = avail[np.array([pos[int(i)] for i in ids])]
    return avail // (3600 * 10 ** 6)


def main():
    alpha = json.load(open(ALPHA_FILE))
    alphas = (("primary", alpha["alpha_primary"]), ("sensitivity", alpha["alpha_sensitivity"]))
    tb = [json.loads(l) for l in open(os.path.join(TB_RUN, "LEDGER.jsonl")) if l.strip()]
    ok = {(r["key"][1], r["key"][2]): r for r in tb if r["status"] == "OK"}
    complete = [k for k in DAYS if all((k, j) in ok for j in DAYS if j != k)]
    partial = len(complete) < 5
    if partial and "--partial" not in sys.argv:
        raise SystemExit(f"T-b ledger: {len(ok)} OK of {len(tb)} records; complete outers {complete}; use --partial")
    flat = {tuple(r["key"]): r for r in (json.loads(l) for l in open(os.path.join(FLAT_RUN, "LEDGER.jsonl")) if l.strip())
            if r["status"] == "OK" and r["key"][0] == SEL}
    flat_vec = {d: load_verified(FLAT_RUN, flat[(SEL, d)]["detail"]["persisted"]) for d in DAYS}
    os.makedirs(OUT, exist_ok=True)
    results, problems = {}, []
    inner_ap = {}
    for k in complete:
        comp, pooled = [], []
        for j in DAYS:                                   # pooled in day order Mon -> Fri
            if j == k:
                continue
            r = ok[(k, j)]
            det = r["detail"]
            if det["config_sha256"] != flat[(SEL, k)]["detail"]["config_sha256"] or det["outer"] != k or det["inner"] != j \
                    or k in det["training_days"] or j in det["training_days"]:
                problems.append(f"{k}/{j}: fold identity or config mismatch")
            v = load_verified(TB_RUN, det["persisted"])
            if not (np.array_equal(v["ids"], flat_vec[j]["ids"]) and np.array_equal(v["labels"], flat_vec[j]["labels"])):
                problems.append(f"{k}/{j}: inner row ids/labels differ from the frozen day-{j} order")
            pooled.append(v["scores"][v["labels"] == 0])
            inner_ap[f"{k}/{j}"] = det["ap_inner"]
            comp.append(dict(fold_id=det["fold_id"], ledger_attempt_id=r["attempt_id"],
                             scores_sha256=det["persisted"]["scores"]["sha256"], labels_sha256=det["persisted"]["labels"]["sha256"],
                             n_benign=int((v["labels"] == 0).sum())))
        x = np.ascontiguousarray(np.concatenate(pooled), dtype=np.float64)
        outer = flat_vec[k]
        s, y = outer["scores"], outer["labels"].astype(bool)
        hrs = hours_of(k, outer["ids"])
        uh = np.unique(hrs)
        rec = dict(config=SEL, outer=k, components=comp, n_benign_pooled=int(x.size),
                   pooled_benign_sha256=hashlib.sha256(x.tobytes()).hexdigest(), quantile_method="higher", comparison=">",
                   outer_scores_sha256=flat[(SEL, k)]["detail"]["persisted"]["scores"]["sha256"],
                   outer_model="flat RF-03 fit for outer day (identical fit, stored bytes re-read; no refit)", alpha={})
        for name, a in alphas:
            tau = float(np.quantile(x, 1 - a, method="higher"))
            cal_fpr = float((x > tau).mean())
            if cal_fpr > a:
                problems.append(f"{k} alpha={a}: calibration FPR {cal_fpr} > alpha")
            al = s > tau
            per_hour = [dict(hour_index=int(h), n_flows=int((hrs == h).sum()), alerts=int(al[hrs == h].sum()),
                             benign_alerts=int((al & ~y)[hrs == h].sum())) for h in uh]
            e = dict(alpha=a, role=name, tau=tau, tau_hex=float.hex(tau), calibration_fpr=cal_fpr,
                     n_calibration_ties_at_tau=int((x == tau).sum()),
                     achieved_fpr=float(al[~y].mean()), benign_alerts=int((al & ~y).sum()), n_benign=int((~y).sum()),
                     alerts_day=int(al.sum()), n_flows=int(s.size),
                     recall=(float(al[y].mean()) if y.any() else None), tp=int((al & y).sum()), n_attack=int(y.sum()),
                     precision=(float((al & y).sum() / al.sum()) if al.sum() else None),
                     alerts_per_hour=dict(max=int(max(p["alerts"] for p in per_hour)), median=float(np.median([p["alerts"] for p in per_hour])),
                                          n_hours=len(per_hour)),
                     benign_alerts_per_hour_max=int(max(p["benign_alerts"] for p in per_hour)), per_hour=per_hour,
                     outer_ties_at_tau=int((s == tau).sum()))
            rec["alpha"][name] = e
        ob = s[~y]
        qs = Q + tuple(1 - a for _, a in alphas)
        rec["score_scale_diagnostic"] = dict(
            note="evaluation only; never alters tau, alpha or selection",
            quantiles={str(q): dict(inner_oof_benign=float(np.quantile(x, q)), outer_heldout_benign=float(np.quantile(ob, q))) for q in qs},
            ks_distance=float(ks_2samp(x, ob).statistic))
        path = os.path.join(OUT, f"outer={k}.json")
        if partial:
            pass                                          # partial: no write-once per-outer records
        elif os.path.exists(path):
            problems.append(f"{path} exists; not overwritten")
        else:
            json.dump(rec, open(path, "x"), indent=1)
            os.chmod(path, 0o444)
        results[k] = rec
    summary = dict(status=("PARTIAL" if partial else "PASS") if not problems else "FAIL", problems=problems, config=SEL,
                   complete_outers=complete, alphas=dict(alphas),
                   inner_ap=inner_ap, tb_ledger_sha256=sha(os.path.join(TB_RUN, "LEDGER.jsonl")),
                   calibration_files=({} if partial else {k: sha(os.path.join(OUT, f"outer={k}.json")) for k in DAYS}),
                   per_outer={k: dict(n_benign_pooled=v["n_benign_pooled"], ks=v["score_scale_diagnostic"]["ks_distance"],
                                      **{n: {f: v["alpha"][n][f] for f in ("tau", "tau_hex", "calibration_fpr", "achieved_fpr",
                                         "benign_alerts", "n_benign", "alerts_day", "recall", "tp", "n_attack", "precision",
                                         "alerts_per_hour", "benign_alerts_per_hour_max")} for n, _ in alphas})
                              for k, v in results.items()})
    stem = "TB_RESULTS_PARTIAL" if partial else "TB_RESULTS"
    json.dump(summary, open(os.path.join(HERE, stem + ".json"), "w"), indent=1)
    L = [f"# RF-03 T-b source-only calibration: **{summary['status']}**", "",
         "tau_k from benign inner-OOF scores of the 3-day RF-03 models on the 4 training days of outer fold k; applied to "
         "the stored flat RF-03 outer scores. Alert iff score > tau_k. Labels of day k used only for the metrics.", ""]
    for n, a in alphas:
        L += [f"## alpha = {a} ({n})", "",
              "| Outer day | n benign pooled | tau_k | calib. FPR | achieved FPR | benign alerts / benign | recall | precision | alerts/day | alerts/h max | KS |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in complete:
            e, v = results[k]["alpha"][n], results[k]
            rc = "undefined (no attacks)" if e["recall"] is None else f"{e['recall']:.4f} ({e['tp']:,}/{e['n_attack']:,})"
            pr = "-" if e["precision"] is None else f"{e['precision']:.4f}"
            L.append(f"| {k} | {v['n_benign_pooled']:,} | {e['tau']:.6f} | {e['calibration_fpr']:.5f} | **{e['achieved_fpr']:.5f}** | "
                     f"{e['benign_alerts']:,} / {e['n_benign']:,} | {rc} | {pr} | {e['alerts_day']:,} | {e['alerts_per_hour']['max']:,} | "
                     f"{v['score_scale_diagnostic']['ks_distance']:.4f} |")
        L.append("")
    L += ["Inner AP (descriptive; 3-day models): " + ", ".join(f"{k}: {('undef' if a is None else f'{a:.4f}')}" for k, a in inner_ap.items()),
          "", "Problems: " + ("none" if not problems else "; ".join(problems))]
    open(os.path.join(HERE, stem + ".md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
