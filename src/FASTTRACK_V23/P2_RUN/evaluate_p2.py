#!/usr/bin/env python3
"""P2 evaluation from the SEALED predictions (labels are read here for the first time in this phase; nothing is refitted,
recalibrated or re-thresholded). Refuses to run unless P2_SCORES_SEALED.json exists and every sealed vector matches.

Block A (ordinary detector performance, census): AP, ROC-AUC; T-b with the pre-registered final thresholds tau_final(0.01),
  tau_final(0.05): achieved FPR, recall, precision, alerts/day and per hour; FPR on Attempted (benign-labelled) rows;
  sensitivity target (Attempted rows excluded); score-scale diagnostic P1 flat-OOF benign vs P2 benign.
Block B (operational): T-a hard cap q = 0.01 and K = 500 (frozen reference functions, stored tie keys, T2 hourly batches per
  file); annotation-budget estimation M0 (stratified SRSWOR, b in {100, 250, 500} per hour, 1,000 replicates, seed 20261107):
  combined ratio estimators of recall and FPR under T-b(0.01) and T-a(q), linearisation 95% CI with fpc; bias, coverage,
  mean CI width, P(no audited attack flow) versus the census value.
"""
import hashlib
import json
import os
import sys

import numpy as np
import polars as pl
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.dirname(HERE)
BRIEF = os.path.dirname(FT)
sys.path[:0] = [os.path.join(BRIEF, "amendment_A1"), os.path.join(BRIEF, "amendment_A1b")]
import cv_threshold_reference_A1b as A1b  # noqa: E402

SEAL = os.path.join(HERE, "P2_SCORES_SEALED.json")
SCORES = os.path.join(HERE, "p2_run", "scores")
P2_DIR = os.path.expanduser("~/Desktop/IDS_RAW_DATA_STAGE02A0/P2_extracted")
FILES = {"Wed14": "Wednesday-14-02-2018.csv", "Fri16": "Friday-16-02-2018.csv"}
FLAT = os.path.join(BRIEF, "FLAT_CV_P1", "cv_run")
HOUR = 3600 * 10 ** 6


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tb_block(s, y, tau, hrs):
    al = s > tau
    neg, pos = ~y, y
    per = [dict(hour=int(h), N=int((hrs == h).sum()), alerts=int(al[hrs == h].sum()), benign_alerts=int((al & neg)[hrs == h].sum()),
                attacks=int(pos[hrs == h].sum()), tp=int((al & pos)[hrs == h].sum())) for h in np.unique(hrs)]
    return dict(tau=tau, achieved_fpr=float(al[neg].mean()), benign_alerts=int((al & neg).sum()), n_benign=int(neg.sum()),
                recall=(float(al[pos].mean()) if pos.any() else None), tp=int((al & pos).sum()), n_attack=int(pos.sum()),
                precision=(float((al & pos).sum() / al.sum()) if al.sum() else None), alerts_day=int(al.sum()),
                alerts_per_hour_max=max(p["alerts"] for p in per), alerts_per_hour_median=float(np.median([p["alerts"] for p in per])),
                benign_alerts_per_hour_max=max(p["benign_alerts"] for p in per), per_hour=per), al


def ta_block(s, y, hrs, keys, quota_fn):
    al = np.zeros(s.size, bool)
    per = []
    for h in np.unique(hrs):
        m = np.flatnonzero(hrs == h)
        q = quota_fn(m.size)
        a, inf = A1b.ta_select_hardcap(s[m], keys[m], q)
        al[m] = a
        per.append(dict(hour=int(h), N=int(m.size), quota=int(q), alerted=inf["alerted"], unused=inf["unused_capacity"],
                        omitted_group=inf["omitted_boundary_group_size"], attacks=int(y[m].sum()), tp=int((a & y[m]).sum())))
    K = int(y.sum()); TP = int((al & y).sum())
    att = [p for p in per if p["attacks"]]
    return dict(alerts=int(al.sum()), recall_flow=(TP / K if K else None), tp=TP, n_attack=K,
                max_recall_given_quota=(sum(min(p["quota"], p["attacks"]) for p in per) / K if K else None),
                precision=(TP / al.sum() if al.sum() else None), fpr=float(al[~y].mean()), benign_alerts=int((al & ~y).sum()),
                recall_hour_block_mean=(float(np.mean([p["tp"] / p["attacks"] for p in att])) if att else None),
                n_hours=len(per), n_attack_hours=len(att), unused_capacity=sum(p["unused"] for p in per),
                n_hours_with_omitted_group=sum(p["omitted_group"] > 0 for p in per), per_hour=per), al


def estimation(al, y, hrs, b, rng, reps):
    """M0 stratified SRSWOR; cells per hour: [alert&attack, noalert&attack, alert&benign, noalert&benign]."""
    hs = np.unique(hrs)
    C = np.array([[int((al & y)[hrs == h].sum()), int((~al & y)[hrs == h].sum()), int((al & ~y)[hrs == h].sum()),
                   int((~al & ~y)[hrs == h].sum())] for h in hs], dtype=np.int64)
    N = C.sum(1)
    bh = np.minimum(b, N)
    true_R = C[:, 0].sum() / C[:, :2].sum() if C[:, :2].sum() else None
    true_F = C[:, 2].sum() / C[:, 2:].sum()
    X = np.stack([rng.multivariate_hypergeometric(C[i], int(bh[i]), size=reps) for i in range(len(hs))], axis=1)  # reps x H x 4
    w = (N / bh)[None, :]
    fpc = (1 - bh / N)[None, :]
    out = {}
    for name, num, den, truth in (("recall", 0, (0, 1), true_R), ("fpr", 2, (2, 3), true_F)):
        if truth is None:
            out[name] = None; continue
        D = (w * X[:, :, list(den)].sum(2)).sum(1)
        T = (w * X[:, :, num]).sum(1)
        defined = D > 0
        est = np.where(defined, T / np.where(defined, D, 1), np.nan)
        # linearised z per sampled unit: in-numerator cell: 1 - est; other denominator cell: -est; outside: 0
        other = [c for c in den if c != num][0]
        n1, n2 = X[:, :, num], X[:, :, other]
        e = est[:, None]
        sz = n1 * (1 - e) + n2 * (-e)
        sz2 = n1 * (1 - e) ** 2 + n2 * e ** 2
        bb = bh[None, :]
        s2 = np.where(bb > 1, (sz2 - sz ** 2 / bb) / np.maximum(bb - 1, 1), 0.0)
        V = (N[None, :] ** 2 * fpc * s2 / bb).sum(1) / np.where(defined, D, 1) ** 2
        half = 1.96 * np.sqrt(np.maximum(V, 0))
        cov = np.abs(est - truth) <= half
        out[name] = dict(census=float(truth), n_replicates=reps, n_defined=int(defined.sum()),
                         p_undefined=float(1 - defined.mean()), mean_estimate=float(np.nanmean(est)),
                         bias=float(np.nanmean(est) - truth), rmse=float(np.sqrt(np.nanmean((est - truth) ** 2))),
                         coverage_95=float(cov[defined].mean()) if defined.any() else None,
                         mean_ci_width=float(np.nanmean(2 * half[defined])) if defined.any() else None)
    out["labels_per_day"] = int(bh.sum())
    out["audited_attacks_mean"] = float(X[:, :, :2].sum((1, 2)).mean())
    out["p_zero_audited_attacks"] = float((X[:, :, :2].sum((1, 2)) == 0).mean())
    out["n_attack_hours"] = int((C[:, :2].sum(1) > 0).sum())
    return out


def metrics(s, y, attempted, is_att_label, hrs, keys, comp, p1b, tau, q, K, bs, reps, rng):
    R = dict(n_rows=int(s.size), n_attack=int(y.sum()), prevalence=float(y.mean()), n_attempted=int(attempted.sum()),
             composition=comp, n_hours=int(np.unique(hrs).size), n_distinct_scores=int(np.unique(s).size),
             ap=float(average_precision_score(y, s)), roc_auc=float(roc_auc_score(y, s)))
    R["attack_episode"] = dict(labels=[c["Label"] for c in comp if c["Label"] != "BENIGN" and c["Attempted Category"] == -1],
                               note="one attack label (one documented episode) per day: episode recall = day recall")
    tb = {}
    for a in ("0.01", "0.05"):
        blk, al = tb_block(s, y, tau[a], hrs)
        blk["fpr_on_attempted_rows"] = float(al[attempted].mean()) if attempted.any() else None
        blk["fpr_benign_label_only"] = float(al[~is_att_label].mean())
        keep = ~attempted
        blk["sensitivity_target"] = dict(achieved_fpr=float(al[keep & ~y].mean()), recall=float(al[y].mean()) if y.any() else None,
                                         ap=float(average_precision_score(y[keep], s[keep])))
        tb[a] = blk
        if a == "0.01":
            al_tb = al
    R["tb"] = tb
    ob = s[~y]
    R["score_scale"] = dict(ks_p1_vs_p2_benign=float(ks_2samp(p1b, ob).statistic),
                            quantiles={str(x): dict(p1_benign=float(np.quantile(p1b, x)), p2_benign=float(np.quantile(ob, x)))
                                       for x in (0.5, 0.9, 0.99, 0.999)})
    R["ta_q"], al_ta = ta_block(s, y, hrs, keys, lambda n: A1b.quota_relative(q, n))
    R["ta_K"], _ = ta_block(s, y, hrs, keys, lambda n: A1b.quota_absolute(K, n))
    R["estimation"] = {f"b={b}": dict(tb_alpha_0_01=estimation(al_tb, y, hrs, b, rng, reps),
                                      ta_q=estimation(al_ta, y, hrs, b, rng, reps)) for b in bs}
    return R


def main():
    seal = json.load(open(SEAL))
    pre = json.load(open(os.path.join(FT, "P2_PREREGISTRATION.json")))
    if seal["preregistration_sha256"] != sha(os.path.join(FT, "P2_PREREGISTRATION.json")):
        raise SystemExit("pre-registration changed after sealing")
    tau = {a: float.fromhex(pre["tau_final"][a]["tau_hex"]) for a in ("0.01", "0.05")}
    q, K, bs, seed, reps = pre["q"], pre["K"], pre["b"], pre["estimation_seed"], pre["replicates"]
    # P1 pooled benign flat-OOF scores (score-scale reference)
    flat = {tuple(r["key"]): r for r in (json.loads(l) for l in open(os.path.join(FLAT, "LEDGER.jsonl")) if l.strip()) if r["status"] == "OK"}
    p1b = np.concatenate([np.load(os.path.join(FLAT, "scores", flat[("RF-03", d)]["detail"]["persisted"]["scores"]["file"]))[
        np.load(os.path.join(FLAT, "scores", flat[("RF-03", d)]["detail"]["persisted"]["labels"]["file"])) == 0] for d in ("Mon", "Tue", "Wed", "Thu", "Fri")])
    res = dict(seal_sha256=sha(SEAL), preregistration_sha256=seal["preregistration_sha256"], tau=pre["tau_final"], q=q, K=K, b=bs, days={})
    rng = np.random.default_rng(seed)
    arrays = {}
    for d, fn in FILES.items():
        meta = seal["days"][d]
        v = {}
        for k, m in meta["persisted"].items():
            p = os.path.join(SCORES, m["file"])
            if sha(p) != m["sha256"]:
                raise SystemExit(f"sealed vector {m['file']} changed")
            v[k] = np.load(p, allow_pickle=False)
        csv = os.path.join(P2_DIR, fn)
        if sha(csv) != meta["csv_sha256"]:
            raise SystemExit("P2 CSV changed")
        lab = pl.read_csv(csv, columns=["id", "Label", "Attempted Category"],
                          schema_overrides={"id": pl.Int64, "Label": pl.Utf8, "Attempted Category": pl.Int64})
        if not np.array_equal(lab["id"].to_numpy(), v["ids"]):
            raise SystemExit("label rows do not align with sealed row ids")
        att = lab["Attempted Category"].to_numpy()
        is_att_label = (lab["Label"] != "BENIGN").to_numpy()
        comp = lab.group_by(["Label", "Attempted Category"]).len().sort("len", descending=True).to_dicts()
        del lab
        arrays[d] = (v["scores"], is_att_label & (att == -1), is_att_label & (att != -1), is_att_label,
                     v["availability_us"] // HOUR, v["tie_keys"], comp)
        R = metrics(*arrays[d], p1b, tau, q, K, bs, reps, rng)
        res["days"][d] = R
    # pooled P2: concatenation of both days; hourly batches stay per capture file (distinct absolute UTC hours)
    cat = [np.concatenate([arrays[d][i] for d in FILES]) for i in range(6)]
    comp = {}
    for d in FILES:
        for c in arrays[d][6]:
            comp[(c["Label"], c["Attempted Category"])] = comp.get((c["Label"], c["Attempted Category"]), 0) + c["len"]
    comp = [dict(Label=k[0], **{"Attempted Category": k[1]}, len=n) for k, n in sorted(comp.items(), key=lambda x: -x[1])]
    res["days"]["P2_pooled"] = metrics(*cat, comp, p1b, tau, q, K, bs, reps, rng)
    json.dump(res, open(os.path.join(HERE, "P2_RESULTS.json"), "w"), indent=1)
    write_md(res)


def write_md(res):
    f = lambda x, n=4: "—" if x is None else f"{x:.{n}f}"
    L = ["# P2 results — RF-03 final detector (P1 all days), sealed predictions", "",
         f"Seal `{res['seal_sha256']}` · pre-registration `{res['preregistration_sha256']}`. **P2 is a previously inspected development domain, not a blind test.**", "",
         "## Composition", "", "| Day | rows | attacks (primary) | prevalence | Attempted rows (benign) | hourly batches |", "|---|---|---|---|---|---|"]
    for d, R in res["days"].items():
        L.append(f"| {d} | {R['n_rows']:,} | {R['n_attack']:,} ({', '.join(R['attack_episode']['labels'])}) | {R['prevalence']:.4f} | {R['n_attempted']:,} | {R['n_hours']} |")
    L += ["", "## Block A — detector performance (census)", "", "| Day | AP | ROC-AUC | α | τ | achieved FPR | recall | precision | alerts/day | alerts/h max | benign alerts/h max | FPR on Attempted | sens. target FPR / AP |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d, R in res["days"].items():
        for a, t in R["tb"].items():
            L.append(f"| {d} | {R['ap']:.4f} | {R['roc_auc']:.4f} | {a} | {t['tau']:.6f} | **{t['achieved_fpr']:.5f}** | {f(t['recall'])} | {f(t['precision'])} | "
                     f"{t['alerts_day']:,} | {t['alerts_per_hour_max']:,} | {t['benign_alerts_per_hour_max']:,} | {f(t['fpr_on_attempted_rows'])} | "
                     f"{t['sensitivity_target']['achieved_fpr']:.5f} / {t['sensitivity_target']['ap']:.4f} |")
    L += ["", "Score scale (P1 flat-OOF benign vs P2 benign): " + "; ".join(f"{d} KS {R['score_scale']['ks_p1_vs_p2_benign']:.4f}" for d, R in res["days"].items()),
          "", "## Block B — budget-constrained alerting (hard cap)", "",
          "| Day | budget | alerts | recall (flow) | max recall permitted | precision | FPR | recall hour-block mean | attack hours |", "|---|---|---|---|---|---|---|---|---|"]
    for d, R in res["days"].items():
        for lab, k in ((f"q={res['q']}", "ta_q"), (f"K={res['K']}", "ta_K")):
            e = R[k]
            L.append(f"| {d} | {lab} | {e['alerts']:,} | {f(e['recall_flow'])} ({e['tp']:,}/{e['n_attack']:,}) | {f(e['max_recall_given_quota'])} | {f(e['precision'])} | "
                     f"{e['fpr']:.5f} | {f(e['recall_hour_block_mean'])} | {e['n_attack_hours']} |")
    L += ["", "## Block B — annotation-budget estimation (M0, 1,000 replicates)", "",
          "| Day | b/h | labels/day | rule | metric | census | bias | RMSE | 95% CI coverage | mean CI width | P(undefined) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d, R in res["days"].items():
        for b, E in R["estimation"].items():
            for rule, e in E.items():
                for m in ("recall", "fpr"):
                    x = e[m]
                    if x is None:
                        continue
                    L.append(f"| {d} | {b[2:]} | {e['labels_per_day']:,} | {rule} | {m} | {x['census']:.5f} | {x['bias']:+.5f} | {x['rmse']:.5f} | "
                             f"{f(x['coverage_95'], 3)} | {f(x['mean_ci_width'], 5)} | {x['p_undefined']:.3f} |")
    open(os.path.join(HERE, "P2_RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
