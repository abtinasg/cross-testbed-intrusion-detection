#!/usr/bin/env python3
"""Flat-CV results from the stored vectors only (recomputes AP from scores/labels; checks SHA-256 and the ledger AP).

Primary criterion (CV v2 / DR-06): mAP_day = mean(AP_Tue, AP_Wed, AP_Thu, AP_Fri); Monday AP is AP_UNDEFINED_NO_POSITIVES.
The flat mAP_day of the best configuration is labelled `selection-optimistic` (CV v2 section 8.2).
Monday benign alert behaviour: the frozen FPR / alert volume needs the T-b threshold (inner OOF; separate phase). Here it
is DESCRIPTIVE ONLY: Monday benign score quantiles and the share of Monday flows with score >= 0.5 (predict() default).
Writes cv_run/FLAT_CV_RESULTS.json and cv_run/FLAT_CV_RESULTS.md. Read-only on the ledger and score files.
"""
import hashlib
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "cv_run")
DAYS = ("Tue", "Wed", "Thu", "Fri")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    recs = [json.loads(l) for l in open(os.path.join(RUN, "LEDGER.jsonl")) if l.strip()]
    ok = {tuple(r["key"]): r for r in recs if r["status"] == "OK"}
    bad = [r for r in recs if r["status"] != "OK" and tuple(r["key"]) not in ok]          # unresolved failures
    superseded = [dict(key=r["key"], status=r["status"], attempt=r.get("attempt", 1), error=r.get("error"))
                  for r in recs if r["status"] != "OK" and tuple(r["key"]) in ok]
    configs = list(dict.fromkeys(r["key"][0] for r in recs))
    sys.path[:0] = [os.path.join(os.path.dirname(HERE), d) for d in ("PILOT_BINDING_G2", "PILOT_INTEGRATION_G1", "R5_FREEZE_F7", "PILOT_G2_CORRECTIONS")]
    import f3_binding as B
    configs = list(B.load_bound_grid())
    rows, problems = [], []
    for c in configs:
        row = dict(config=c, ap={}, mon=None, peak_gib=0.0, swap_gib=0.0, time_s=0.0)
        for d in DAYS + ("Mon",):
            r = ok.get((c, d))
            if r is None:
                continue
            per = r["detail"]["persisted"]
            for meta in per.values():
                if sha(os.path.join(RUN, "scores", meta["file"])) != meta["sha256"]:
                    problems.append(f"{c}/{d}: {meta['file']} SHA-256 mismatch")
            s = np.load(os.path.join(RUN, "scores", per["scores"]["file"]))
            y = np.load(os.path.join(RUN, "scores", per["labels"]["file"]))
            g = r["guard"]
            row["peak_gib"] = max(row["peak_gib"], g["peak_tree_mem"] / 2 ** 30)
            row["swap_gib"] = max(row["swap_gib"], g["peak_swap_growth"] / 2 ** 30)
            row["time_s"] += g["elapsed_s"]
            if d == "Mon":
                if y.sum() != 0 or r["ap_status"]["reason"] != "AP_UNDEFINED_NO_POSITIVES":
                    problems.append(f"{c}/Mon: expected AP_UNDEFINED_NO_POSITIVES")
                q = np.quantile(s, [0.5, 0.9, 0.99, 0.999])
                row["mon"] = dict(ap="AP_UNDEFINED_NO_POSITIVES", n=int(s.size), q50=float(q[0]), q90=float(q[1]),
                                  q99=float(q[2]), q999=float(q[3]), max=float(s.max()),
                                  share_ge_0_5=float((s >= 0.5).mean()), n_ge_0_5=int((s >= 0.5).sum()))
            else:
                ap = float(average_precision_score(y, s))
                if abs(ap - r["ap_status"]["ap"]) > 1e-12:
                    problems.append(f"{c}/{d}: recomputed AP {ap} != ledger {r['ap_status']['ap']}")
                row["ap"][d] = dict(ap=ap, prevalence=float(y.mean()))
        row["complete"] = len(row["ap"]) == 4 and row["mon"] is not None
        row["mAP_day"] = float(np.mean([row["ap"][d]["ap"] for d in DAYS])) if len(row["ap"]) == 4 else None
        rows.append(row)
    ranked = sorted([r for r in rows if r["mAP_day"] is not None], key=lambda r: -r["mAP_day"])
    out = dict(n_ok=len(ok), n_not_ok=len(bad), not_ok=[dict(key=r["key"], status=r["status"]) for r in bad],
               superseded_failures=superseded, problems=problems, rows=rows, best=(ranked[0]["config"] if ranked else None),
               best_label="selection-optimistic (flat CV; not an unbiased estimate)")
    json.dump(out, open(os.path.join(RUN, "FLAT_CV_RESULTS.json"), "w"), indent=1)
    prev = {d: next((r["ap"][d]["prevalence"] for r in rows if d in r["ap"]), None) for d in DAYS}
    L = [f"# Flat CV (P1, primary target): {len(ok)}/60 OK" + (f", {len(bad)} not OK" if bad else ""), "",
         "| Config | AP Tue | AP Wed | AP Thu | AP Fri | **mAP_day** | Mon AP | Mon benign share score>=0.5 | Mon q99.9 | peak GiB | fit time |",
         "|---|---|---|---|---|---|---|---|---|---|---|",
         "| *prevalence* | " + " | ".join(f"{prev[d]:.4f}" if prev[d] is not None else "-" for d in DAYS) + " | | | | | | |"]
    f = lambda d, r: f"{r['ap'][d]['ap']:.4f}" if d in r["ap"] else "-"
    for r in sorted(rows, key=lambda r: -(r["mAP_day"] if r["mAP_day"] is not None else -1)):
        m = r["mon"]
        L.append(f"| {r['config']} | " + " | ".join(f(d, r) for d in DAYS) +
                 f" | **{r['mAP_day']:.4f}** | " if r["mAP_day"] is not None else f"| {r['config']} | " + " | ".join(f(d, r) for d in DAYS) + " | - | ")
        L[-1] += (f"undefined | {m['share_ge_0_5']:.5f} ({m['n_ge_0_5']}) | {m['q999']:.4f} |" if m else "- | - | - |")
        L[-1] += f" {r['peak_gib']:.2f} | {r['time_s'] / 60:.0f} min |"
    L += ["", f"Best by mAP_day: **{out['best']}** ({out['best_label']}).",
          "Monday: AP_UNDEFINED_NO_POSITIVES; the score>=0.5 share is descriptive only (frozen Monday FPR requires T-b).",
          "Superseded failed attempts (replaced by an authorised recovery attempt): " +
          ("none" if not superseded else "; ".join(f"{x['key'][0]}/{x['key'][1]} attempt {x['attempt']} {x['status']} ({x['error']})" for x in superseded)),
          "Integrity problems: " + ("none" if not problems else "; ".join(problems))]
    open(os.path.join(RUN, "FLAT_CV_RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
