#!/usr/bin/env python3
"""Print the paper's Tables 1-5 as Markdown from the stored result JSONs.

The loaders and the table1()..table5() functions below are a VERBATIM excerpt of the manuscript builder
(build_docx.py) that produced the submitted tables; only the Markdown printer at the end is new.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
V = json.load(open(os.path.join(BRIEF, "P1_FLAT_CV_CLOSEOUT", "P1_FLAT_CV_VERIFICATION.json")))
TB = json.load(open(os.path.join(BRIEF, "TB_CALIBRATION_RF03", "TB_RESULTS.json")))
TA = json.load(open(os.path.join(BRIEF, "FASTTRACK_V23", "P1_TA_RESULTS.json")))
P2 = json.load(open(os.path.join(BRIEF, "FASTTRACK_V23", "P2_RUN", "P2_RESULTS.json")))

f3 = lambda x: f"{x:.3f}"
f4 = lambda x: f"{x:.4f}"
n0 = lambda x: f"{x:,}"


def table1():
    hdr = ["Config", "AP Tue", "AP Wed", "AP Thu", "AP Fri", "mAP_day"]
    rows = [["prevalence"] + [f4(V["prevalence"][d]) for d in ("Tue", "Wed", "Thu", "Fri")] + [""]]
    for r in V["ranking"]:
        rows.append([r["config"]] + [f4(r["ap"][d]) for d in ("Tue", "Wed", "Thu", "Fri")] + [f4(r["mAP_day"])])
    return hdr, rows


def table2():
    hdr = ["Held-out day", "τ (α=0.01)", "FPR", "recall", "precision", "alerts/day", "τ (α=0.05)", "FPR", "recall", "precision"]
    rows = []
    for d in ("Mon", "Tue", "Wed", "Thu", "Fri"):
        a, b = TB["per_outer"][d]["primary"], TB["per_outer"][d]["sensitivity"]
        rc = lambda e: "undef." if e["recall"] is None else f3(e["recall"])
        pr = lambda e: "—" if e["recall"] is None else f3(e["precision"])
        rows.append([d, f4(a["tau"]), f4(a["achieved_fpr"]), rc(a), pr(a), n0(a["alerts_day"]), f4(b["tau"]), f4(b["achieved_fpr"]), rc(b), pr(b)])
    return hdr, rows


def table3():
    hdr = ["P2 day", "AP (prevalence)", "ROC-AUC", "α", "achieved FPR", "recall", "precision", "alerts/day", "max alerts/h"]
    rows = []
    for d, lab in (("Wed14", "14-02 (SSH)"), ("Fri16", "16-02 (DoS Hulk)"), ("P2_pooled", "Pooled")):
        R = P2["days"][d]
        for i, a in enumerate(("0.01", "0.05")):
            t = R["tb"][a]
            rows.append([lab if i == 0 else "", f"{R['ap']:.3f} ({R['prevalence']:.3f})" if i == 0 else "", f3(R["roc_auc"]) if i == 0 else "",
                         a, f3(t["achieved_fpr"]), f3(t["recall"]), f3(t["precision"]), n0(t["alerts_day"]), n0(t["alerts_per_hour_max"])])
    return hdr, rows


def table4():
    hdr = ["Day", "Budget", "alerts", "recall", "max recall permitted", "precision", "FPR"]
    rows = []
    for d in ("Tue", "Wed", "Thu", "Fri"):
        for lab, k in (("q = 0.01", "primary_q"), ("K = 500", "sensitivity_K")):
            e = TA["days"][d][k]
            rows.append([f"P1 {d}", lab, n0(e["alerts"]), f"{e['recall_flow']:.4f}", f"{e['max_recall_given_quota']:.4f}", f3(e["precision"]), f4(e["fpr"])])
    for d, lab in (("Wed14", "P2 14-02"), ("Fri16", "P2 16-02")):
        for bl, k in (("q = 0.01", "ta_q"), ("K = 500", "ta_K")):
            e = P2["days"][d][k]
            rows.append([lab, bl, n0(e["alerts"]), f"{e['recall_flow']:.4f}", f"{e['max_recall_given_quota']:.4f}", f3(e["precision"]), f4(e["fpr"])])
    return hdr, rows


def table5():
    hdr = ["P2 day", "b / hour", "labels/day", "quantity (census)", "bias", "RMSE", "95% CI coverage", "mean CI width"]
    rows = []
    for d, lab in (("Wed14", "14-02"), ("Fri16", "16-02")):
        for b in ("b=100", "b=250", "b=500"):
            E = P2["days"][d]["estimation"][b]
            for rule, m, name in (("tb_alpha_0_01", "fpr", "FPR, T-b"), ("tb_alpha_0_01", "recall", "recall, T-b"), ("ta_q", "recall", "recall, T-a q")):
                x = E[rule][m]
                if d == "Wed14" and rule == "tb_alpha_0_01" and m == "recall":
                    continue                                   # census recall 1.000, estimated exactly (stated in text)
                rows.append([lab, b[2:], n0(E[rule]["labels_per_day"]), f"{name} ({x['census']:.4f})", f"{x['bias']:+.4f}", f"{x['rmse']:.4f}",
                             f"{x['coverage_95']:.3f}", f"{x['mean_ci_width']:.4f}"])
    return hdr, rows


def md(hdr, rows):
    out = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


if __name__ == '__main__':
    for i, fn in enumerate((table1, table2, table3, table4, table5), 1):
        print(f'\n### Table {i}\n'); print(md(*fn()))
