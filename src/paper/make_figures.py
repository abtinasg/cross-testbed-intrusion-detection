#!/usr/bin/env python3
"""Figures for the submission, drawn ONLY from stored, verified result files (no recomputation of any metric).
Fig. 1: achieved FPR under the frozen T-b thresholds (alpha 0.01, 0.05): P1 held-out days vs P2 days.
Fig. 2: flow-level recall under hourly alert budgets (q = 0.01, K = 500) with the maximum recall each quota permits.
Sources: TB_CALIBRATION_RF03/TB_RESULTS.json, FASTTRACK_V23/P1_TA_RESULTS.json, FASTTRACK_V23/P2_RUN/P2_RESULTS.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
TB = json.load(open(os.path.join(BRIEF, "TB_CALIBRATION_RF03", "TB_RESULTS.json")))
TA = json.load(open(os.path.join(BRIEF, "FASTTRACK_V23", "P1_TA_RESULTS.json")))
P2 = json.load(open(os.path.join(BRIEF, "FASTTRACK_V23", "P2_RUN", "P2_RESULTS.json")))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
C1, C2, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8.5,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})


def fig1():
    cats = ["Mon", "Tue", "Wed", "Thu", "Fri", "14-02\n(SSH)", "16-02\n(DoS Hulk)"]
    f01 = [TB["per_outer"][d]["primary"]["achieved_fpr"] for d in ("Mon", "Tue", "Wed", "Thu", "Fri")] + \
          [P2["days"][d]["tb"]["0.01"]["achieved_fpr"] for d in ("Wed14", "Fri16")]
    f05 = [TB["per_outer"][d]["sensitivity"]["achieved_fpr"] for d in ("Mon", "Tue", "Wed", "Thu", "Fri")] + \
          [P2["days"][d]["tb"]["0.05"]["achieved_fpr"] for d in ("Wed14", "Fri16")]
    fig, ax = plt.subplots(figsize=(6.3, 2.9), dpi=300)
    x = list(range(len(cats)))
    ax.axhline(0.01, color=C1, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0.05, color=C2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(4.58, 0.01 * 1.1, "nominal α = 0.01", color=MUTED, fontsize=7.5, va="bottom")
    ax.text(4.58, 0.05 * 1.1, "nominal α = 0.05", color=MUTED, fontsize=7.5, va="bottom")
    ax.scatter([i - 0.09 for i in x], f01, s=34, marker="o", color=C1, edgecolor="white", linewidth=1.2, zorder=3, label="α = 0.01 (primary)")
    ax.scatter([i + 0.09 for i in x], f05, s=34, marker="s", color=C2, edgecolor="white", linewidth=1.2, zorder=3, label="α = 0.05 (sensitivity)")
    for i in (5, 6):
        ax.annotate(f"{f01[i]:.3f}", (i - 0.09, f01[i]), xytext=(0, -7), textcoords="offset points", ha="center", va="top", fontsize=7.5, color=INK)
        ax.annotate(f"{f05[i]:.3f}", (i + 0.09, f05[i]), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.axvline(4.5, color=MUTED, lw=0.8)
    ax.set_yscale("log")
    ax.set_ylim(0.004, 0.9)
    ax.set_xlim(-0.5, 6.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Achieved false-positive rate (log scale)")
    ax.yaxis.grid(True, color=GRID, lw=0.6, which="major")
    ax.set_axisbelow(True)
    ax.text(2.0, 0.68, "Source: held-out CIC-IDS2017 days", ha="center", fontsize=8, color=INK)
    ax.text(5.6, 0.68, "Target: CSE-CIC-IDS2018", ha="center", fontsize=8, color=INK)
    ax.legend(loc="upper left", bbox_to_anchor=(0.04, 0.9), frameon=False, fontsize=7.5, handletextpad=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_tb_fpr_p1_vs_p2.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


def fig2():
    cats = ["Tue", "Wed", "Thu", "Fri", "14-02\n(SSH)", "16-02\n(DoS Hulk)"]
    q = [TA["days"][d]["primary_q"] for d in ("Tue", "Wed", "Thu", "Fri")] + [P2["days"][d]["ta_q"] for d in ("Wed14", "Fri16")]
    k = [TA["days"][d]["sensitivity_K"] for d in ("Tue", "Wed", "Thu", "Fri")] + [P2["days"][d]["ta_K"] for d in ("Wed14", "Fri16")]
    rq = [e["recall_flow"] for e in q]; rk = [e["recall_flow"] for e in k]
    mq = [e["max_recall_given_quota"] for e in q]; mk = [e["max_recall_given_quota"] for e in k]
    fig, ax = plt.subplots(figsize=(6.3, 2.9), dpi=300)
    w = 0.34
    for j, (vals, caps, col, lab, off) in enumerate(((rq, mq, C1, "q = 0.01 (relative quota)", -w / 2 - 0.01),
                                                    (rk, mk, C2, "K = 500 alerts/h (absolute cap)", w / 2 + 0.01))):
        xs = [i + off for i in range(len(cats))]
        ax.bar(xs, vals, width=w, color=col, edgecolor="white", linewidth=1, label=lab, zorder=2)
        ax.scatter(xs, caps, marker="_", s=160, color=INK, linewidths=1.6, zorder=3, label="maximum recall the quota permits" if j == 0 else None)
        for xv, v, c in zip(xs, vals, caps):
            ax.annotate("0" if v == 0 else (f"{v:.3f}" if v >= 0.001 else f"{v:.4f}"), (xv, max(v, c)), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color=INK, rotation=90)
    ax.axvline(3.5, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 0.36)
    ax.set_ylabel("Flow-level recall under budget")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.text(1.5, 0.335, "Source: held-out CIC-IDS2017 days", ha="center", fontsize=8, color=INK)
    ax.text(4.5, 0.335, "Target: CSE-CIC-IDS2018", ha="center", fontsize=8, color=INK)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), frameon=False, fontsize=7.5, handletextpad=0.4)
    fig.tight_layout()
    p = os.path.join(OUT, "fig2_budget_recall_p1_vs_p2.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


if __name__ == "__main__":
    print(fig1()); print(fig2())
