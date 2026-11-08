#!/usr/bin/env python3
"""Synthetic source fixtures for G1 tests: five synthetic day CSVs (frozen 91-column header), a synthetic
allow-list (P1 entries + one forbidden P2 file) and a matching non-P1 SourceProfile. No dataset file is opened."""
import csv
import json
import os

import numpy as np

import pilot_guarded_runner as PR

R, S, C1 = PR.R, PR.S, PR.R.C1
CON = R.load_contract_verified()
SCHEMA = CON["schema"]
POS = {c: j for j, c in enumerate(SCHEMA)}
HEADER = S.load_expected_header()
GIB = 1024 ** 3


def synth(n, seed):
    r = np.random.default_rng(seed)
    X = np.abs(r.lognormal(3, 3, size=(n, 72)))
    for c in CON["integer"]:
        X[:, POS[c]] = np.floor(X[:, POS[c]])
    X[:, POS["Protocol"]] = r.choice([0, 1, 6, 17], n)
    X[:, POS["ICMP Code"]] = r.choice([-1, 0, 3], n)
    X[:, POS["ICMP Type"]] = r.choice([-1, 0, 8], n)
    return X


def _fmt(v):
    return "inf" if v == np.inf else (str(int(v)) if float(v).is_integer() and abs(v) < 1e15 else repr(float(v)))


def write_day_csv(path, X, attempted, labels=None, ids=None, header=None):
    header = list(header or HEADER)
    n = X.shape[0]
    ids = list(range(1, n + 1)) if ids is None else ids
    labels = labels if labels is not None else ["DoS" if r % 3 == 0 else "BENIGN" for r in range(n)]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in range(n):
            row = []
            for h in header:
                if h in POS:
                    row.append(_fmt(X[r, POS[h]]))
                elif h == "id":
                    row.append(str(ids[r]))
                elif h == "Label":
                    row.append(labels[r])
                elif h == "Attempted Category":
                    row.append(f"{float(attempted[r]):.1f}")
                elif h in ("Flow ID", "Src IP", "Dst IP", "Timestamp"):
                    row.append(f"s{r}")
                else:
                    row.append("0")
            w.writerow(row)


class Fixture:
    def __init__(self, d, sizes=(60, 50, 70, 40, 80), extra_allowed=()):
        self.d, self.files, self.X, self.att = d, {}, {}, {}
        os.makedirs(d, exist_ok=True)
        al = {"P1": {}, "forbidden": {}}
        for k, (day, n) in enumerate(zip(S.DAYS, sizes)):
            X = synth(n, 200 + k)
            if day == "Tue":
                X[0, POS["Flow Bytes/s"]] = np.inf
            att = np.where(np.arange(n) % 7 == 3, 0, -1)
            p = os.path.join(d, f"{day}.csv"); write_day_csv(p, X, att)
            self.files[day], self.X[day], self.att[day] = p, X, att
            al["P1"][C1.sha256_file(p)] = dict(day=day, file=f"{day}.csv", bytes=os.path.getsize(p))
        for p, day in extra_allowed:                     # deliberately allow-listed manipulated files
            al["P1"][C1.sha256_file(p)] = dict(day=day, file=os.path.basename(p), bytes=os.path.getsize(p))
        q = os.path.join(d, "p2.csv"); write_day_csv(q, synth(10, 99), np.full(10, -1)); self.p2 = q
        al["forbidden"][C1.sha256_file(q)] = dict(partition="P2", file="p2.csv", bytes=os.path.getsize(q))
        self.al_path = os.path.join(d, "allow.json")
        with open(self.al_path, "w") as fh:
            json.dump(al, fh)
        self.al = C1.load_allowlist(self.al_path)
        self.profile = S.SourceProfile("TEST", self.al["_sha256"], tuple(zip(S.DAYS, sizes)),
                                       tuple((day, int(np.sum(self.att[day] != -1))) for day in S.DAYS))

    def paths(self, days=S.DAYS):
        return [self.files[d] for d in days]


def ok_probe(n=200):
    return [(60 * GIB, 12 * GIB)] * n


def runner(fx, work, phase="flat", config_ids=("SYN-A",), selected=None, probe=None, sources=None, **kw):
    kw.setdefault("poll_s", 0.05)
    return PR.PilotRunner(work, fx.al_path, sources or fx.paths(), fx.profile, phase=phase, config_ids=list(config_ids),
                          selected=selected, probe=ok_probe() if probe is None else probe, **kw)


def job(config_id="SYN-A", outer="Wed", inner=None, branch="LR", faults=()):
    return PR.JobSpec(config_id, S.FoldSpec(outer, inner), branch, "SYNTH-CENTROID", tuple(faults))
