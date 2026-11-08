#!/usr/bin/env python3
"""Synthetic fixtures for G2 tests (built on G1 fixtures). Optional benign-only Monday. No dataset file is opened."""
import dataclasses
import json
import os

import numpy as np

import g1_fixtures as F
import pilot_runner_g2 as P2

S, C1 = P2.S, P2.R.C1
GIB = F.GIB


def fixture(d, mon_benign=False):
    fx = F.Fixture(d)
    if mon_benign:
        F.write_day_csv(fx.files["Mon"], fx.X["Mon"], fx.att["Mon"], labels=["BENIGN"] * fx.X["Mon"].shape[0])
        al = {"P1": {C1.sha256_file(fx.files[day]): dict(day=day, file=f"{day}.csv", bytes=os.path.getsize(fx.files[day])) for day in S.DAYS},
              "forbidden": {C1.sha256_file(fx.p2): dict(partition="P2", file="p2.csv", bytes=os.path.getsize(fx.p2))}}
        with open(fx.al_path, "w") as fh:
            json.dump(al, fh)
        fx.al = C1.load_allowlist(fx.al_path)
        fx.profile = dataclasses.replace(fx.profile, allowlist_sha256=fx.al["_sha256"])
    return fx


def runner(fx, work, phase="flat", selected=None, probe=None, **kw):
    kw.setdefault("poll_s", 0.05)
    return P2.PilotRunnerG2(work, fx.al_path, fx.paths(), fx.profile, phase=phase, selected=selected,
                            probe=F.ok_probe(2000) if probe is None else probe, **kw)


def job(config_id="LR-01", outer="Wed", inner=None, faults=()):
    return P2.BoundJob(config_id, S.FoldSpec(outer, inner), tuple(faults))
