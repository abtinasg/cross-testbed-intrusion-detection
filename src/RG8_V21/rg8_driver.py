#!/usr/bin/env python3
"""Test driver: one PilotRunnerRG8 fit in a separate process (so the test can SIGKILL the parent). Synthetic only.
Usage: python3 -I -B rg8_driver.py '<json args>'"""
import dataclasses
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rg8_runner as RR  # noqa: E402

S, P2 = RR.S, RR.P2


def main(a):
    prof = S.SourceProfile(**{k: (tuple(map(tuple, v)) if isinstance(v, list) else v) for k, v in a["profile"].items()})
    r = RR.PilotRunnerRG8(a["work"], a["al_path"], a["sources"], prof, phase="flat",
                          _unpinned_test_policy_path=a["policy_path"])
    job = P2.BoundJob(a["config_id"], S.FoldSpec(a["outer"], None), tuple(map(tuple, a["faults"])))
    rec = r.run(job)
    print(json.dumps(dict(status=rec["status"], failure_class=rec.get("failure_class"))), flush=True)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
