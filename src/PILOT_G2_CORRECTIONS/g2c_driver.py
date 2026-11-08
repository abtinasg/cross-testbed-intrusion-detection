#!/usr/bin/env python3
"""Test helper (synthetic only): a separate pilot process for G2-C recovery/concurrency tests.
Usage: python3 -I -B g2c_driver.py <json>
  json: work, al_path, sources, profile, mode ("fit" | "construct"), config_id, outer, faults, time_limit_s, start_at
Prints one JSON result line. Probe values are synthetic (test profile only)."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_runner_g2c as C  # noqa: E402

S, P2 = C.S, C.P2


def main():
    a = json.loads(sys.argv[1])
    prof = S.SourceProfile(a["profile"]["name"], a["profile"]["allowlist_sha256"],
                           tuple(map(tuple, a["profile"]["file_rows"])), tuple(map(tuple, a["profile"]["attempted_rows"])))
    if a.get("start_at"):
        time.sleep(max(0.0, a["start_at"] - time.time()))
    out = dict(pid=os.getpid())
    try:
        rn = C.PilotRunnerG2C(a["work"], a["al_path"], a["sources"], prof, phase="flat",
                              probe=[(60 * 1024 ** 3, 12 * 1024 ** 3)] * 20, poll_s=0.05,
                              time_limit_s=a.get("time_limit_s", 3 * 3600))
        out["recovery_state"] = rn.recovery["state"]
        if a["mode"] == "fit":
            rec = rn.run(P2.BoundJob(a["config_id"], S.FoldSpec(a["outer"]), tuple(map(tuple, a.get("faults", [])))))
            out.update(status=rec["status"], failure_class=rec["failure_class"])
    except C.PilotStop as e:
        out["pilot_stop"] = str(e)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
