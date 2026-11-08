#!/usr/bin/env python3
"""Test helper: a pilot parent that starts one synthetic guarded fit (sleeping in the fit stage) so that the test can
SIGKILL this parent mid-fit. Usage: python3 -I -B g2_crash_driver.py <json args>. Synthetic data only."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_runner_g2 as P2  # noqa: E402

S = P2.S


def main():
    a = json.loads(sys.argv[1])
    prof = S.SourceProfile(a["profile"]["name"], a["profile"]["allowlist_sha256"],
                           tuple(map(tuple, a["profile"]["file_rows"])), tuple(map(tuple, a["profile"]["attempted_rows"])))
    rn = P2.PilotRunnerG2(a["work"], a["al_path"], a["sources"], prof, phase="flat",
                          probe=[(60 * 1024 ** 3, 12 * 1024 ** 3)] * 10, poll_s=0.05)
    rn.run(P2.BoundJob(a["config_id"], S.FoldSpec(a["outer"]), (("fit", "sleep"),)))


if __name__ == "__main__":   # required: spawn re-imports the main module in the child
    main()
