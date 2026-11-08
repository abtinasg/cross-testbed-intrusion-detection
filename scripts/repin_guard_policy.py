#!/usr/bin/env python3
"""Re-pin the resource-guard policies to THIS machine's interpreter (see docs/RELEASE_NOTES.md).

The frozen policies pin one interpreter path, and the code pins the SHA-256 of the policies and of the guard modules.
This script changes nothing scientific. It:
  1. writes os.path.realpath(sys.executable) into "executable" of the V20 and A1 policies (limits unchanged);
  2. replaces the pinned policy hashes in rg8_guard.py, rg8_guard_v23.py, run_flat_cv.py and run_p1_smoke.py;
  3. replaces the pinned guard-module hashes (V21_GUARD_SHA256, V23_GUARD_SHA256) that step 2 changed;
  4. prints the new authorization tokens.
Run it with the interpreter you will use for the experiments:   python -I -B scripts/repin_guard_policy.py
The numpy / scikit-learn / psutil / Python versions are NOT changed: they must equal the policy's "environment" block.
"""
import hashlib
import json
import os
import platform
import re
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
V20 = "RG8_V21/RG8_POLICY_PROPOSED.json"
A1 = "RG8_V22_A1_SMOKE/A1_SMOKE_POLICY.json"
G21, G23 = "RG8_V21/rg8_guard.py", "RG8_V23_EXIT_RACE/rg8_guard_v23.py"


def p(rel):
    return os.path.join(SRC, rel)


def sha(rel):
    return hashlib.sha256(open(p(rel), "rb").read()).hexdigest()


def sub_const(rel, name, new):
    text = open(p(rel)).read()
    pat = re.compile(rf'^({name} = ")[0-9a-f]{{64}}(")', re.M)
    if len(pat.findall(text)) != 1:
        raise SystemExit(f"{rel}: expected exactly one {name}")
    open(p(rel), "w").write(pat.sub(lambda m: m.group(1) + new + m.group(2), text))


def main():
    exe = os.path.realpath(sys.executable)
    for rel in (V20, A1):
        pol = json.load(open(p(rel)))
        env = pol["environment"]
        import numpy, sklearn, psutil
        got = (platform.python_version(), numpy.__version__, sklearn.__version__, psutil.__version__)
        want = (env["python"], env["numpy"], env["sklearn"], env["psutil"])
        if got != want:
            raise SystemExit(f"this interpreter has {got}, the policy requires {want}: install requirements.txt first")
        raw = open(p(rel)).read()
        new = re.sub(r'("executable": ")[^"]*(")', lambda m: m.group(1) + exe + m.group(2), raw, count=1)
        open(p(rel), "w").write(new)
    v20, a1 = sha(V20), sha(A1)
    for rel in (G21, G23):
        sub_const(rel, "POLICY_SHA256", v20)
    sub_const("FLAT_CV_P1/run_flat_cv.py", "V20_POLICY_SHA256", v20)
    sub_const("RG8_V22_A1_SMOKE/run_p1_smoke.py", "V20_POLICY_SHA256", v20)
    sub_const("RG8_V22_A1_SMOKE/run_p1_smoke.py", "A1_POLICY_SHA256", a1)
    g21, g23 = sha(G21), sha(G23)
    sub_const("FLAT_CV_P1/run_flat_cv.py", "V21_GUARD_SHA256", g21)
    sub_const("RG8_V22_A1_SMOKE/run_p1_smoke.py", "V21_GUARD_SHA256", g21)
    sub_const("FASTTRACK_V23/P2_RUN/run_p2.py", "V23_GUARD_SHA256", g23)
    print(f"interpreter      {exe}")
    print(f"V20 policy       {v20}\nA1 policy        {a1}\nrg8_guard.py     {g21}\nrg8_guard_v23.py {g23}")
    print(f"tokens: GO-A1-LR01-TUE-{a1[:12]}  GO-FLATCV-P1-{v20[:12]}  GO-TB-RF03-P1-{v20[:12]}  GO-P2-RF03-d1a6585ac71b")


if __name__ == "__main__":
    main()
