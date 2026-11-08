#!/usr/bin/env python3
"""RG8 preflight-only reachability probe (V21). Measures; never starts a worker, never trains, never reads P1 rows.

Run from Terminal.app with Safari, other browsers, V2Box and the Claude desktop app QUIT:
  <PINNED_INTERPRETER:anaconda3/bin/python3> -I -B rg8_preflight_probe.py
It samples available memory, pressure and swap every 5 s for 60 s, then runs the official RG8 preflight with its
bounded 300 s wait. It writes one new JSON file (never overwrites) in ./preflight_probes/ and prints the verdict.
"""
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import psutil  # noqa: E402
import rg8_guard as G  # noqa: E402

GIB = 1024 ** 3


def main():
    pol = G.load_policy()
    env = G.check_environment(pol)
    s = G.Sensors()
    series = []
    for i in range(13):
        series.append(dict(t=i * 5, available=s.available(), pressure=s.pressure(), swap_used=s.swap_used()))
        if i < 12:
            time.sleep(5)
    pf = G.preflight_with_wait(pol, os.path.expanduser("~/Desktop"))
    top = sorted(((p.info["memory_info"].rss if p.info["memory_info"] else 0, p.info["name"])
                  for p in psutil.process_iter(["name", "memory_info"])), reverse=True)[:12]
    out = dict(probe="RG8 preflight-only reachability probe (V21)", utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               host=platform.node(), policy_sha256=pol["_sha256"], environment=env, series=series, official_preflight=pf,
               top_processes_rss=[dict(rss=r, name=n) for r, n in top], training_started=False)
    d = os.path.join(HERE, "preflight_probes"); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"probe_{out['utc'].replace(':', '')}.json")
    with open(path, "x") as fh:
        json.dump(out, fh, indent=1)
    av = [x["available"] / GIB for x in series]
    print(f"available GiB min/max over 60 s: {min(av):.2f}/{max(av):.2f}; required {pol['preflight']['min_available_bytes'] / GIB:.2f}")
    print("OFFICIAL PREFLIGHT:", "PASS" if pf["ok"] else "FAIL", json.dumps(pf["checks"]))
    print("written:", path)


if __name__ == "__main__":
    main()
