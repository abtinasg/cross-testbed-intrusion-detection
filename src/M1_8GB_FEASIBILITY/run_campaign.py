#!/usr/bin/env python3
"""Synthetic workload campaign for the 8 GiB M1 (one monitored child per fit, strictly sequential).
Usage: python3 -I -B run_campaign.py <work_dir> <results_jsonl> <scale> <runs>
  runs: comma list of CONFIG:VARIANT (e.g. HGB-01:V0,HGB-01:V1,LR-01:V1). Largest outer fold (Tue held out).
  scale 1.0 = exact P1 row counts (synthetic). Synthetic CSVs are generated once into <work_dir>/data_<scale>.
Failure policy: the campaign STOPS at the first non-OK run (no retry, no fewer rows, no other config/params);
the failure and its resource measurements are recorded.
"""
import dataclasses
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import feasibility_harness as H  # noqa: E402
import pilot_runner_g2 as P2  # noqa: E402
import synth_cic as Y  # noqa: E402

S, R, C1 = P2.S, P2.R, P2.R.C1


def prepare(work, scale):
    d = os.path.join(work, f"data_{scale}")
    meta_p = os.path.join(d, "meta.json")
    if os.path.exists(meta_p):
        with open(meta_p) as fh:
            return json.load(fh)
    os.makedirs(d, exist_ok=False)
    con = R.load_contract_verified(); hdr = S.load_expected_header()
    rows, att, paths, al = {}, {}, {}, {"P1": {}, "forbidden": {}}
    t0 = time.time()
    for k, (day, n) in enumerate(Y.P1_ROWS):
        n = max(1000, int(round(n * scale)))
        p = os.path.join(d, f"SYN_{day}.csv")
        b, na = Y.write_day_csv(p, n, 31 + k, con["schema"], con["integer"], hdr)
        rows[day], att[day], paths[day] = n, na, p
        al["P1"][C1.sha256_file(p)] = dict(day=day, file=os.path.basename(p), bytes=b)
        print(f"generated {day}: {n} rows, {b / 2**20:.0f} MiB, {time.time() - t0:.0f} s", flush=True)
    ap = os.path.join(d, "synthetic_allowlist.json")
    with open(ap, "w") as fh:
        json.dump(al, fh)
    a = C1.load_allowlist(ap)
    prof = S.SourceProfile("SYNTHETIC-M1", a["_sha256"], tuple((dd, rows[dd]) for dd in S.DAYS), tuple((dd, att[dd]) for dd in S.DAYS))
    meta = dict(scale=scale, paths=paths, allowlist_path=ap, profile=dataclasses.asdict(prof),
                csv_bytes={dd: os.path.getsize(paths[dd]) for dd in S.DAYS}, generation_s=round(time.time() - t0))
    with open(meta_p, "w") as fh:
        json.dump(meta, fh, indent=1)
    return meta


def main(work, results, scale, runs):
    meta = prepare(work, scale)
    tmp = os.path.join(work, "tmp"); os.makedirs(tmp, exist_ok=True)
    samples = results.replace(".jsonl", "_samples.jsonl")
    for item in runs:
        cid, var = item.split(":")
        ev = os.path.join(work, "evidence", f"{scale}_{cid}_{var}_{int(time.time())}"); os.makedirs(ev)
        spec = dict(config_id=cid, variant=var, outer="Tue", paths=meta["paths"], allowlist_path=meta["allowlist_path"],
                    profile=meta["profile"], evidence_dir=ev, tmp_dir=tmp)
        rec = H.run_monitored(H.job_full_route, (spec,), f"{scale}:{cid}:{var}", samples, disk_path=work)
        rec.update(scale=scale, csv_bytes_total=sum(meta["csv_bytes"].values()), evidence_bytes=sum(
            os.path.getsize(os.path.join(ev, f)) for f in os.listdir(ev)))
        with open(results, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        r = rec["result"] or {}
        print(f"{rec['run']}: {rec['status']} peak_rss={rec['peak_tree_rss'] / 2**30:.2f}GiB swap_growth={rec['swap_growth'] / 2**30:.2f}GiB "
              f"min_avail={rec['min_available'] / 2**30:.2f}GiB pressure={rec['max_pressure_level']} elapsed={rec['elapsed_s']}s "
              f"fit={r.get('fit_s')}s {rec['error'] or ''}", flush=True)
        if rec["status"] != "OK":
            print("CAMPAIGN STOPPED at first failure (no retry, no reduced rows, no parameter change)", flush=True)
            return 1
    print("CAMPAIGN COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4].split(",")))
