#!/usr/bin/env python3
"""Read-only seal verification BEFORE any P2 label is read. Writes SEAL_VERIFICATION.json."""
import hashlib, json, os, sys
import numpy as np, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); FT = os.path.dirname(HERE)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
P = []
seal_p = os.path.join(HERE, "P2_SCORES_SEALED.json"); seal = json.load(open(seal_p))
if sha(seal_p) != "6b5d12ef166f95c55b9a1e52029ce1f3e6829d9dd34af3e0ff1913b646bff856": P.append("seal hash")
pre = json.load(open(os.path.join(FT, "P2_PREREGISTRATION.json")))
if seal["preregistration_sha256"] != sha(os.path.join(FT, "P2_PREREGISTRATION.json")): P.append("prereg hash")
L = [json.loads(l) for l in open(os.path.join(HERE, "p2_run", "LEDGER.jsonl"))]
if seal["ledger_sha256"] != sha(os.path.join(HERE, "p2_run", "LEDGER.jsonl")): P.append("ledger changed after seal")
fin = [r for r in L if r["key"] == ["FINAL_FIT"] and r["status"] == "OK"][0]["detail"]
for k in ("model", "artefact"):
    if seal["final_fit"][k] != fin[k] or sha(os.path.join(HERE, "p2_run", "final_model", fin[k]["file"])) != fin[k]["sha256"]: P.append(f"final {k}")
if fin["config_sha256"] != "8a93f95e96226ac2179c6ac658328d26d8caaca14807bdce6630e8fb89be2c2f": P.append("config")
rows = {}
for d, meta in seal["days"].items():
    if meta["csv_sha256"] != pre["p2_files"][d]: P.append(f"{d} csv not pre-registered")
    rec = [r for r in L if r["key"] == ["SCORE", d] and r["status"] == "OK"][-1]["detail"]
    if rec["model_sha256"] != fin["model"]["sha256"] or rec["artefact_sha256"] != fin["artefact"]["sha256"]: P.append(f"{d} model link")
    if rec["label_columns_read"] is not False: P.append(f"{d} label flag")
    pqp = os.path.expanduser(f"~/Desktop/IDS_RAW_DATA_STAGE02A0/work/p2_run/{meta['label_free_parquet']['file']}")
    f = pq.ParquetFile(pqp)
    if sha(pqp) != meta["label_free_parquet"]["sha256"] or {"Label", "Attempted Category"} & set(f.schema_arrow.names) or len(f.schema_arrow.names) != 74:
        P.append(f"{d} label-free parquet")
    v = {}
    for k, m in meta["persisted"].items():
        p = os.path.join(HERE, "p2_run", "scores", m["file"])
        if sha(p) != m["sha256"]: P.append(f"{m['file']} hash")
        v[k] = np.load(p)
    n = meta["n_rows"]
    if not (all(x.shape == (n,) for x in v.values()) and np.array_equal(v["ids"], np.arange(1, n + 1))): P.append(f"{d} rows/ids")
    if not (np.all(np.isfinite(v["scores"])) and v["scores"].min() >= 0 and v["scores"].max() <= 1): P.append(f"{d} scores")
    a = v["availability_us"] // 10 ** 6
    rows[d] = dict(n=n, n_distinct_scores=int(np.unique(v["scores"]).size), first_avail=str(np.datetime64(int(a.min()), "s")),
                   last_avail=str(np.datetime64(int(a.max()), "s")), n_hours=int(np.unique(v["availability_us"] // 3600_000_000).size))
rv = seal["recovery"]
orph = os.path.join(HERE, "p2_run", "attempt1_orphans")
ow = os.path.expanduser("~/Desktop/IDS_RAW_DATA_STAGE02A0/work/p2_run/attempt1_orphans")
for f_, h in rv["attempt1_unverified_outputs_moved_sha256"].items():
    p = os.path.join(orph if f_.endswith(".npy") else ow, f_)
    if sha(p) != h: P.append(f"orphan {f_}")
out = dict(status="PASS" if not P else "FAIL", problems=P, seal_sha256=sha(seal_p), days=rows, guard_by_key=seal["guard_by_key"],
           recovery_token=rv["token"], no_label_column_in_scoring_inputs=not any("label" in p for p in P))
json.dump(out, open(os.path.join(HERE, "SEAL_VERIFICATION.json"), "w"), indent=1)
print(json.dumps(out, indent=1)); sys.exit(0 if not P else 1)
