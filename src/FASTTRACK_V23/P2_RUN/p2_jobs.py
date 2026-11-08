#!/usr/bin/env python3
"""P2 guarded jobs (module-level so the spawn worker can import them).

job_final_fit : ONE RF-03 fit on all five P1 days through the frozen route (load_day_block -> assemble_training_matrix ->
                fit_p1_final -> require_verified_artifact -> transform -> F3 estimator). The only addition is FinalFoldSpec, a
                subclass of the frozen FoldSpec whose training days are all five P1 days (no held-out day). Frozen files are
                imported unchanged. Persists the estimator (joblib) and the preprocessing artefact (pickle) with SHA-256.
job_score_p2  : LABEL-BLIND scoring of one P2 day. (1) streaming SHA-256 of the CSV = pinned hash; (2) header validated against
                the frozen header; (3) polars scan with column projection (id, Timestamp, 72 schema columns, all as raw text)
                -> label-free Parquet (the Label and Attempted Category columns are never selected or materialised);
                (4) the Parquet is read in batches; each distinct token is parsed with the frozen parse_token_strict;
                transform with the final artefact; predict_proba; T2 availability time; A1.2 tie key on the raw float64 features.
                Write-once outputs: scores, row ids, availability (us), tie keys.
"""
import dataclasses
import hashlib
import os
import pickle
import time

AUTH = "GO-P2-RF03-d1a6585ac71b"
SEL = "RF-03"


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _write_once_npy(path, arr):
    import numpy as np
    with open(path, "xb") as fh:
        np.save(fh, arr, allow_pickle=False)
        fh.flush(); os.fsync(fh.fileno())
    os.chmod(path, 0o444)
    return dict(file=os.path.basename(path), sha256=_sha(path), dtype=str(arr.dtype), n=int(arr.shape[0]))


def _write_once_bytes(path, data):
    with open(path, "xb") as fh:
        fh.write(data); fh.flush(); os.fsync(fh.fileno())
    os.chmod(path, 0o444)
    return dict(file=os.path.basename(path), sha256=_sha(path), bytes=len(data))


def final_fold_class(S):
    @dataclasses.dataclass(frozen=True)
    class FinalFoldSpec(S.FoldSpec):
        """All five P1 days, no held-out day (final source detector)."""
        outer: str = None
        inner: str = None

        def __post_init__(self):
            if self.outer is not None or self.inner is not None:
                raise S.PreprocessingError("FinalFoldSpec has no held-out day")

        def training_days(self):
            return tuple(S.DAYS)

        @property
        def fold_id(self):
            return "final=P1-all"
    return FinalFoldSpec


def job_final_fit(spec):
    import gc
    import warnings
    import numpy as np
    import joblib
    import pilot_runner_g2 as P2
    import rg8_guard as RG
    G1, S, R, B, A = P2.G1, P2.S, P2.R, P2.B, P2.A
    if spec.get("authorization") != AUTH:
        raise RuntimeError("NOT_AUTHORIZED")
    grid = B.load_bound_grid()
    cfg = grid[SEL]
    if cfg.config_sha256 != spec["config_sha256"]:
        raise RuntimeError("RF-03 config_sha256 differs from frozen F3")
    prof = S.P1_PROFILE
    al = R.C1.load_allowlist(spec["allowlist_path"])
    if al["_sha256"] != prof.allowlist_sha256:
        raise RuntimeError("allow-list does not match the P1 profile")
    con = R.load_contract_verified()
    fold = final_fold_class(S)()
    t0 = time.perf_counter()
    RG.stage("start")
    blocks = [S.load_day_block(spec["paths"][d], al, con, "primary", prof) for d in fold.training_days()]
    RG.stage("loaded_all_p1_days")
    tm = S.assemble_training_matrix(blocks, fold, al, con, "primary", prof)
    del blocks; gc.collect()
    art = S.fit_p1_final(tm, con, cfg.branch, al, fold, spec["evidence_dir"], prof)
    G1.require_verified_artifact(art, tm, fold, al, con, prof)
    RG.stage("preprocessing_fitted")
    ytr = G1.binary_target(tm.labels, tm.attempted, "primary").astype(np.int8)
    mm = os.path.join(spec["tmp_dir"], "Xtr_final.npy")
    out, cols = R.transform(tm.X, art, con, out_path=mm)
    out.flush(); del out
    n_train, x_sha = int(tm.X.shape[0]), tm.x_sha256
    n_pos = int(ytr.sum())
    del tm; gc.collect()
    Xtr = np.load(mm, mmap_mode="r")
    est = A.build_estimator(cfg)
    tf = time.perf_counter()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        est.fit(Xtr, ytr)
    fit_s = time.perf_counter() - tf
    RG.stage("model_fitted")
    if list(est.classes_) != [0, 1]:
        raise RuntimeError("classes_ != [0, 1]")
    n_cols = int(Xtr.shape[1])
    del Xtr
    os.remove(mm)
    mpath = os.path.join(spec["model_dir"], "RF-03_P1all.joblib")
    if os.path.exists(mpath):
        raise RuntimeError("final model file exists; never overwritten")
    joblib.dump(est, mpath)
    os.chmod(mpath, 0o444)
    apath = os.path.join(spec["model_dir"], "RF-03_P1all_preprocessing_artefact.pkl")
    art_meta = _write_once_bytes(apath, pickle.dumps(art, protocol=5))
    RG.stage("persisted")
    return dict(config_id=SEL, config_sha256=cfg.config_sha256, fold_id=fold.fold_id, training_days=list(fold.training_days()),
                n_train=n_train, n_pos=n_pos, train_x_sha256=x_sha, n_model_columns=n_cols, model_columns=list(cols),
                artefact_fold_id=art.get("fold_id"), artefact_partition=art.get("partition"), artefact_spec=art.get("r5_spec"),
                model=dict(file=os.path.basename(mpath), sha256=_sha(mpath), bytes=os.path.getsize(mpath)), artefact=art_meta,
                fit_s=round(fit_s, 1), total_s=round(time.perf_counter() - t0, 1),
                fit_warnings=sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in wl}))


def job_score_p2(spec):
    import sys
    import numpy as np
    import polars as pl
    import pyarrow.parquet as pq
    import joblib
    import pilot_runner_g2 as P2
    import rg8_guard as RG
    S, R = P2.S, P2.R
    sys.path.insert(0, spec["a1_dir"])
    import cv_threshold_reference as A1
    if spec.get("authorization") != AUTH:
        raise RuntimeError("NOT_AUTHORIZED")
    t0 = time.perf_counter()
    RG.stage("start")
    if _sha(spec["model_path"]) != spec["model_sha256"] or _sha(spec["artefact_path"]) != spec["artefact_sha256"]:
        raise RuntimeError("final model or artefact differs from the sealed final-fit record")
    if _sha(spec["csv_path"]) != spec["csv_sha256"]:
        raise RuntimeError("P2 CSV SHA-256 differs from the pinned hash")
    RG.stage("hashes_verified")
    con = R.load_contract_verified()
    schema = list(con["schema"])
    import csv
    with open(spec["csv_path"], newline="") as fh:
        header = next(csv.reader(fh))
    S.REF.validate_header(header, S.load_expected_header())
    keep = ["id", "Timestamp"] + schema
    forbidden = {"Label", "Attempted Category"}
    if forbidden & set(keep):
        raise RuntimeError("label column in projection")
    pq_path = spec["parquet_path"]
    if os.path.exists(pq_path):
        raise RuntimeError("label-free parquet exists; never overwritten")
    pl.scan_csv(spec["csv_path"], infer_schema=False).select(keep).sink_parquet(pq_path)
    os.chmod(pq_path, 0o444)
    pf = pq.ParquetFile(pq_path)
    if set(pf.schema_arrow.names) != set(keep) or forbidden & set(pf.schema_arrow.names):
        raise RuntimeError("label-free parquet schema mismatch")
    n = pf.metadata.num_rows
    if n != spec["expected_rows"]:
        raise RuntimeError(f"P2 rows {n} != pinned {spec['expected_rows']}")
    pq_sha = _sha(pq_path)
    RG.stage("label_free_parquet_written")
    est = joblib.load(spec["model_path"])
    with open(spec["artefact_path"], "rb") as fh:
        art = pickle.load(fh)
    if art.get("fold_id") != "final=P1-all":
        raise RuntimeError("artefact is not the final P1 artefact")
    scores = np.empty(n, np.float64); ids = np.empty(n, np.int64); avail = np.empty(n, np.int64); keys = np.empty(n, np.uint64)
    pos = 0
    n_tokens_parsed = 0
    for batch in pf.iter_batches(batch_size=250_000, columns=keep):
        df = pl.from_arrow(batch)
        m = df.height
        bid = df["id"].cast(pl.Int64, strict=True).to_numpy()
        if not np.array_equal(bid, np.arange(pos + 1, pos + m + 1)):
            raise RuntimeError("P2 id != file row (file order not preserved)")
        X = np.empty((m, len(schema)), np.float64)
        for j, c in enumerate(schema):
            s = df[c]
            if s.null_count():
                raise RuntimeError(f"empty token in {c!r}")
            u = s.unique()
            vals = [R.C1.parse_token_strict(t, c, con) for t in u.to_list()]
            n_tokens_parsed += len(vals)
            X[:, j] = s.replace_strict(u, pl.Series(vals, dtype=pl.Float64), return_dtype=pl.Float64).to_numpy()
        ts = df["Timestamp"].str.to_datetime("%Y-%m-%d %H:%M:%S%.f", time_unit="us", strict=True).cast(pl.Int64).to_numpy()
        dur = X[:, schema.index("Flow Duration")]
        if np.any(dur != np.floor(dur)) or np.any(dur < 0):
            raise RuntimeError("Flow Duration not a non-negative integer")
        a = A1.availability_time(ts, dur.astype(np.int64), "T2")
        Xt, _ = R.transform(X, art, con)
        p = est.predict_proba(Xt)[:, 1]
        if not np.all(np.isfinite(p)):
            raise RuntimeError("non-finite score")
        scores[pos:pos + m] = p; ids[pos:pos + m] = bid; avail[pos:pos + m] = a
        keys[pos:pos + m] = A1.tie_key(X, a)
        pos += m
        RG.stage(f"scored_{pos}")
    if pos != n:
        raise RuntimeError("row count mismatch after scoring")
    d = spec["day"]
    out = spec["score_dir"]
    persisted = dict(scores=_write_once_npy(os.path.join(out, f"scores_P2_{d}.npy"), scores),
                     ids=_write_once_npy(os.path.join(out, f"row_ids_P2_{d}.npy"), ids),
                     availability_us=_write_once_npy(os.path.join(out, f"availability_us_P2_{d}.npy"), avail),
                     tie_keys=_write_once_npy(os.path.join(out, f"tie_keys_P2_{d}.npy"), keys))
    RG.stage("persisted")
    return dict(day=d, n_rows=int(n), csv_sha256=spec["csv_sha256"], label_free_parquet=dict(file=os.path.basename(pq_path), sha256=pq_sha,
                columns=keep), model_sha256=spec["model_sha256"], artefact_sha256=spec["artefact_sha256"], persisted=persisted,
                n_distinct_scores=int(np.unique(scores).size), n_distinct_tokens_parsed=n_tokens_parsed,
                label_columns_read=False, total_s=round(time.perf_counter() - t0, 1))
