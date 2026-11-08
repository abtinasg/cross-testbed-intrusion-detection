#!/usr/bin/env python3
"""Flat-CV job: ONE fit of a frozen F3 configuration on the V1 route, one outer day held out, primary target.

This is RG8_V22_A1_SMOKE/p1_smoke_job.job_lr01_smoke generalised from (LR-01, Tue) to any (F3 config, outer day).
The route, frozen calls and their order are unchanged:
  hashed load_day_block (training days) -> assemble_training_matrix -> fit_p1_final (R3/R4/F7) -> verified-artefact gate
  -> transform to an on-disk memmap -> hashed load + transform of the held-out day -> F3 estimator fit -> predict_proba.
Only differences from the A1 job: the scope check accepts the 60 flat keys, the authorization token is the Flat-CV
token, score files are named per key, and a held-out day without positives (Mon) returns ap_heldout = None.
Module-level so that the spawn worker can import it.
"""
import hashlib
import os
import time

AUTH_PREFIX = "GO-FLATCV-P1-"


def expected_token(policy_sha256):
    return AUTH_PREFIX + policy_sha256[:12]


def _write_once_npy(path, arr):
    import numpy as np
    with open(path, "xb") as fh:                     # never overwrites
        np.save(fh, arr, allow_pickle=False)
        fh.flush(); os.fsync(fh.fileno())
    os.chmod(path, 0o444)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return dict(file=os.path.basename(path), sha256=h.hexdigest(), dtype=str(arr.dtype), n=int(arr.shape[0]))


def score_files(config_id, outer):
    return {k: f"{k}_{config_id}_{outer}.npy" for k in ("scores", "labels", "row_ids")}


def job_flat_cv(spec):
    import gc
    import resource
    import warnings
    import numpy as np
    import psutil
    import threadpoolctl
    import pilot_runner_g2 as P2
    import rg8_guard as RG
    G1, S, R, B, A = P2.G1, P2.S, P2.R, P2.B, P2.A
    config_id, outer = spec["config_id"], spec["outer"]

    # ---- scope and authorization: before any data file is opened
    grid = B.load_bound_grid()
    if (config_id, outer) not in B.expected_plan(grid, "flat"):
        raise RuntimeError(f"Flat-CV scope: ({config_id}, {outer}) is not a frozen flat key")
    prof = S.SourceProfile(spec["profile"]["name"], spec["profile"]["allowlist_sha256"],
                           tuple(map(tuple, spec["profile"]["file_rows"])), tuple(map(tuple, spec["profile"]["attempted_rows"])))
    real = prof == S.P1_PROFILE or spec["profile"]["allowlist_sha256"] == S.P1_PROFILE.allowlist_sha256
    if real and spec.get("authorization") != expected_token(spec["policy_sha256"]):
        raise RuntimeError("NOT_AUTHORIZED: real P1 requires the Flat-CV authorization token")
    al = R.C1.load_allowlist(spec["allowlist_path"])
    if al["_sha256"] != prof.allowlist_sha256:
        raise RuntimeError("allow-list does not match the profile")

    proc = psutil.Process()
    t0 = time.perf_counter()
    marks = []

    def mark(stage):
        RG.stage(stage)
        gc.collect()
        marks.append(dict(stage=stage, t=round(time.perf_counter() - t0, 2), rss=proc.memory_info().rss,
                          maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

    con = R.load_contract_verified()
    cfg = grid[config_id]
    if cfg.config_sha256 != spec["config_sha256"]:
        raise RuntimeError(f"{config_id} config_sha256 differs from frozen F3")
    fold = S.FoldSpec(outer)
    paths = spec["paths"]
    load = lambda d: S.load_day_block(paths[d], al, con, "primary", prof)
    mark("start")
    blocks = [load(d) for d in fold.training_days()]
    mark("loaded_training_days")
    tm = S.assemble_training_matrix(blocks, fold, al, con, "primary", prof)
    del blocks
    mark("assembled")
    art = S.fit_p1_final(tm, con, cfg.branch, al, fold, spec["evidence_dir"], prof)
    mark("preprocessing_fitted")
    G1.require_verified_artifact(art, tm, fold, al, con, prof)
    ytr = G1.binary_target(tm.labels, tm.attempted, "primary").astype(np.int8)
    mm_path = os.path.join(spec["tmp_dir"], f"Xtr_{cfg.config_id}_{outer}.npy")
    out, _ = R.transform(tm.X, art, con, out_path=mm_path)
    out.flush(); del out
    del tm
    mark("train_transformed_to_disk_raw_released")
    tmp_bytes = os.path.getsize(mm_path)
    Xtr = np.load(mm_path, mmap_mode="r")
    val = load(outer)
    Xva, _ = R.transform(val.X, art, con)
    yva = G1.binary_target(val.labels, val.attempted, "primary").astype(np.int8)
    ids_va = np.asarray(val.ids)
    del val
    mark("validation_loaded_transformed")
    est = A.build_estimator(cfg)
    mark("estimator_constructed")
    tf = time.perf_counter()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        est.fit(Xtr, ytr)
    fit_s = time.perf_counter() - tf
    fit_warnings = sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in wlist})
    mark("model_fitted")
    if list(est.classes_) != [0, 1]:
        raise RuntimeError(f"classes_ {list(est.classes_)} != [0, 1]: predict_proba[:, 1] would not be the attack class")
    tp = time.perf_counter()
    p = np.ascontiguousarray(est.predict_proba(Xva)[:, 1], dtype=np.float64)
    pred_s = time.perf_counter() - tp
    mark("predicted")
    if not np.all(np.isfinite(p)) or p.shape[0] != yva.shape[0] or ids_va.shape[0] != yva.shape[0]:
        raise RuntimeError("non-finite scores or length mismatch")
    fn = score_files(config_id, outer)
    persisted = dict(scores=_write_once_npy(os.path.join(spec["score_dir"], fn["scores"]), p),
                     labels=_write_once_npy(os.path.join(spec["score_dir"], fn["labels"]), yva),
                     ids=_write_once_npy(os.path.join(spec["score_dir"], fn["row_ids"]), ids_va))
    mark("scores_persisted")
    from sklearn.metrics import average_precision_score
    n_pos = int(yva.sum()); n_neg = int(yva.size - n_pos)
    ap = float(average_precision_score(yva, p)) if n_pos else None
    it = getattr(est, "n_iter_", None)
    n_iter = None if it is None else int(np.max(it))
    max_iter = est.get_params().get("max_iter")
    converged = None if (n_iter is None or max_iter is None or cfg.family != "LR") else n_iter < int(max_iter)
    n_train, n_val, n_cols = int(Xtr.shape[0]), int(Xva.shape[0]), int(Xtr.shape[1])
    del est, Xtr, Xva, p
    mark("cleanup")
    os.remove(mm_path)
    return dict(config_id=cfg.config_id, config_sha256=cfg.config_sha256, family=cfg.family, outer=outer, branch=cfg.branch,
                profile=prof.name, n_train=n_train, n_val=n_val, n_model_columns=n_cols, n_pos=n_pos, n_neg=n_neg,
                ap_heldout=ap, prevalence=float(n_pos / max(1, n_pos + n_neg)), n_iter=n_iter, max_iter=max_iter,
                converged=converged, fit_warnings=fit_warnings, fit_s=round(fit_s, 2), predict_s=round(pred_s, 2),
                total_s=round(time.perf_counter() - t0, 1), marks=marks, temp_file_bytes=tmp_bytes, persisted=persisted,
                threadpools=[dict(api=x.get("internal_api"), n=x.get("num_threads")) for x in threadpoolctl.threadpool_info()])
