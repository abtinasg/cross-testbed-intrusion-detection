#!/usr/bin/env python3
"""T-b inner job: ONE inner fit of the flat-selected configuration RF-03 (CV v2 section 4, step 1).

Identical route to FLAT_CV_P1/p1_cv_job.job_flat_cv (same frozen calls in the same order). The only differences:
  * fold = FoldSpec(outer=k, inner=j): training days = D minus {k, j} (3 days); scored day = j (a training day of outer k);
  * scope = the 20 frozen T-b keys of RF-03 (f3_binding.expected_plan(grid, "tb", "RF-03"));
  * authorization token prefix GO-TB-RF03-P1-;
  * score files are named per (outer, inner).
The outer held-out day k is never opened by this job (its labels are never read for T-b).
"""
import hashlib
import os
import time

AUTH_PREFIX = "GO-TB-RF03-P1-"
SELECTED = "RF-03"


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


def score_files(outer, inner):
    return {k: f"{k}_{SELECTED}_outer={outer}_inner={inner}.npy" for k in ("scores", "labels", "row_ids")}


def job_tb_inner(spec):
    import gc
    import resource
    import warnings
    import numpy as np
    import psutil
    import threadpoolctl
    import pilot_runner_g2 as P2
    import rg8_guard as RG
    G1, S, R, B, A = P2.G1, P2.S, P2.R, P2.B, P2.A
    config_id, outer, inner = spec["config_id"], spec["outer"], spec["inner"]

    # ---- scope and authorization: before any data file is opened
    grid = B.load_bound_grid()
    if config_id != SELECTED or (config_id, f"{outer}/{inner}") not in B.expected_plan(grid, "tb", SELECTED):
        raise RuntimeError(f"T-b scope: ({config_id}, {outer}/{inner}) is not a frozen T-b key of {SELECTED}")
    prof = S.SourceProfile(spec["profile"]["name"], spec["profile"]["allowlist_sha256"],
                           tuple(map(tuple, spec["profile"]["file_rows"])), tuple(map(tuple, spec["profile"]["attempted_rows"])))
    real = prof == S.P1_PROFILE or spec["profile"]["allowlist_sha256"] == S.P1_PROFILE.allowlist_sha256
    if real and spec.get("authorization") != expected_token(spec["policy_sha256"]):
        raise RuntimeError("NOT_AUTHORIZED: real P1 T-b requires the T-b authorization token")
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
    fold = S.FoldSpec(outer, inner)
    if outer in fold.training_days() or inner in fold.training_days() or len(fold.training_days()) != 3:
        raise RuntimeError("T-b fold must train on exactly the 3 days D minus {outer, inner}")
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
    mm_path = os.path.join(spec["tmp_dir"], f"Xtr_{cfg.config_id}_{outer}_{inner}.npy")
    out, _ = R.transform(tm.X, art, con, out_path=mm_path)
    out.flush(); del out
    del tm
    mark("train_transformed_to_disk_raw_released")
    tmp_bytes = os.path.getsize(mm_path)
    Xtr = np.load(mm_path, mmap_mode="r")
    val = load(inner)                                   # inner held-out day j; the outer day k is never loaded
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
        raise RuntimeError(f"classes_ {list(est.classes_)} != [0, 1]")
    tp = time.perf_counter()
    p = np.ascontiguousarray(est.predict_proba(Xva)[:, 1], dtype=np.float64)
    pred_s = time.perf_counter() - tp
    mark("predicted")
    if not np.all(np.isfinite(p)) or p.shape[0] != yva.shape[0] or ids_va.shape[0] != yva.shape[0]:
        raise RuntimeError("non-finite scores or length mismatch")
    fn = score_files(outer, inner)
    persisted = dict(scores=_write_once_npy(os.path.join(spec["score_dir"], fn["scores"]), p),
                     labels=_write_once_npy(os.path.join(spec["score_dir"], fn["labels"]), yva),
                     ids=_write_once_npy(os.path.join(spec["score_dir"], fn["row_ids"]), ids_va))
    mark("scores_persisted")
    from sklearn.metrics import average_precision_score
    n_pos = int(yva.sum()); n_neg = int(yva.size - n_pos)
    ap = float(average_precision_score(yva, p)) if n_pos else None
    n_train, n_val, n_cols = int(Xtr.shape[0]), int(Xva.shape[0]), int(Xtr.shape[1])
    del est, Xtr, Xva, p
    mark("cleanup")
    os.remove(mm_path)
    return dict(config_id=cfg.config_id, config_sha256=cfg.config_sha256, family=cfg.family, outer=outer, inner=inner,
                fold_id=fold.fold_id, training_days=list(fold.training_days()), branch=cfg.branch, profile=prof.name,
                n_train=n_train, n_val=n_val, n_model_columns=n_cols, n_pos=n_pos, n_neg=n_neg, ap_inner=ap,
                prevalence=float(n_pos / max(1, n_pos + n_neg)), fit_warnings=fit_warnings, fit_s=round(fit_s, 2),
                predict_s=round(pred_s, 2), total_s=round(time.perf_counter() - t0, 1), marks=marks,
                temp_file_bytes=tmp_bytes, persisted=persisted,
                threadpools=[dict(api=x.get("internal_api"), n=x.get("num_threads")) for x in threadpoolctl.threadpool_info()])
