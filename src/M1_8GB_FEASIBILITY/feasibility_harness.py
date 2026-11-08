#!/usr/bin/env python3
"""8 GiB M1 resource-feasibility harness (synthetic only; NO real data, NO project labels, NO real IDS model).

job_full_route(spec) runs, in ONE monitored child process, the complete frozen memory path on synthetic CSVs:
  P1-style hashed load_day_block -> assemble_training_matrix -> fit_p1_final (R3/R4/F7 R5) -> verified-artefact gate
  -> transform -> F3 estimator constructed by the G2 adapter (build_estimator; all 12 pinned configs) -> fit on
  SYNTHETIC features with SYNTHETIC labels (no signal) -> predict_proba on the synthetic validation day -> cleanup.
Variants (identical numbers by construction; equivalence is tested in test_m1_equivalence.py):
  V0 = the current G2 child order (all needed days loaded, training+validation outputs in RAM, float64 to RF).
  V1 = memory-ordered: validation day loaded only after the training raw matrix is released; gate before transform;
       training output written to a disk-backed .npy and re-opened read-only; raw TrainingMatrix released before the
       model fit; RF receives the float32 Fortran array that scikit-learn would otherwise create internally.
run_monitored(...) samples the child process tree (RSS, CPU, threads) and the system (available memory, swap,
macOS memory-pressure level, free disk) and kills the child on the UNCHANGED 6.5 GiB RSS limit, the time limit, or
sustained CRITICAL memory pressure (measurement safety stop). It never retries and never reduces rows.
"""
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
for _p in (os.path.join(BRIEF, "PILOT_BINDING_G2"), os.path.join(BRIEF, "PILOT_INTEGRATION_G1"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GIB = 1024 ** 3


def pressure_level():
    try:
        return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True,
                                  text=True, timeout=5).stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def swap_used():
    return psutil.swap_memory().used


def host_snapshot(path_for_disk):
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    out = dict(total=vm.total, available=vm.available, wired=getattr(vm, "wired", None), active=vm.active,
               inactive=vm.inactive, free=vm.free, swap_total=sw.total, swap_used=sw.used,
               pressure_level=pressure_level(), disk_free=shutil.disk_usage(path_for_disk).free,
               cpu_logical=psutil.cpu_count(), load_avg=os.getloadavg())
    try:
        vs = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = 16384
        get = lambda k: int([l for l in vs.splitlines() if l.startswith(k)][0].split(":")[1].strip(" ."))
        out.update(compressor_occupied=get("Pages occupied by compressor") * page,
                   compressor_stored=get("Pages stored in compressor") * page)
    except Exception:  # noqa: BLE001
        pass
    return out


# ----------------------------------------------------------------------------- child job
def job_full_route(spec):
    import gc
    import resource
    import numpy as np
    import threadpoolctl
    import pilot_runner_g2 as P2
    G1, S, R, B, A = P2.G1, P2.S, P2.R, P2.B, P2.A
    proc = psutil.Process()
    t0 = time.perf_counter()
    marks = []

    def mark(stage):
        gc.collect()
        marks.append(dict(stage=stage, t=round(time.perf_counter() - t0, 2), rss=proc.memory_info().rss,
                          maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

    con = R.load_contract_verified()
    al = R.C1.load_allowlist(spec["allowlist_path"])
    prof = S.SourceProfile(spec["profile"]["name"], spec["profile"]["allowlist_sha256"],
                           tuple(map(tuple, spec["profile"]["file_rows"])), tuple(map(tuple, spec["profile"]["attempted_rows"])))
    if prof == S.P1_PROFILE or al["_sha256"] == S.P1_PROFILE.allowlist_sha256:
        raise RuntimeError("feasibility harness refuses real P1 data")
    cfg = B.load_bound_grid()[spec["config_id"]]
    fold = S.FoldSpec(spec["outer"])
    paths, val_day, v1 = spec["paths"], spec["outer"], spec["variant"] == "V1"
    load = lambda d: S.load_day_block(paths[d], al, con, "primary", prof)
    mark("start")
    if v1:
        blocks = [load(d) for d in fold.training_days()]
        mark("loaded_training_days")
        tm = S.assemble_training_matrix(blocks, fold, al, con, "primary", prof)
        del blocks
        mark("assembled")
    else:
        blocks = {d: load(d) for d in S.DAYS if d in set(fold.training_days()) | {val_day}}
        mark("loaded_all_needed_days")
        tm = S.assemble_training_matrix([blocks[d] for d in fold.training_days()], fold, al, con, "primary", prof)
        val = blocks[val_day]
        del blocks
        mark("assembled")
    art = S.fit_p1_final(tm, con, cfg.branch, al, fold, spec["evidence_dir"], prof)
    mark("preprocessing_fitted")
    tmp_bytes = 0
    if v1:
        G1.require_verified_artifact(art, tm, fold, al, con, prof)
        ytr = G1.binary_target(tm.labels, tm.attempted, "primary").astype(np.int8)
        mm_path = os.path.join(spec["tmp_dir"], f"Xtr_{cfg.config_id}.npy")
        out, cols = R.transform(tm.X, art, con, out_path=mm_path)
        out.flush(); del out
        del tm
        mark("train_transformed_to_disk_raw_released")
        tmp_bytes = os.path.getsize(mm_path)
        Xtr = np.load(mm_path, mmap_mode="r")
        val = load(val_day)
        Xva, _ = R.transform(val.X, art, con)
        yva = G1.binary_target(val.labels, val.attempted, "primary").astype(np.int8)
        del val
        mark("validation_loaded_transformed")
        if cfg.family == "RF":                                     # the cast scikit-learn would do internally
            Xf = np.empty(Xtr.shape, dtype=np.float32, order="F")
            for j in range(Xtr.shape[1]):
                Xf[:, j] = Xtr[:, j]
            del Xtr
            Xtr = Xf
            Xva = np.asfortranarray(Xva, dtype=np.float32)
            mark("rf_float32_cast")
    else:
        Xtr, _ = R.transform(tm.X, art, con)
        Xva, _ = R.transform(val.X, art, con)
        G1.require_verified_artifact(art, tm, fold, al, con, prof)
        ytr = G1.binary_target(tm.labels, tm.attempted, "primary").astype(np.int8)
        yva = G1.binary_target(val.labels, val.attempted, "primary").astype(np.int8)
        mark("transformed_in_ram")
    est = A.build_estimator(cfg)
    mark("estimator_constructed")
    tf = time.perf_counter()
    est.fit(Xtr, ytr)                                             # SYNTHETIC data and labels only
    fit_s = time.perf_counter() - tf
    mark("model_fitted")
    import joblib
    import multiprocessing as _mp
    info = dict(family=cfg.family, child_daemon=_mp.current_process().daemon, n_jobs_param=est.get_params().get("n_jobs"),
                joblib_effective_n_jobs=joblib.effective_n_jobs(est.get_params().get("n_jobs") or 1))
    if cfg.family == "RF":
        nodes = [t.tree_.node_count for t in est.estimators_]
        info.update(total_nodes=int(sum(nodes)), max_nodes=int(max(nodes)), max_depth=int(max(t.tree_.max_depth for t in est.estimators_)),
                    approx_model_bytes=int(sum(t.tree_.__getstate__()["nodes"].nbytes + t.tree_.value.nbytes for t in est.estimators_)))
    elif cfg.family == "HGB":
        info.update(n_iter=int(est.n_iter_))
    else:
        info.update(n_iter=int(np.max(est.n_iter_)))
    tp = time.perf_counter()
    p = est.predict_proba(Xva)[:, 1]
    pred_s = time.perf_counter() - tp
    mark("predicted")
    from sklearn.metrics import average_precision_score
    syn_ap = float(average_precision_score(yva, p)) if yva.any() else None
    n_train, n_val, n_out = int(Xtr.shape[0]), int(Xva.shape[0]), int(Xtr.shape[1])
    del est, Xtr, Xva, p
    mark("cleanup")
    if v1:
        os.remove(mm_path)
    return dict(config_id=cfg.config_id, variant=spec["variant"], outer=spec["outer"], branch=cfg.branch,
                n_train=n_train, n_val=n_val, n_model_columns=n_out, fit_s=round(fit_s, 1), predict_s=round(pred_s, 1),
                total_s=round(time.perf_counter() - t0, 1), marks=marks, model_info=info, temp_file_bytes=tmp_bytes,
                synthetic_ap_no_signal=syn_ap, synthetic_prevalence=float(yva.mean()),
                threadpools=[dict(api=x.get("internal_api"), n=x.get("num_threads")) for x in threadpoolctl.threadpool_info()])


def _child(fn, args, conn):
    try:
        conn.send(("OK", fn(*args), None))
    except BaseException as e:  # noqa: BLE001
        conn.send(("FIT_ERROR", None, f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def _tree(ps):
    try:
        return [ps] + ps.children(recursive=True)
    except psutil.Error:
        return [ps]


def run_monitored(fn, args, run_id, sample_log, limit_bytes=int(6.5 * GIB), time_limit_s=3 * 3600, sample_s=0.5,
                  critical_stop_s=10.0, disk_path=".", daemon=False):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(fn, args, child), daemon=daemon)   # non-daemon: RF n_jobs=4 honoured
    pre = host_snapshot(disk_path)
    t0 = time.monotonic()
    p.start(); child.close()
    root = psutil.Process(p.pid)
    peak = dict(rss=0, swap=pre["swap_used"], min_available=pre["available"], max_pressure=pre["pressure_level"] or 0,
                max_threads=0, max_cpu=0.0, min_disk=pre["disk_free"])
    status = result = error = None
    crit_since, last_p, plevel, n = None, 0.0, pre["pressure_level"], 0
    cpu_seen = {}
    with open(sample_log, "a") as log:
        while True:
            procs = _tree(root)
            rss = thr = 0
            cpu = 0.0
            for q in procs:
                try:
                    rss += q.memory_info().rss; thr += q.num_threads()
                    if q.pid not in cpu_seen:
                        cpu_seen[q.pid] = q; q.cpu_percent(None)
                    else:
                        cpu += cpu_seen[q.pid].cpu_percent(None)
                except psutil.Error:
                    pass
            now = time.monotonic() - t0
            if now - last_p >= 2.0:
                plevel, last_p = pressure_level(), now
            vm = psutil.virtual_memory(); sw = swap_used(); dfree = shutil.disk_usage(disk_path).free
            peak.update(rss=max(peak["rss"], rss), swap=max(peak["swap"], sw), min_available=min(peak["min_available"], vm.available),
                        max_pressure=max(peak["max_pressure"], plevel or 0), max_threads=max(peak["max_threads"], thr),
                        max_cpu=max(peak["max_cpu"], cpu), min_disk=min(peak["min_disk"], dfree))
            if n % 4 == 0:
                log.write(json.dumps(dict(run=run_id, t=round(now, 1), rss=rss, threads=thr, cpu=round(cpu), available=vm.available,
                                          swap_used=sw, pressure=plevel, disk_free=dfree)) + "\n")
            n += 1
            if parent.poll():
                status, result, error = parent.recv()
                break
            if rss > limit_bytes:
                status, error = "RESOURCE_EXCEEDED", f"child tree RSS {rss} > limit {limit_bytes}"
                break
            if now > time_limit_s:
                status, error = "TIME_EXCEEDED", f"elapsed {now:.0f} s > {time_limit_s} s"
                break
            if plevel == 4:
                crit_since = crit_since or now
                if now - crit_since >= critical_stop_s:
                    status, error = "RESOURCE_EXCEEDED", f"memory pressure CRITICAL for >= {critical_stop_s} s"
                    break
            else:
                crit_since = None
            if not p.is_alive() and not parent.poll():
                status, error = "FIT_ERROR", f"child exited with code {p.exitcode} without a result"
                break
            time.sleep(sample_s)
    if p.is_alive():
        for q in _tree(root):
            try:
                q.kill()
            except psutil.Error:
                pass
        p.join(timeout=30)
    post = host_snapshot(disk_path)
    return dict(run=run_id, status=status, error=error, child_daemon=daemon, elapsed_s=round(time.monotonic() - t0, 1), limit_bytes=limit_bytes,
                peak_tree_rss=peak["rss"], peak_swap_used=peak["swap"], swap_growth=peak["swap"] - pre["swap_used"],
                min_available=peak["min_available"], max_pressure_level=peak["max_pressure"], max_threads=peak["max_threads"],
                max_cpu_percent=round(peak["max_cpu"]), min_disk_free=peak["min_disk"], host_before=pre, host_after=post,
                result=(result if status == "OK" else None))
