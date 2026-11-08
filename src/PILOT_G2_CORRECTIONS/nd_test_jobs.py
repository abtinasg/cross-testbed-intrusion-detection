#!/usr/bin/env python3
"""Top-level synthetic jobs for the non-daemon worker tests (spawn needs importable functions). Synthetic only."""
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.dirname(HERE)
for _p in (os.path.join(BRIEF, "M1_8GB_FEASIBILITY"), os.path.join(BRIEF, "PILOT_BINDING_G2"),
           os.path.join(BRIEF, "PILOT_INTEGRATION_G1"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def effective_workers():
    import joblib
    import threadpoolctl
    from sklearn.ensemble._base import _partition_estimators
    return dict(daemon=mp.current_process().daemon, pgid_is_self=os.getpgid(0) == os.getpid(),
                joblib_effective_n_jobs_4=joblib.effective_n_jobs(4), rf_partition_n_jobs=_partition_estimators(200, 4)[0],
                openmp=[x.get("num_threads") for x in threadpoolctl.threadpool_info() if x.get("internal_api") == "openmp"])


def _synthetic(n, seed):
    import numpy as np
    import pilot_runner_g2 as P2
    import synth_cic as Y
    R = P2.R
    con = R.load_contract_verified()
    X = Y.synth_matrix(n, seed, con["schema"], con["integer"])
    lab, att = Y.synth_labels(n, seed)
    y = ((lab != "BENIGN") & (att == -1)).astype(np.int8)
    return R, con, X, y


def fit_predict(config_id, n=12000, seed=3):
    """Frozen F3 estimator on SYNTHETIC data and labels; returns predictions (and RF tree arrays)."""
    import numpy as np
    import pilot_runner_g2 as P2
    R, con, X, y = _synthetic(n, seed)
    _, _, Xv, _ = _synthetic(3000, seed + 1)
    cfg = P2.B.load_bound_grid()[config_id]
    art = R.fit_core(X, con["schema"], cfg.branch, "P1:synthetic", "nd")
    Xt, _ = R.transform(X, art, con); Xvt, _ = R.transform(Xv, art, con)
    est = P2.A.build_estimator(cfg).fit(Xt, y)
    out = dict(config_id=config_id)
    if cfg.family == "RF":
        out["trees"] = [(t.tree_.feature.copy(), t.tree_.threshold.copy(), t.tree_.value.copy()) for t in est.estimators_]
        est.set_params(n_jobs=1)                                   # deterministic accumulation for comparison only
    out["proba"] = est.predict_proba(Xvt)[:, 1]
    return out


def _hold(mib, seconds):
    import numpy as np
    a = np.ones(int(mib * 1024 * 1024 / 8))
    time.sleep(seconds)
    return float(a[0])


def allocate_tree(self_mib, child_mib, n_children, hold_s):
    """Worker holds self_mib and starts n_children grandchildren holding child_mib each."""
    import numpy as np
    ctx = mp.get_context("spawn")
    kids = [ctx.Process(target=_hold, args=(child_mib, hold_s)) for _ in range(n_children)]
    for k in kids:
        k.start()
    a = np.ones(int(self_mib * 1024 * 1024 / 8))
    time.sleep(hold_s)
    for k in kids:
        k.join()
    return dict(ok=float(a[0]), kids=[k.pid for k in kids])


def hog_grandchild():
    ctx = mp.get_context("spawn")
    k = ctx.Process(target=_hold_growing)
    k.start(); k.join()
    return "unreachable"


def _hold_growing():
    import numpy as np
    hold = []
    for _ in range(400):
        hold.append(np.ones(8 * 1024 * 1024)); time.sleep(0.02)
    time.sleep(3600)


def sleep_with_grandchild(seconds):
    ctx = mp.get_context("spawn")
    k = ctx.Process(target=time.sleep, args=(3600,))
    k.start()
    time.sleep(seconds)
    return "done"


def leave_grandchild_running():
    """Returns immediately while a DAEMON-free grandchild keeps running (would be an orphan without cleanup)."""
    import subprocess
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
    return dict(grandchild=p.pid)


def short_sleep(seconds):
    time.sleep(seconds)
    return "slept"
