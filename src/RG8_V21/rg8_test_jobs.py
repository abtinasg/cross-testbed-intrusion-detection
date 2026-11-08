#!/usr/bin/env python3
"""Synthetic job functions for the RG8 tests (module-level so that the spawn worker can import them). No data."""
import os
import subprocess
import sys
import time

GIB = 1024 ** 3


def ok_job(x):
    import joblib
    import multiprocessing as mp
    import rg8_guard as RG
    RG.stage("compute")
    return dict(value=x * 2, daemon=mp.current_process().daemon, effective_n_jobs=joblib.effective_n_jobs(4),
                pgid_is_self=os.getpgrp() == os.getpid(), executable=os.path.realpath(sys.executable))


def instant_job():
    return dict(done=True)


def hog_job(step_mib=64, max_gib=6):
    import numpy as np
    import rg8_guard as RG
    RG.stage("hog")
    hold = []
    for _ in range(int(max_gib * 1024 / step_mib)):
        hold.append(np.ones(step_mib * 1024 * 1024 // 8))
        time.sleep(0.05)
    time.sleep(3600)


def grandchild_hog_job():
    """The worker itself stays small; a grandchild process allocates (tree accounting must catch it)."""
    code = ("import numpy as np, time\nh=[]\nfor _ in range(96):\n    h.append(np.ones(8*1024*1024)); time.sleep(0.05)\n"
            "time.sleep(3600)\n")
    subprocess.Popen([sys.executable, "-I", "-c", code])
    time.sleep(3600)


def sleep_job(seconds=3600):
    time.sleep(seconds)
    return dict(slept=seconds)


def short_sleep_job(seconds):
    time.sleep(seconds)
    return dict(slept=seconds)


def grandchild_sleep_job(pid_file):
    """V21 cleanup tests: the worker spawns a long-lived grandchild, records its pid, then sleeps."""
    g = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(3600)"])
    with open(pid_file, "w") as fh:
        fh.write(str(g.pid))
    time.sleep(3600)
