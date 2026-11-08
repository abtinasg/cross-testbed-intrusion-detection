"""Synthetic jobs for the V23 exit-race test (module-level for spawn). No data."""
import time


def flag_then_return(flag):
    time.sleep(0.5)
    open(flag, "w").close()
    return dict(value=42)


def flag_then_raise(flag):
    time.sleep(0.5)
    open(flag, "w").close()
    raise RuntimeError("job failed after flag")


def long_sleep():
    time.sleep(60)
    return dict(slept=60)
