#!/usr/bin/env python3
"""Focused regression test of the V23 verified-exit race rule (synthetic; no data). 5 cases:
 T1 V21 reproduces the failure: sensor fails after the job returned -> MEASUREMENT_UNAVAILABLE
 T2 V23 same scenario -> OK, payload delivered, exit_race recorded, no survivors
 T3 V23 sensor fails while the worker is still running -> MEASUREMENT_UNAVAILABLE, worker killed
 T4 V23 job raises after the flag, then the sensor fails -> not OK
 T5 V23 resource breach still stops (tree memory limit) -> RESOURCE_EXCEEDED
"""
import copy, json, os, sys, tempfile, time, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
V21 = os.path.join(os.path.dirname(HERE), "RG8_V21")
sys.path[:0] = [HERE, V21]
import rg8_guard as G21  # noqa: E402
import rg8_guard_v23 as G23  # noqa: E402
import rg8_test_jobs as J21  # noqa: E402
import v23_test_jobs as J  # noqa: E402

GIB = 1024 ** 3
BASE = json.load(open(G21.POLICY_PATH))


def policy(G, d, **over):
    pol = copy.deepcopy(BASE)
    pol["monitor"].update(dict(tree_mem_limit_bytes=int(1.0 * GIB), poll_s=0.1, max_sample_gap_s=5.0, time_limit_s=120))
    pol["monitor"].update(over)
    p = os.path.join(d, "policy.json"); json.dump(pol, open(p, "w"))
    return G.load_policy(p, _unpinned_test_policy=True)


def failing_pressure(G, flag, delay=0.3):
    class S(G.Sensors):
        def pressure(self):
            if os.path.exists(flag):
                time.sleep(delay)
                raise G.SensorUnavailable("simulated: exiting worker unreadable")
            return super().pressure()
    return S()


class T(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(); self.flag = os.path.join(self.d, "flag")

    def run_g(self, G, fn, args, sensors=None, **over):
        return G.run_guarded_rg8(fn, args, policy=policy(G, self.d, **over), work_dir=self.d, child_lock_path=os.path.join(self.d, "lock"),
                                 sample_log_path=os.path.join(self.d, "s.jsonl"), stage_file=os.path.join(self.d, "st.jsonl"), sensors=sensors)

    def test_T1_v21_reproduces(self):
        r = self.run_g(G21, J.flag_then_return, (self.flag,), failing_pressure(G21, self.flag))
        self.assertEqual((r["status"], r["failure_class"]), ("FIT_ERROR", "MEASUREMENT_UNAVAILABLE"))

    def test_T2_v23_accepts_verified_exit(self):
        r = self.run_g(G23, J.flag_then_return, (self.flag,), failing_pressure(G23, self.flag))
        self.assertEqual(r["status"], "OK", r)
        self.assertEqual(r["payload"], dict(value=42))
        self.assertIsNotNone(r["exit_race"]); self.assertEqual(r["exit_race"]["worker_exitcode"], 0)
        self.assertEqual(r["survivors"], [])

    def test_T3_v23_running_worker_fails_closed(self):
        open(self.flag, "w").close()                                   # sensor fails while long_sleep is running
        t = time.time()
        r = self.run_g(G23, J.long_sleep, (), failing_pressure(G23, self.flag, delay=0.0))
        self.assertEqual((r["status"], r["failure_class"]), ("FIT_ERROR", "MEASUREMENT_UNAVAILABLE"))
        self.assertIsNone(r["payload"]); self.assertIsNone(r["exit_race"]); self.assertEqual(r["survivors"], [])
        self.assertLess(time.time() - t, 30)

    def test_T4_v23_job_error_not_accepted(self):
        r = self.run_g(G23, J.flag_then_raise, (self.flag,), failing_pressure(G23, self.flag))
        self.assertNotEqual(r["status"], "OK"); self.assertIsNone(r["payload"]); self.assertIsNone(r["exit_race"])

    def test_T5_v23_breach_still_stops(self):
        r = self.run_g(G23, J21.hog_job, (), None, tree_mem_limit_bytes=int(0.5 * GIB))
        self.assertEqual((r["status"], r["failure_class"]), ("RESOURCE_EXCEEDED", "TREE_MEMORY"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
