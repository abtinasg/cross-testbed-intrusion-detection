#!/usr/bin/env python3
"""Run the research test suites shipped with the code.

The suites live next to the modules they test (they import their siblings by directory, exactly as in the research
package). This runner executes each suite in its own directory with the current interpreter in isolated mode.

    python -I -B tests/run_release_tests.py            # all suites (about 6 minutes on an 8 GB Apple M1)
    python -I -B tests/run_release_tests.py --quick    # skip the three slowest guard suites

No suite needs the datasets: they use synthetic fixtures. Suites that would touch real data skip themselves.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
SUITES = [
    "preprocessing_review/test_prep_reference.py",
    "correction_package_C1/prep/test_prep_v0_2.py",
    "correction_package_C1/guard/test_guard_v2_1.py",
    "correction_package_C2/lean/test_prep_exact.py",
    "R5_FREEZE_F7/test_r5_final.py",
    "amendment_A1/test_cv_threshold_reference.py",
    "amendment_A1b/test_cv_threshold_reference_A1b.py",
    "PILOT_BINDING_G2/test_g2_ap_validation.py",
    "PILOT_BINDING_G2/test_g2_recovery.py",
    "PILOT_G2_CORRECTIONS/test_nondaemon_worker.py",
    "PILOT_G2_CORRECTIONS/test_g2c_live_parent_recovery.py",
    "PILOT_INTEGRATION_G1/test_g1_source_enforcement.py",
    "RG8_V22_A1_SMOKE/test_a1_smoke.py",
    "RG8_V23_EXIT_RACE/test_v23_exit_race.py",
    "PILOT_BINDING_G2/test_g2_f3_binding_and_adapter.py",   # slow
    "RG8_V21/test_rg8_guard.py",                            # slow
    "PILOT_INTEGRATION_G1/test_g1_guard_integration.py",    # slow
]
SLOW = set(SUITES[-3:])
# test_r5_final.py class T7 checks that every file in the private research package's FREEZE_LOG is unchanged. Those
# internal process records are not part of this release, so T7 is not run here (the test file itself is unchanged).
ARGS = {"R5_FREEZE_F7/test_r5_final.py": ["T1_EquivalenceWithApprovedCandidateA", "T2_ZV1", "T3_KappaDiagnostic",
                                         "T4_EnvironmentAndPins", "T5_SourceProvenance", "T6_HighPrecisionReference"],
        # same reason: T5_FrozenArtefactsPreserved checks the private FREEZE_LOG
        "PILOT_BINDING_G2/test_g2_f3_binding_and_adapter.py": ["T1_F3Binding", "T2_NoSubstitutionEndToEnd",
                                                               "T3_GridCompletion", "T4_ProposedAdapter"]}


def main():
    quick = "--quick" in sys.argv
    failed = []
    for rel in SUITES:
        if quick and rel in SLOW:
            print(f"  skip  {rel} (--quick)")
            continue
        d, f = os.path.split(os.path.join(ROOT, rel))
        t = time.time()
        try:
            r = subprocess.run([sys.executable, "-I", "-B", f] + ARGS.get(rel, []), cwd=d, capture_output=True, text=True, timeout=900)
            last = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
            ok = r.returncode == 0
        except subprocess.TimeoutExpired:
            last, ok = ["TIMEOUT after 900 s"], False
        print(f"{'  ok  ' if ok else '  FAIL'} {rel} ({time.time() - t:.0f} s) {last[0][:100]}", flush=True)
        if not ok:
            failed.append(rel)
    print(f"\n{len(failed)} failed" if failed else "\nall suites passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
