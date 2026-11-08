# Release notes — v1.0.0 (2026-11-08)

This release is the code and aggregate output of the study as run on 2026-11-06/07. It was assembled from the
private research package by copying files byte-for-byte. `docs/PROVENANCE_FILE_MAP.json` lists every copied file
with its SHA-256 in the research package and in this release.

## Bytes that differ from the research package (none is scientific)

| File(s) | Change | Why |
|---|---|---|
| `src/RG8_V21/RG8_POLICY_PROPOSED.json` (V20 uniform policy) | `"executable"` value `/Users/<user>/anaconda3/bin/python3` → `<PINNED_INTERPRETER:anaconda3/bin/python3>` | removes a personal filesystem path. Original SHA-256 `3ba9d56dfd26df5f1e0e196e6bba2b9e081c5158e5a993faef917bcde6891380`, release `eb30ba81a6b0b5a72db80baa1e02f9c7ab708fdde83060a3378d8b6263b5523e` |
| `src/RG8_V22_A1_SMOKE/A1_SMOKE_POLICY.json` | same | original `76406993e7d4d565837a5ed34785badabc4db926d4ac75e9508bcd80f49a21c3`, release `3703cd744203419f1d78539ccca049c137d5e83e6bc69f3fe581c56c53dba69a` |
| `src/RG8_V22_A1_SMOKE/smoke_run/RESULT.json` | same path redacted in the recorded environment; the recorded list of running applications replaced by `<redacted: local process list>` | personal path and machine information |
| `src/RG8_V21/preflight_probes/probe_2026-11-06T215715Z.json`, `probe_2026-11-06T220928Z.json` | host name, interpreter path and local process list replaced by `<redacted…>`; all memory/swap/pressure measurements unchanged (re-serialised as JSON) | machine information; the A1 test F4 reads only the measurements |
| `src/RG8_V21/rg8_preflight_probe.py`, `src/RG8_V21/test_rg8_guard.py`, `src/RG8_V22_A1_SMOKE/run_p1_smoke.py`, `src/RG8_V22_A1_SMOKE/test_a1_smoke.py` | same path redacted **in docstrings only** (usage lines) | personal path; code unchanged |
| `src/FLAT_CV_P1/start_flat_cv.sh`, `src/TB_CALIBRATION_RF03/start_tb.sh`, `src/FASTTRACK_V23/P2_RUN/start_p2.command` | `PY=/Users/<user>/anaconda3/bin/python3` → `PY=${PY:-python3}` | personal path; the interpreter is now set by the `PY` environment variable |
| `src/paper/make_tables.py` | new file. The loaders and `table1()`–`table5()` are a verbatim excerpt of the manuscript builder that produced the submitted tables; only the Markdown printer is new | the full builder also embeds the manuscript text and the conference's Word template, which are not part of this code release |
| `src/paper/make_figures.py` | byte-identical; moved from the manuscript folder | — |

Because the two policy files were redacted, their hashes no longer equal the pinned constants. That is intended: the
policy pins one interpreter on one machine and must be re-pinned on any other host (next section).

## Re-pinning the resource-guard policy on your machine

`python -I -B scripts/repin_guard_policy.py` does all of the following and prints the new tokens. For this release
it was checked on a copy of the repository: after re-pinning, all 17 test suites passed. The manual procedure:

1. In `src/RG8_V21/RG8_POLICY_PROPOSED.json` and `src/RG8_V22_A1_SMOKE/A1_SMOKE_POLICY.json`, set `"executable"` to
   `os.path.realpath(sys.executable)` of your Python 3.13.5 interpreter. The numpy, scikit-learn and psutil versions
   must match the `environment` block, or the guard refuses to start.
2. `shasum -a 256` both files.
3. Replace `3ba9d56dfd26df5f1e0e196e6bba2b9e081c5158e5a993faef917bcde6891380` with the new V20 hash in:
   - `src/RG8_V21/rg8_guard.py` (`POLICY_SHA256`)
   - `src/RG8_V23_EXIT_RACE/rg8_guard_v23.py` (`POLICY_SHA256`)
   - `src/FLAT_CV_P1/run_flat_cv.py` (`V20_POLICY_SHA256`)
   - `src/RG8_V22_A1_SMOKE/run_p1_smoke.py` (`V20_POLICY_SHA256`)
4. Replace `76406993e7d4d565837a5ed34785badabc4db926d4ac75e9508bcd80f49a21c3` with the new A1 hash in
   `src/RG8_V22_A1_SMOKE/run_p1_smoke.py` (`A1_POLICY_SHA256`).
5. The authorization tokens become `GO-A1-LR01-TUE-<first 12 hex of the A1 hash>`,
   `GO-FLATCV-P1-<first 12 hex of the V20 hash>` and `GO-TB-RF03-P1-<first 12 hex of the V20 hash>`. The P2 token
   `GO-P2-RF03-d1a6585ac71b` is derived from `OPERATIONAL_SETTINGS_v1.json` and does not change.

6. Step 3 changes the two guard files, and their own hashes are pinned too. Update
   `V21_GUARD_SHA256` (the new SHA-256 of `src/RG8_V21/rg8_guard.py`) in `src/FLAT_CV_P1/run_flat_cv.py` and
   `src/RG8_V22_A1_SMOKE/run_p1_smoke.py`. Update `V23_GUARD_SHA256` (the new SHA-256 of
   `src/RG8_V23_EXIT_RACE/rg8_guard_v23.py`) in `src/FASTTRACK_V23/P2_RUN/run_p2.py`.
   `grep -rn '_SHA256 = "' src` lists every pinned hash.

## <a name="fresh-run"></a>Starting a fresh run

The repository ships the original run's records. Remove the generated outputs before a new run; the runners treat
existing ledger entries as done and refuse to overwrite write-once files:

```bash
cd src
rm -r FLAT_CV_P1/cv_run TB_CALIBRATION_RF03/tb_run TB_CALIBRATION_RF03/tb_calibration FASTTRACK_V23/P2_RUN/p2_run
rm RG8_V22_A1_SMOKE/smoke_run/RESULT.json \
   P1_FLAT_CV_CLOSEOUT/P1_FLAT_CV_VERIFICATION.json P1_FLAT_CV_CLOSEOUT/P1_FLAT_CV_VERIFICATION.md \
   TB_CALIBRATION_RF03/TB_RESULTS.json TB_CALIBRATION_RF03/TB_RESULTS.md \
   FASTTRACK_V23/P1_TA_RESULTS.json FASTTRACK_V23/P1_TA_RESULTS.md FASTTRACK_V23/FINAL_DETECTOR_TB_THRESHOLDS.json \
   FASTTRACK_V23/P1_FREEZE_CHECK.json FASTTRACK_V23/P2_PREREGISTRATION.json \
   FASTTRACK_V23/P2_RUN/P2_SCORES_SEALED.json FASTTRACK_V23/P2_RUN/SEAL_VERIFICATION.json \
   FASTTRACK_V23/P2_RUN/P2_RESULTS.json FASTTRACK_V23/P2_RUN/P2_RESULTS.md
```

Keep a copy (or use `git stash`/a second clone) if you want to compare your results with the originals.

## Run-specific audit checks (important for a fresh run)

Two scripts audit **this study's** run history, not a generic one:

- `src/P1_FLAT_CV_CLOSEOUT/verify_p1_flat_cv.py` requires the only non-OK ledger record to be LR-02/Tue attempt 1
  (killed by an external SIGTERM) with a consistent recovery chain.
- `src/FASTTRACK_V23/p1_freeze_and_preregister.py` requires the only non-OK T-b record to be RF-03 Fri/Tue attempt 1,
  with its recovery chain and a launcher status ending in `rc=0`.

A new run with no interruptions (or different ones) will make these audits report problems, and the freeze will not
write `P2_PREREGISTRATION.json`. Adapt those recovery-chain assertions to your own run history. The scientific
computations they perform (AP recomputation, selection, threshold re-derivation, hash checks) are unaffected.

`recover_tb.command` and `recover_p2_wed14.command` reproduce the two authorised one-time recoveries of the original
run. Their tokens are bound to the original STOPPED records.

## What was deliberately not released

- Datasets, row-level derived arrays (`*.npy`, `*.parquet`), and the trained model. See `docs/DATASETS.md`.
- Per-fit evidence folders (`evidence/`), launcher status files, console logs, `RUN_EVENTS.jsonl`,
  `PREFLIGHT_PAUSES.jsonl` and `AUTHORIZATION.json`. These are machine-specific runtime logs, and some contain local
  paths. The ledgers that are released keep each fit's hashes, timing and guard summary.
- Internal project records: decision records, specification drafts, freeze logs, manuscript drafts and
  correspondence. Test class `T7` of `test_r5_final.py` and class `T5_FrozenArtefactsPreserved` of `test_g2_f3_binding_and_adapter.py` check those private files and are therefore not run by
  `tests/run_release_tests.py`. All other classes run unchanged.
- Earlier guard versions and exploratory stages that the final pipeline does not import. The one exception is
  `src/RG8_UNIFIED_GUARD_V20/RG8_POLICY_EVIDENCE.json`: the measurements from which the V20 policy limits were
  derived. It is byte-identical and is read by the A1 policy tests.
