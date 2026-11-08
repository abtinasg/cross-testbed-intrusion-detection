# Cross-Testbed Transfer of Source-Calibrated Intrusion Detection: Threshold Failure and Alert-Budget Constraints

Research code, frozen configurations, evaluation scripts and aggregate results for the paper of the same title.

**Authors**

1. Armin Tahamtan — Faculty of Engineering, Islamic Azad University, West Tehran Branch, Tehran, Iran
2. Abtin Asghari (**corresponding author**, 0025636170@iau.ir) — Department of Computer Engineering, Islamic Azad University, West Tehran Branch, Tehran, Iran
3. MohammadMehdi Mehraein — Department of Financial Management, Islamic Azad University, Central Tehran Branch, Tehran, Iran

**Venue:** 11th National Congress of Electrical and Computer Engineering of Iran (COMCONF11).  
**Status:** manuscript prepared for submission. It has not yet been accepted or published, and no DOI exists.  
**Licence:** code and documentation MIT (see `LICENSE`). **The datasets are not included** and keep their own terms
(see [`docs/DATASETS.md`](docs/DATASETS.md)).

---

## Research overview

Flow-based intrusion detectors are usually reported with threshold-free metrics on splits of a single dataset. That
says little about the false-positive rate (FPR) an operator inherits when a fixed alert threshold is moved to a
different network. This study measures that transfer for one detector under a protocol frozen before the results
existed:

1. **Source (P1):** corrected CIC-IDS2017, five days (Mon–Fri). Twelve configurations (logistic regression,
   histogram gradient boosting, random forest; `src/next_workstream/CONFIG_GRID_F3_v1.csv`) are compared by
   leave-one-day-out cross-validation. Criterion: mAP_day = mean AP over Tue–Fri (Monday has no attacks). **RF-03**
   (random forest: 100 trees, max depth 20, minimum leaf size 20, √p features, 50% bootstrap samples) is selected and
   refitted on all five days.
2. **Source-only threshold calibration (T-b):** for each held-out day an inner leave-one-day-out produces out-of-fold
   scores; τ = the higher-order (1 − α) quantile of their benign scores, α ∈ {0.01, 0.05}. For the final detector the
   same rule is applied to the pooled benign out-of-fold scores of all five days: τ(0.01) = 0.033896,
   τ(0.05) = 0.003304. These values were fixed before any target day was scored.
3. **Target (P2):** two corrected CSE-CIC-IDS2018 days (14-02-2018 SSH brute force; 16-02-2018 DoS Hulk), scored
   label-blind with the final detector and sealed (SHA-256) before labels were read.
4. **Operational analyses:** hourly alert budgets (T-a: relative quota q = 1% per hour; absolute capacity K = 500 per
   hour; hard cap with a frozen tie-break) and annotation budgets (stratified per-hour audits of b = 100/250/500 labels,
   1,000 replicates, ratio estimators of FPR and recall with 95% intervals).

**Principal findings (negative results are the point of the paper):**

- On held-out source days the α = 0.01 threshold kept its nominal FPR (0.0064–0.0118). At α = 0.05 the benign-only
  Monday reached 0.073.
- **The same source-derived threshold failed on the target testbed:** at α = 0.01 the FPR was 0.174 (14-02) and 0.161
  (16-02), about 16× nominal, producing up to 211,326 alerts in one hour. Removing "Attempted" flows still left
  FPR 0.145.
- SSH brute force was detected in full but at precision 0.085. DoS Hulk, a family present in training, was largely
  missed (recall 0.016, ROC-AUC 0.623).
- Hourly budgets turned the false-positive surge into missed attacks: q = 1% recovered 11.5% and 0.06% of attack
  flows. K = 500 recovered none.
- Audits of 100–500 labels per hour estimated the inflated FPR with 95% coverage of 0.93–0.96, but under-covered
  recall values close to zero.
- **No adaptive or recalibration method is proposed or evaluated.**

## Repository layout

```
cross-testbed-intrusion-detection/
├── README.md, LICENSE, CITATION.cff, requirements.txt, .gitignore
├── src/                     the research code, in its ORIGINAL folder structure (see below)
│   ├── FLAT_CV_P1/          P1 leave-one-day-out CV of the 12 configurations (60 fits) + summary
│   ├── P1_FLAT_CV_CLOSEOUT/ integrity re-check of the 60 fits, AP recomputation, RF-03 selection
│   ├── TB_CALIBRATION_RF03/ T-b inner fits (20) and source-only threshold calibration
│   ├── FASTTRACK_V23/       P1 alert-budget evaluation, final thresholds, P1 freeze + P2 pre-registration
│   │   └── P2_RUN/          final fit, label-blind P2 scoring, seal, seal check, P2 evaluation
│   ├── RG8_V21/             resource guard (memory/swap/pressure monitor) + its frozen policy
│   ├── RG8_V22_A1_SMOKE/    the first real fit (LR-01 held out on Tuesday), imported into the CV
│   ├── RG8_V23_EXIT_RACE/   V23 guard correction (verified-exit rule), used for the remaining P2 keys
│   ├── PILOT_*/, R5_FREEZE_F7/, correction_package_C1..C4/, preprocessing_review/, amendment_A1*/
│   │                        frozen data loading, validation, preprocessing, model adapter, job runner,
│   │                        hard-cap selection and tie-key reference functions (with their tests)
│   ├── M1_8GB_FEASIBILITY/  synthetic-data and measurement helpers used by the guard tests
│   ├── next_workstream/CONFIG_GRID_F3_v1.csv, FEATURE_SCHEMA_PRIMARY_72_DRAFT.csv
│   └── paper/               make_tables.py, make_figures.py, figures/
├── STAGE02A_0_DATASET_AUDIT/07_CROSS_DATASET_FEATURE_MATRIX.csv
│                            per-column aggregate statistics (no rows) read by the synthetic test-data generator
├── scripts/data_acquisition/ selective HTTP-Range download of the two P2 day files
├── scripts/repin_guard_policy.py  re-pins the resource-guard policy to your interpreter
├── tests/run_release_tests.py  runs the 17 research test suites
├── configs/README.md        index of every frozen configuration file
├── results/README.md        index of every aggregate result file + the paper tables
└── docs/                    DATASETS.md, RELEASE_NOTES.md, PROVENANCE_FILE_MAP.json
```

**Why `src/` keeps the original folder names.** Modules locate each other, and the frozen files, by paths relative
to the research package root. Renaming or regrouping folders would change the code. Every file in `src/` is a
byte-identical copy of the file that produced the results, except for the documented, non-scientific redactions in
[`docs/RELEASE_NOTES.md`](docs/RELEASE_NOTES.md). Per-file SHA-256 values, original vs release, are in
`docs/PROVENANCE_FILE_MAP.json`.

## Environment

| Item | Value used for all reported results |
|---|---|
| Hardware | Apple M1, 8 GB RAM (8 logical CPUs) |
| OS | macOS 26.6.1 |
| Python | 3.13.5 (Anaconda), always run as `python -I -B` |
| Packages | `requirements.txt` (numpy 2.4.6, scikit-learn 1.9.0, scipy 1.18.0, polars 1.40.1, pyarrow 24.0.0, psutil 5.9.0, threadpoolctl 3.5.0, joblib 1.5.3, matplotlib 3.11.0) |
| Seeds | estimators `random_state = 20261104` (`src/PILOT_BINDING_G2/ADAPTER_PINNED_PARAMS_v1_PROPOSED.json`); audit simulation seed `20261107` (`src/FASTTRACK_V23/OPERATIONAL_SETTINGS_v1.json`) |

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Platform constraint.** The resource guard (`src/RG8_V21/rg8_guard.py`) reads macOS kernel memory-pressure and
swap counters through `libSystem`. The guarded training runners therefore run on **macOS only**. The evaluation
scripts, the table/figure scripts and most test suites are plain Python.

## What you can run without the datasets (minutes)

```bash
python -I -B scripts/repin_guard_policy.py       # once per machine (see Step 0 below); needed by the guard suites
python -I -B tests/run_release_tests.py          # 17 research test suites, synthetic fixtures (~8 min on M1)
python -I -B tests/run_release_tests.py --quick  # skips the 3 slowest guard suites
cd src/paper
python -I -B make_tables.py    # prints Tables 1-5 of the paper from the stored result JSONs
python -I -B make_figures.py   # rewrites figures/fig1_*.png and fig2_*.png (byte-identical to the paper figures)
```

## Full reproduction (needs the datasets and many hours of computation)

Obtain and check the data first ([`docs/DATASETS.md`](docs/DATASETS.md)). Compute used in the original runs on the
8 GB M1: 60 CV fits ≈ 2.6 h of fit time (longest fit 252 s), 20 T-b fits ≈ 0.7 h, final fit + P2 scoring ≈ 7 min,
plus waits for free memory. Each fit runs in a separate guarded process. A fit starts only when ≥ 3 GiB of memory is
available, and stops if the process tree exceeds 4.75 GiB, swap grows by more than 2.75 GiB, or it runs over 1 h.
Close browsers and other large applications before starting.

### Step 0 — two things you must do on a new machine

1. **Start from a clean state.** This repository ships the completed run records of the study (ledgers, recovery
   records, sealed P2 manifest, results). The runners treat existing ledger entries as already done, and refuse to
   overwrite write-once files. Before a fresh run, delete the generated outputs listed in
   [`docs/RELEASE_NOTES.md#fresh-run`](docs/RELEASE_NOTES.md#fresh-run).
2. **Re-pin the guard policy to your interpreter.** The frozen policy pins the interpreter path and the numpy,
   scikit-learn and psutil versions. The code pins the SHA-256 of the policy files and of the guard modules, and the
   authorization tokens are derived from those hashes. Run, with the interpreter you will use:

   ```bash
   python -I -B scripts/repin_guard_policy.py   # rewrites "executable" + the pinned hashes; prints the new tokens
   ```

   No limit or scientific setting changes. The manual procedure is in `docs/RELEASE_NOTES.md`. The guard test
   suites need this step too: before re-pinning, the suites that load the frozen policy refuse to run because the
   redacted interpreter path cannot match.

### Steps (each runner has `--check`, which validates everything and writes nothing)

`<V20>` is the first 12 hex digits of your re-pinned V20 policy hash; `<A1>` is the same for the A1 policy.
Set `PY` to your interpreter for the launchers (`export PY=$(which python)`).

| Stage | Command (run from the folder shown) | Output |
|---|---|---|
| 1. First fit: LR-01 held out on Tuesday | `src/RG8_V22_A1_SMOKE`: `python -I -B run_p1_smoke.py --check`, then `... --execute --authorize GO-A1-LR01-TUE-<A1>` | `smoke_run/RESULT.json` |
| 2. P1 model selection: 60 LODO fits | `src/FLAT_CV_P1`: `python -I -B run_flat_cv.py --check`, then `./start_flat_cv.sh` (calls `run_flat_cv.py --execute --authorize GO-FLATCV-P1-<V20>`, resumes after low-memory pauses, then runs `summarize_flat_cv.py`) | `cv_run/LEDGER.jsonl`, `cv_run/FLAT_CV_RESULTS.*` |
| 3. Close-out and RF-03 selection | `src/P1_FLAT_CV_CLOSEOUT`: `python -I -B verify_p1_flat_cv.py` | `P1_FLAT_CV_VERIFICATION.*` |
| 4. T-b inner fits and calibration | `src/TB_CALIBRATION_RF03`: `python -I -B run_tb.py --check`, then `./start_tb.sh` (token `GO-TB-RF03-P1-<V20>`; runs `calibrate_tb.py` at the end) | `TB_RESULTS.*`, `tb_calibration/RF-03/` |
| 5. P1 alert budgets + final-detector thresholds | `src/FASTTRACK_V23`: `python -I -B p1_operational_eval.py` | `P1_TA_RESULTS.*`, `FINAL_DETECTOR_TB_THRESHOLDS.json` |
| 6. P1 freeze and P2 pre-registration | `src/FASTTRACK_V23`: `python -I -B p1_freeze_and_preregister.py` | `P1_FREEZE_CHECK.json`, `P2_PREREGISTRATION.json` |
| 7. Final fit + label-blind P2 scoring + seal | `src/FASTTRACK_V23/P2_RUN`: `python -I -B run_p2.py --check`, then `./start_p2.command` (token `GO-P2-RF03-d1a6585ac71b`, the SHA-256 prefix of `OPERATIONAL_SETTINGS_v1.json`) | `p2_run/`, `P2_SCORES_SEALED.json` |
| 8. Seal check, then P2 evaluation | `src/FASTTRACK_V23/P2_RUN`: `python -I -B verify_seal.py`, then `python -I -B evaluate_p2.py` | `SEAL_VERIFICATION.json`, `P2_RESULTS.*` |
| 9. Paper tables and figures | `src/paper`: `python -I -B make_tables.py`, `python -I -B make_figures.py` | Markdown tables, `figures/` |

`recover_tb.command` and `recover_p2_wed14.command` are the one-time recoveries that the original runs needed: one
fit killed by an external SIGTERM, and one monitor exit race fixed by the V23 guard. They are kept for provenance.
Their tokens are tied to the original STOPPED records and will not apply to a new run.

**Expected agreement.** All estimators use a fixed seed and the library versions are pinned. A full re-run has not
been repeated for this release, so bit-identical agreement on other hardware is not claimed. Different hardware,
thread counts or library builds can change scores in the last digits, and therefore the counts near a threshold.
The stored ledgers record SHA-256 hashes of every original score vector, so you can compare against them.

## Results (verified aggregate values; full files indexed in [`results/README.md`](results/README.md))

### P1 model selection (leave-one-day-out AP; mAP_day is selection-optimistic)

| Config | AP Tue | AP Wed | AP Thu | AP Fri | mAP_day |
|---|---|---|---|---|---|
| prevalence | 0.0215 | 0.3456 | 0.1986 | 0.4656 | |
| **RF-03** (selected) | 0.4568 | 0.9930 | 0.9839 | 0.9998 | **0.8584** |
| RF-04 | 0.4515 | 0.9928 | 0.9838 | 0.9996 | 0.8569 |
| HGB-03 | 0.6666 | 0.7504 | 0.9587 | 0.9984 | 0.8435 |
| HGB-05 | 0.4992 | 0.8893 | 0.9483 | 0.9990 | 0.8339 |
| RF-02 | 0.3486 | 0.9935 | 0.9835 | 0.9996 | 0.8313 |
| RF-01 | 0.3332 | 0.9923 | 0.9825 | 0.9997 | 0.8269 |
| HGB-02 | 0.3327 | 0.9569 | 0.9706 | 0.9995 | 0.8149 |
| HGB-01 | 0.2225 | 0.9703 | 0.9745 | 0.9990 | 0.7915 |
| HGB-04 | 0.6309 | 0.4245 | 0.9712 | 0.9879 | 0.7536 |
| LR-01 | 0.0111 | 0.5104 | 0.8932 | 0.9983 | 0.6032 |
| LR-02 | 0.0112 | 0.4479 | 0.7746 | 0.9979 | 0.5579 |
| LR-03 | 0.0116 | 0.4695 | 0.6703 | 0.9974 | 0.5372 |

### T-b source-only calibration on held-out source days (RF-03, α = 0.01)

| Held-out day | τ | achieved FPR | recall | precision | alerts/day |
|---|---|---|---|---|---|
| Mon | 0.045819 | 0.00989 | undefined (no attacks) | — | 3,677 |
| Tue | 0.052022 | 0.00641 | 0.5676 | 0.6609 | 5,954 |
| Wed | 0.027546 | 0.01140 | 0.9817 | 0.9785 | 172,203 |
| Thu | 0.024374 | 0.01176 | 0.9937 | 0.9544 | 74,868 |
| Fri | 0.027878 | 0.00992 | 0.9975 | 0.9887 | 257,203 |

At α = 0.05 the achieved FPR was 0.0417–0.0482 on attack days and **0.0729 on the benign-only Monday**.

### P2 cross-testbed detector metrics with the frozen source thresholds (final RF-03)

| Target day | AP (prevalence) | ROC-AUC | α | achieved FPR | recall | precision | alerts/day | max alerts/hour |
|---|---|---|---|---|---|---|---|---|
| 14-02 (SSH brute force) | 0.176 (0.016) | 0.964 | 0.01 | **0.174** | 1.000 | 0.085 | 1,103,856 | 211,326 |
| | | | 0.05 | **0.346** | 1.000 | 0.045 | 2,101,381 | 333,718 |
| 16-02 (DoS Hulk) | 0.271 (0.244) | 0.623 | 0.01 | **0.161** | 0.016 | 0.030 | 926,832 | 187,833 |
| | | | 0.05 | **0.324** | 0.124 | 0.110 | 2,034,967 | 414,730 |
| Pooled | 0.170 (0.143) | 0.631 | 0.01 | **0.168** | 0.064 | 0.060 | 2,030,688 | 211,326 |

**Cross-testbed FPR degradation:** nominal 0.01 → achieved 0.0064–0.0118 on held-out source days, but
0.161–0.174 on the target days. Benign 99th score percentile: 0.93–0.96 on the target vs 0.034 for the pooled source
out-of-fold benign scores (KS 0.577). Sensitivity target without "Attempted" flows: FPR 0.145 on both days.

### Hourly alert budgets (T-a, hard cap)

| Data | Budget | alerts | recall (flow) | max recall the quota permits | precision | FPR |
|---|---|---|---|---|---|---|
| P1 Tue / Wed / Thu / Fri | q = 1% | 3,216 / 4,962 / 3,616 / 5,470 | 0.124 / 0.018 / 0.023 / 0.016 | 0.280 / 0.018 / 0.024 / 0.017 | 0.27 / 0.61 / 0.46 / 0.76 | 0.0046–0.0075 |
| P2 14-02 | q = 1% | 58,976 | **0.1153** | 0.1418 | 0.184 | 0.0083 |
| P2 14-02 | K = 500 | 6,500 | **0.0000** | 0.0106 | 0.000 | 0.0011 |
| P2 16-02 | q = 1% | 73,897 | **0.0006** | 0.0134 | 0.015 | 0.0130 |
| P2 16-02 | K = 500 | 5,500 | **0.0000** | 0.0003 | 0.000 | 0.0010 |

In the three P2 hourly batches that contained attacks, all 500 alerts per hour under K = 500 were benign.

### Annotation-budget estimation (P2, b labels per hour, 1,000 replicates; 95% CI coverage)

| Day | b | FPR under T-b(0.01): coverage | recall under T-b(0.01): coverage | recall under T-a(q): coverage |
|---|---|---|---|---|
| 14-02 | 100 / 250 / 500 | 0.949 / 0.946 / 0.960 | 1.000 (census recall 1.000) | 0.796 / 0.898 / 0.927 |
| 16-02 | 100 / 250 / 500 | 0.931 / 0.944 / 0.950 | 0.688 / 0.896 / 0.933 | **0.049 / 0.106 / 0.230** |

Bias, RMSE and interval widths for every cell are in `src/FASTTRACK_V23/P2_RUN/P2_RESULTS.md`.

## Known limitations

- One source dataset, one target dataset, two target days, one selected model family.
- The P2 days were previously inspected during dataset auditing, so they are a development domain, not a blind test.
- The selection score (mAP_day) is selection-optimistic: it is computed on the folds used for selection.
- The source reference distribution (out-of-fold scores of the CV models) and the target scores (final model) come
  from different models.
- Recall estimation by audit is unreliable when the true recall is near zero.
- The guarded runners are macOS-specific and pinned to exact library versions.

## Citation

See `CITATION.cff`. Please also cite the dataset papers listed in `docs/DATASETS.md`.

## Contact

Abtin Asghari (corresponding author), 0025636170@iau.ir.
