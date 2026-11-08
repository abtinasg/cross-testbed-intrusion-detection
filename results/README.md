# Aggregate results (index)

Every result file stays where its stage wrote it, inside `src/`, because the later stages and the table/figure scripts
read them from there. No file below contains row-level data.

| Stage | Result | File(s) |
|---|---|---|
| Paper Tables 1–5 (generated) | Markdown | [`PAPER_TABLES.md`](PAPER_TABLES.md) |
| Paper Figures 1–2 | PNG, 300 dpi | `src/paper/figures/` |
| P1 first real fit (LR-01 held out on Tuesday) | per-fit record | `src/RG8_V22_A1_SMOKE/smoke_run/RESULT.json` |
| P1 leave-one-day-out CV, 60 fits | AP per configuration and day, mAP_day ranking | `src/FLAT_CV_P1/cv_run/FLAT_CV_RESULTS.{json,md}` |
| | per-fit ledger (hashes of every score vector, timings, guard summary) and the one recovery (LR-02/Tue) | `src/FLAT_CV_P1/cv_run/LEDGER.jsonl`, `RECOVERY_LR-02__Tue.json`, `STOPPED_LR-02__Tue__attempt1.json` |
| P1 close-out and RF-03 selection | PASS, AP recomputed from stored vectors | `src/P1_FLAT_CV_CLOSEOUT/P1_FLAT_CV_VERIFICATION.{json,md}` |
| T-b source-only calibration, 20 inner fits | τ_k, achieved FPR, recall, precision, alerts per day and hour, KS | `src/TB_CALIBRATION_RF03/TB_RESULTS.{json,md}`, `tb_calibration/RF-03/outer=*.json` |
| | ledger and the one recovery (Fri/Tue) | `src/TB_CALIBRATION_RF03/tb_run/` |
| P1 hourly alert budgets (T-a) and final-detector thresholds | q = 0.01 and K = 500 on Tue–Fri; τ_final | `src/FASTTRACK_V23/P1_TA_RESULTS.{json,md}`, `FINAL_DETECTOR_TB_THRESHOLDS.json` |
| P1 freeze and P2 pre-registration | PASS; hashes of settings, thresholds and plan | `src/FASTTRACK_V23/P1_FREEZE_CHECK.json`, `P2_PREREGISTRATION.json` |
| P2 final fit and label-blind scoring | ledger, Wed14 recovery under the V23 guard, seal (SHA-256 of every prediction file, model and artefact) | `src/FASTTRACK_V23/P2_RUN/p2_run/`, `P2_SCORES_SEALED.json`, `SEAL_VERIFICATION.json` |
| P2 evaluation | detector metrics, cross-testbed FPR, budgets, annotation-budget estimation (bias, RMSE, coverage, CI width) | `src/FASTTRACK_V23/P2_RUN/P2_RESULTS.{json,md}` |

The seal and ledgers refer to score files (`*.npy`), a Parquet file and a model file that are **not** in this release
(see `docs/DATASETS.md`). Their SHA-256 values let you check a re-run against the original outputs.
