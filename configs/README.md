# Frozen configurations (index)

The configuration files stay where the code reads them, inside `src/`. Most are hash-pinned by the code, so moving
them would break the checks. Each was written before the results that depend on it existed.

| What | File | Pinned / checked by |
|---|---|---|
| Model grid F3: 12 configurations (LR-01..03, HGB-01..05, RF-01..04), hyperparameters, seed 20261104 | `src/next_workstream/CONFIG_GRID_F3_v1.csv` | `F3_SHA256` in `src/PILOT_BINDING_G2/f3_binding.py` |
| Estimator pins per family (library defaults made explicit, `random_state`, `n_jobs`) | `src/PILOT_BINDING_G2/ADAPTER_PINNED_PARAMS_v1_PROPOSED.json` | `PINS_SHA256` in `src/PILOT_BINDING_G2/model_adapter_g2.py` |
| Feature schema: 72 primary features in order, types, units | `src/FEATURE_SCHEMA_PRIMARY_72_DRAFT.csv` | `SCHEMA_ORDERED_SHA256` in `src/R5_FREEZE_F7/r5_final.py` |
| Expected 91-column CSV header | `src/preprocessing_review/EXPECTED_CSV_HEADER_91.txt` | `src/R5_FREEZE_F7/r5_p1_source.py` |
| P1 source allow-list (5 day CSVs + SHA-256; P2 files listed as forbidden for training) | `src/correction_package_C1/provenance/P1_SOURCE_ALLOWLIST.json` | P1 source loader |
| T-b α prespecification (0.01 primary, 0.05 sensitivity) | `src/RG8_V21/TB_ALPHA_PRESPECIFICATION.json` | `calibrate_tb.py` |
| Operational settings: hourly batching, q = 0.01, K = 500, tie key, audit design (b = 100/250/500, seed) | `src/FASTTRACK_V23/OPERATIONAL_SETTINGS_v1.json` (SHA-256 `d1a6585a…`) | P1 freeze, P2 runner (token) |
| Alert-budget rationale and P2 evaluation plan (written before P2) | `src/FASTTRACK_V23/03_ALERT_BUDGET_SETTINGS.md`, `src/FASTTRACK_V23/04_P2_EVALUATION_PLAN.md` | plan hash in the pre-registration |
| P2 pre-registration (settings, thresholds and plan hashes; write-once) | `src/FASTTRACK_V23/P2_PREREGISTRATION.json` | `run_p2.py`, `evaluate_p2.py` |
| Final-detector thresholds τ(0.01) = 0.033896, τ(0.05) = 0.003304 | `src/FASTTRACK_V23/FINAL_DETECTOR_TB_THRESHOLDS.json` | pre-registration |
| Resource-guard uniform policy (limits, pinned environment) | `src/RG8_V21/RG8_POLICY_PROPOSED.json` (interpreter path redacted) | `POLICY_SHA256`; see `docs/RELEASE_NOTES.md` |
| Policy of the first real fit (LR-01/Tue) | `src/RG8_V22_A1_SMOKE/A1_SMOKE_POLICY.json` (interpreter path redacted) | `A1_POLICY_SHA256` |

Selection criterion and folds: leave-one-day-out over Mon–Fri; mAP_day = mean AP over Tue–Fri; the largest unrounded
value wins; only bitwise-exact ties are broken, in favour of the simpler configuration (`src/FLAT_CV_P1/run_flat_cv.py`,
`src/P1_FLAT_CV_CLOSEOUT/verify_p1_flat_cv.py`).
