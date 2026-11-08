# P2 cross-testbed evaluation — ready-to-run plan (NOT EXECUTED; awaiting owner authorization)

**Date:** 2026-11-07. **Status:** plan only. No P2 row has been scored. No P2 label has been read in this phase.
**Detector:** RF-03 (F3 `config_sha256 8a93f95e…be2c2f`), selected by flat P1 mAP_day (`P1_FLAT_CV_CLOSEOUT/`).
**Settings:** `OPERATIONAL_SETTINGS_v1.json` (SHA-256 `d1a6585ac71b8cf9c41d0901bc878aaf60677f4d94f0ae8a23f14c52f5f2a683`), α from `RG8_V21/TB_ALPHA_PRESPECIFICATION.json`. Thresholds for the final detector are computed from P1 only (`FINAL_DETECTOR_TB_THRESHOLDS.json`, step 0).

## 1. Scope (unchanged from the approved brief)
| Item | Value |
|---|---|
| Target data | P2 = CSE-CIC-IDS2018 (DistriNet improved), **only** `Wednesday-14-02-2018.csv` (SHA-256 `4ceed4ba…8fdb`, 5,898,350 rows) and `Friday-16-02-2018.csv` (`a16b76a4…7622`, 7,390,266 rows) |
| Not used | All other 2018 days; P3 (not acquired, not authorised) |
| Target definition | Primary: positive iff Label ≠ BENIGN and Attempted Category = −1. Sensitivity: Attempted rows excluded |
| Features | The same 72-column primary schema, parsed by name; same R3/R4 artefact as the final P1 fit (no P2 statistic enters preprocessing) |
| Known composition (Stage 02A-0, **already inspected**) | 14-02: SSH-BruteForce 94,197 (1.60%), one episode 18:01:50–19:32:30; FTP-BruteForce 193,354 rows are all *Attempted* (→ benign under the primary target). 16-02: DoS Hulk 1,803,160 (24.40%), one episode 17:45:27–17:58:23; FTP-BruteForce 105,606 *Attempted* |
| Status of P2 | Development domain, **not** an untouched test set. Its label distributions, episode windows and hourly composition were inspected during the Stage 02A-0 feasibility audit. Every P2 number is reported with this disclosure |

## 2. Execution steps (each guarded by RG8 V21 + v20 policy; no retries; write-once outputs)
| Step | What | Reads | Writes | Leakage control |
|---|---|---|---|---|
| **0. Freeze** | Owner approves `OPERATIONAL_SETTINGS_v1.json`, `FINAL_DETECTOR_TB_THRESHOLDS.json` and this plan by SHA-256. A `P2_PREREGISTRATION.json` binds: detector config hash, τ(0.01), τ(0.05), q, K, b grid, seeds, metric list, code hashes | P1 outputs only | `P2_PREREGISTRATION.json` (read-only) | Written before any P2 score exists |
| **1. Final source fit** | One fit of RF-03 on **all five P1 days** (2,099,976 rows), F7 P-TREE preprocessing fitted on the same rows. Persist the estimator (joblib) and preprocessing artefact with SHA-256 | P1 only | `final_model/RF-03_P1all.joblib`, artefact JSON, hashes | P1 allow-list enforced. Needs a minimal **final-fold wrapper** (training days = all five; `FoldSpec` requires an outer day). Frozen F7 code is called unchanged |
| **2. Blind scoring** | Stream each P2 file in chunks of 250,000 rows. Read **only** `id`, `Timestamp`, `Flow Duration` and the 72 schema columns. Transform with the step-1 artefact; `predict_proba[:, 1]`; compute T2 availability time and A1.2 tie key | P2 features only | `scores_P2_<day>.npy`, `row_ids`, `availability_us`, `tie_keys` (write-once, SHA-256) | **Label and Attempted columns are never parsed in this step** (column projection). A P2 scoring-only allow-list (the two hashes above) is separate from the P1 training allow-list, which keeps forbidding them for fitting |
| **3. Seal** | Hash all step-2 vectors into `P2_SCORES_SEALED.json` | step-2 outputs | seal file | After this, scores cannot change; thresholds and budgets were fixed in step 0 |
| **4. Evaluation** | Read `Label` and `Attempted Category`; compute every metric in §3 from the sealed vectors | sealed vectors + P2 labels | `P2_RESULTS.json/.md` | Labels used for metrics only; no threshold, budget, model or feature change afterwards |

**Chronological replay restrictions.** Hourly batches by T2 availability time a = Timestamp + Flow Duration (UTC), per capture file; the 1,193 post-midnight rows of the 14-02 capture form its last batch (hours never cross capture files). All T-a decisions are retrospective batch decisions at the end of each hour; no real-time or causal claim.

**Resources (HYPOTHESIS until measured).** Step 1: one fit on 2.10 M rows, expected peak ≈ 3.0–3.3 GiB (flat 4-day RF-03 fits peaked at 2.64 GiB; ≤ 4.75 GiB policy limit), ≈ 4–6 min. Step 2: chunked, < 1.5 GiB per chunk plus the model; 13.3 M rows; ≈ 20–60 min in total. Disk ≈ 0.5 GiB. Each P2 day is one guarded job (1-h limit each).

**Known execution risk.** The RG8 V21 sampler can fail closed (`MEASUREMENT_UNAVAILABLE`) while a worker that has finished its job is exiting: this happened once in 78 real fits (T-b, outer Fri / inner Tue). The P2 runner should accept a pre-authorised single recovery per key for exactly this failure class at stage `job_returned`. The frozen guard code is not changed.

## 3. Metrics (two separate result blocks)
### Block A — ordinary detector performance (census over all P2 rows)
1. Per-day **AP** with its prevalence baseline (1.60%, 24.40%); ROC-AUC secondary.
2. **T-b** at α = 0.01 (primary) and 0.05: nominal α vs **achieved FPR**, recall, precision, alerts per day and per hour (max, median).
3. FPR on the *Attempted* (benign-labelled) FTP rows reported separately; sensitivity target (Attempted excluded) for items 1–2.
4. Per-episode recall (2 episodes) — the paper's unit of evidence is the episode, not the flow.
5. Score-scale diagnostic: P1 flat-OOF benign vs P2 benign score quantiles and KS distance (evaluation only).

### Block B — budget-constrained operation and estimation
6. **T-a primary** (q = 0.01, hard cap) and **sensitivity** (K = 500/h): recall-at-budget, precision-at-budget, the maximum recall the quota allows (Σ min(k_h, K_h)/Σ K_h), unused capacity, zero-quota hours.
7. **Annotation-budget estimation (M0)**: stratified SRSWOR of b ∈ {100, 250, 500} labels per hour from all flows; combined ratio estimators of per-day recall and FPR under T-b (α = 0.01) and under T-a primary; 1,000 replicates (seed 20261107). Reported against the census value: bias, empirical 95% CI coverage, mean CI width, P(zero audited attack flows), number of attack-containing hours. The ≥ 30 audited positives rule is a screening heuristic only.

### Not claimed
- No adaptive model is trained or evaluated (Track B not run). No statement that adaptation improves detection.
- No significance tests between detectors (one detector, two days, two episodes). No pooled flow-level CI is presented as episode-level uncertainty.
- No natural-drift, real-time, online or cross-network claim.

## 4. Authorization requested
Token for steps 1–4: `GO-P2-RF03-d1a6585ac71b` (prefix + first 12 hex of the settings SHA-256). Step 0 (freeze) is a document approval. Engineering still to write after approval: final-fold wrapper (step 1), P2 scoring-only loader with column projection (step 2), evaluation script (step 4). Estimated effort: one working session; runtime ≈ 1–1.5 h.
