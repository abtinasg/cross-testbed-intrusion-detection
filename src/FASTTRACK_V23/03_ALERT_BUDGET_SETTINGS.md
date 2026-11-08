# Operational settings — proposal for freeze (P1-only reasoning)

Machine-readable record: `OPERATIONAL_SETTINGS_v1.json` (read-only, SHA-256 `d1a6585ac71b8cf9c41d0901bc878aaf60677f4d94f0ae8a23f14c52f5f2a683`). It was written **before** any T-a result on P1 and before any P2 scoring. No P2 outcome informed it.

| Setting | Value | Why |
|---|---|---|
| **Primary hourly alert budget** | relative quota **q = 0.01**, k_h = ⌊0.01·N_h⌋ | Same nominal alert fraction as the primary T-b level α = 0.01. Comparing T-a and T-b then separates *ranking under a fixed volume* from *threshold transfer*. On P1 hours (0.7k–59k flows) this is ≤ ≈ 590 alerts per hour |
| **One sensitivity setting** | absolute capacity **K = 500 alerts/hour**, k_h = min(K, N_h) | Middle of the A1b.2 candidates {100, 500, 2,000}. Does not scale with traffic volume, so it tests whether conclusions hold when the budget is a fixed analyst-side capacity |
| Quota selection | **Hard cap** (A1b.1): rank by score, then tie key; alert an indistinguishable group only if it fits; no backfill. Never exceeds the quota | Frozen primary rule |
| Tie handling | A1.2 key: SHA-256(seed 20261104 ‖ availability time ‖ raw 72-feature vector), first 8 bytes. Label-free, row-order-free | Frozen |
| Batches | Hourly, membership by T2 availability a = Timestamp + Flow Duration, UTC, per capture file; δ_score = 0 | Frozen primary (A1.4) |
| Small or empty hours | ⌊q·N_h⌋ = 0 → no alerts (counted); N_h = 0 → excluded (counted) | Frozen |
| Deferred (not run) | overshoot policy, T-a′/T-a′-S, δ_score = 60 s, T2c/T2+120, K = 100/2,000, q = 0.05 | Fast-track scope |

## Three different budgets — not interchangeable
| Quantity | Symbol | Meaning in this paper |
|---|---|---|
| **Alert budget** | q or K | How many flows the detector may *raise* per hour. Defines T-a. |
| **Analyst review capacity** | R | How many raised alerts analysts can *examine* per hour. **Not modelled**: we assume R = k_h, i.e. every alert inside the quota is reviewed. Stated as an assumption. |
| **Annotation budget** | b ∈ {100, 250, 500}/h | How many flows get a *ground-truth label* per hour for **estimating** the detector's recall and FPR. The audit sample is drawn from all completed flows of the hour (alerted or not, design M0, stratified SRSWOR). These labels never train, adapt or threshold the detector. Grid unchanged from Decision Brief v1 §6. |

## Metrics needed for a defensible conference paper
1. P1: per-day AP and mAP_day (*selection-optimistic*), Monday AP undefined.
2. T-b (α = 0.01, 0.05): nominal vs achieved FPR on every held-out day, Monday included; recall; alerts per day and per hour; score-scale diagnostic.
3. T-a (q = 0.01; K = 500): flow-level recall-at-budget, precision-at-budget, maximum recall the quota permits, hour-block recall mean.
4. P2: the same as 1–3 on the two target days (AP instead of mAP_day), plus per-episode recall.
5. Annotation-budget estimation on P2: bias, CI coverage, CI width, P(zero audited positives) for b ∈ {100, 250, 500}.
6. Resource use per fit (peak memory, wall time).
No significance tests are planned; n = 4 P1 attack days and 2 P2 episodes are reported individually.
