# P1 T-a alert-budget results, RF-03 (flat outer scores; retrospective hourly batches, T2 availability, hard cap)

Settings: OPERATIONAL_SETTINGS_v1.json SHA-256 `d1a6585ac71b8cf9c41d0901bc878aaf60677f4d94f0ae8a23f14c52f5f2a683` (written before this run).

| Day | Budget | alerts | recall (flow) | max recall allowed by quota | recall (hour-block mean) | precision | benign alerts | FPR | attack hours | zero-quota hours |
|---|---|---|---|---|---|---|---|---|---|---|
| Tue | q=0.01 | 3,216 | 0.1243 (862/6,933) | 0.2804 | 0.0912 | 0.2680 | 2,354 | 0.00747 | 5 | 0 |
| Tue | K=500 | 4,676 | 0.1196 (829/6,933) | 0.2886 | 0.0947 | 0.1773 | 3,847 | 0.01221 | 5 | 0 |
| Wed | q=0.01 | 4,962 | 0.0176 (3,023/171,645) | 0.0177 | 0.3204 | 0.6092 | 1,939 | 0.00597 | 4 | 0 |
| Wed | K=500 | 5,000 | 0.0087 (1,500/171,645) | 0.0088 | 0.3292 | 0.3000 | 3,500 | 0.01077 | 4 | 0 |
| Thu | q=0.01 | 3,616 | 0.0231 (1,663/71,907) | 0.0237 | 0.3565 | 0.4599 | 1,953 | 0.00673 | 4 | 1 |
| Thu | K=500 | 4,502 | 0.0150 (1,080/71,907) | 0.0154 | 0.4453 | 0.2399 | 3,422 | 0.01179 | 4 | 0 |
| Fri | q=0.01 | 5,470 | 0.0162 (4,132/254,946) | 0.0173 | 0.2652 | 0.7554 | 1,338 | 0.00457 | 6 | 0 |
| Fri | K=500 | 4,500 | 0.0075 (1,906/254,946) | 0.0086 | 0.2917 | 0.4236 | 2,594 | 0.00887 | 6 | 0 |

## Final-detector T-b thresholds (for P2; P1-only)

Pooled benign flat-OOF scores: 1,594,545; alpha 0.01: tau = 0.033896 (calibration FPR 0.01000); alpha 0.05: tau = 0.003304 (calibration FPR 0.05000)
