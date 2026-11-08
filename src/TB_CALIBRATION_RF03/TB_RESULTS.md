# RF-03 T-b source-only calibration: **PASS**

tau_k from benign inner-OOF scores of the 3-day RF-03 models on the 4 training days of outer fold k; applied to the stored flat RF-03 outer scores. Alert iff score > tau_k. Labels of day k used only for the metrics.

## alpha = 0.01 (primary)

| Outer day | n benign pooled | tau_k | calib. FPR | achieved FPR | benign alerts / benign | recall | precision | alerts/day | alerts/h max | KS |
|---|---|---|---|---|---|---|---|---|---|---|
| Mon | 1,222,921 | 0.045819 | 0.01000 | **0.00989** | 3,677 / 371,624 | undefined (no attacks) | 0.0000 | 3,677 | 878 | 0.2867 |
| Tue | 1,279,400 | 0.052022 | 0.00954 | **0.00641** | 2,019 / 315,145 | 0.5676 (3,935/6,933) | 0.6609 | 5,954 | 2,888 | 0.3710 |
| Wed | 1,269,549 | 0.027546 | 0.00976 | **0.01140** | 3,705 / 324,996 | 0.9817 (168,498/171,645) | 0.9785 | 172,203 | 155,430 | 0.1289 |
| Thu | 1,304,376 | 0.024374 | 0.00995 | **0.01176** | 3,412 / 290,169 | 0.9937 (71,456/71,907) | 0.9544 | 74,868 | 70,331 | 0.1327 |
| Fri | 1,301,934 | 0.027878 | 0.00884 | **0.00992** | 2,904 / 292,611 | 0.9975 (254,299/254,946) | 0.9887 | 257,203 | 139,543 | 0.3509 |

## alpha = 0.05 (sensitivity)

| Outer day | n benign pooled | tau_k | calib. FPR | achieved FPR | benign alerts / benign | recall | precision | alerts/day | alerts/h max | KS |
|---|---|---|---|---|---|---|---|---|---|---|
| Mon | 1,222,921 | 0.003658 | 0.05000 | **0.07286** | 27,078 / 371,624 | undefined (no attacks) | 0.0000 | 27,078 | 6,568 | 0.2867 |
| Tue | 1,279,400 | 0.004652 | 0.05000 | **0.04168** | 13,136 / 315,145 | 0.9234 (6,402/6,933) | 0.3277 | 19,538 | 5,082 | 0.3710 |
| Wed | 1,269,549 | 0.002000 | 0.04992 | **0.04338** | 14,098 / 324,996 | 0.9992 (171,514/171,645) | 0.9240 | 185,612 | 158,123 | 0.1289 |
| Thu | 1,304,376 | 0.001739 | 0.04992 | **0.04647** | 13,485 / 290,169 | 0.9977 (71,739/71,907) | 0.8418 | 85,224 | 72,025 | 0.1327 |
| Fri | 1,301,934 | 0.002773 | 0.05000 | **0.04823** | 14,112 / 292,611 | 0.9997 (254,881/254,946) | 0.9475 | 268,993 | 140,966 | 0.3509 |

Inner AP (descriptive; 3-day models): Mon/Tue: 0.4915, Mon/Wed: 0.9910, Mon/Thu: 0.9852, Mon/Fri: 0.9993, Tue/Mon: undef, Tue/Wed: 0.9968, Tue/Thu: 0.9826, Tue/Fri: 0.9996, Wed/Mon: undef, Wed/Tue: 0.5591, Wed/Thu: 0.9830, Wed/Fri: 0.9970, Thu/Mon: undef, Thu/Tue: 0.0960, Thu/Wed: 0.9905, Thu/Fri: 0.9929, Fri/Mon: undef, Fri/Tue: 0.3819, Fri/Wed: 0.9217, Fri/Thu: 0.8918

Problems: none
