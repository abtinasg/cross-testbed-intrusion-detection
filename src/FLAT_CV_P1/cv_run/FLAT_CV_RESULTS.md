# Flat CV (P1, primary target): 60/60 OK

| Config | AP Tue | AP Wed | AP Thu | AP Fri | **mAP_day** | Mon AP | Mon benign share score>=0.5 | Mon q99.9 | peak GiB | fit time |
|---|---|---|---|---|---|---|---|---|---|---|
| *prevalence* | 0.0215 | 0.3456 | 0.1986 | 0.4656 | | | | | | |
| RF-03 | 0.4568 | 0.9930 | 0.9839 | 0.9998 | **0.8584** | undefined | 0.00002 (9) | 0.1636 | 2.64 | 14 min |
| RF-04 | 0.4515 | 0.9928 | 0.9838 | 0.9996 | **0.8569** | undefined | 0.00002 (8) | 0.1622 | 2.69 | 20 min |
| HGB-03 | 0.6666 | 0.7504 | 0.9587 | 0.9984 | **0.8435** | undefined | 0.00051 (188) | 0.0517 | 2.86 | 10 min |
| HGB-05 | 0.4992 | 0.8893 | 0.9483 | 0.9990 | **0.8339** | undefined | 0.00045 (169) | 0.3519 | 2.77 | 10 min |
| RF-02 | 0.3486 | 0.9935 | 0.9835 | 0.9996 | **0.8313** | undefined | 0.00002 (7) | 0.1575 | 2.67 | 18 min |
| RF-01 | 0.3332 | 0.9923 | 0.9825 | 0.9997 | **0.8269** | undefined | 0.00002 (6) | 0.1428 | 2.67 | 13 min |
| HGB-02 | 0.3327 | 0.9569 | 0.9706 | 0.9995 | **0.8149** | undefined | 0.00023 (86) | 0.0845 | 2.87 | 12 min |
| HGB-01 | 0.2225 | 0.9703 | 0.9745 | 0.9990 | **0.7915** | undefined | 0.00052 (195) | 0.2116 | 2.82 | 10 min |
| HGB-04 | 0.6309 | 0.4245 | 0.9712 | 0.9879 | **0.7536** | undefined | 0.00004 (14) | 0.0002 | 2.77 | 14 min |
| LR-01 | 0.0111 | 0.5104 | 0.8932 | 0.9983 | **0.6032** | undefined | 0.00573 (2130) | 0.8465 | 2.87 | 10 min |
| LR-02 | 0.0112 | 0.4479 | 0.7746 | 0.9979 | **0.5579** | undefined | 0.00542 (2013) | 0.8933 | 2.86 | 10 min |
| LR-03 | 0.0116 | 0.4695 | 0.6703 | 0.9974 | **0.5372** | undefined | 0.00435 (1616) | 0.9051 | 2.93 | 11 min |

Best by mAP_day: **RF-03** (selection-optimistic (flat CV; not an unbiased estimate)).
Monday: AP_UNDEFINED_NO_POSITIVES; the score>=0.5 share is descriptive only (frozen Monday FPR requires T-b).
Superseded failed attempts (replaced by an authorised recovery attempt): LR-02/Tue attempt 1 WRAPPER_ERROR (Terminated: signal SIGTERM)
Integrity problems: none
