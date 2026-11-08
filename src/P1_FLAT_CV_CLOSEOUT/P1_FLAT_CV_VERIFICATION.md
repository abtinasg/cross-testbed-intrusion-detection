# P1 Flat-CV close-out verification: **PASS**

- Unique OK keys: 60/60 (ledger records: 61; the one non-OK record is the superseded LR-02/Tue attempt 1).
- Recovery chain LR-02/Tue: only_non_ok_is_LR02_Tue_attempt1=OK, replaced_ledger_line_hash=OK, archived_STOPPED_hash=OK, token_matches_STOPPED=OK, STOPPED_json_absent_now=OK, INTENT_json_absent_now=OK, attempt2_record_OK=OK, attempt1_evidence_kept=OK, attempt2_evidence_present=OK, attempt1_left_no_score_files=OK, launcher_final_rc0=OK
- All 180 vectors: SHA-256 verified, read-only, finite, pinned row counts; labels/row ids identical across the 12 configs per day.
- AP recomputed from stored vectors = ledger AP for all 48 attack-day keys; Monday AP undefined for all 12.
- Selected: **RF-03** (config_sha256 `8a93f95e96226ac2…`), margin to RF-04 = 0.001421; exact ties: none.
- LEDGER.jsonl SHA-256 `cc0226a722d0f627cd0f69ed081fbddf543980587116c0670d7d78f74020e2eb`

| Rank | Config | AP Tue | AP Wed | AP Thu | AP Fri | mAP_day (selection-optimistic) |
|---|---|---|---|---|---|---|
| 1 | RF-03 | 0.4568 | 0.9930 | 0.9839 | 0.9998 | 0.858368 |
| 2 | RF-04 | 0.4515 | 0.9928 | 0.9838 | 0.9996 | 0.856947 |
| 3 | HGB-03 | 0.6666 | 0.7504 | 0.9587 | 0.9984 | 0.843497 |
| 4 | HGB-05 | 0.4992 | 0.8893 | 0.9483 | 0.9990 | 0.833922 |
| 5 | RF-02 | 0.3486 | 0.9935 | 0.9835 | 0.9996 | 0.831309 |
| 6 | RF-01 | 0.3332 | 0.9923 | 0.9825 | 0.9997 | 0.826943 |
| 7 | HGB-02 | 0.3327 | 0.9569 | 0.9706 | 0.9995 | 0.814938 |
| 8 | HGB-01 | 0.2225 | 0.9703 | 0.9745 | 0.9990 | 0.791544 |
| 9 | HGB-04 | 0.6309 | 0.4245 | 0.9712 | 0.9879 | 0.753642 |
| 10 | LR-01 | 0.0111 | 0.5104 | 0.8932 | 0.9983 | 0.603223 |
| 11 | LR-02 | 0.0112 | 0.4479 | 0.7746 | 0.9979 | 0.557872 |
| 12 | LR-03 | 0.0116 | 0.4695 | 0.6703 | 0.9974 | 0.537202 |

Distinct score values per held-out day (RF-03): Tue 40,904, Wed 39,445, Thu 20,674, Fri 55,124, Mon 57,420

Problems: none
