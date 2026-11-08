#!/bin/zsh
# One-time owner-authorised T-b recovery (Fri/Tue attempt 2), then Fri/Wed and Fri/Thu. Runs start_tb.sh with the token.
cd "${0:A:h}"
exec ./start_tb.sh --recovery-token GO-RECOVER-TB-FRI-TUE-8c3a6e7f980c
