#!/bin/zsh
# P2 launcher (Terminal.app). Exit 2 = low-memory pause: wait 60 s and resume. Any other exit ends the loop (no retry).
cd "${0:A:h}"
PY=${PY:-python3}   # release: set PY to your pinned Python 3.13.5 interpreter (original: local Anaconda path)
mkdir -p p2_run
ST=p2_run/LAUNCHER_STATUS
exec > >(tee -a p2_console.log) 2>&1
ts() { date -u +%FT%TZ; }
if [[ -f $ST ]] && grep -q '^RUNNING' $ST && kill -0 "$(awk '{print $2}' $ST)" 2>/dev/null; then echo "=== REFUSED: launcher already running"; exit 1; fi
echo "RUNNING $$ $(ts)" > $ST
trap 'echo "EXITED $$ $(ts) SIGTERM" > $ST; exit 143' TERM
trap 'echo "EXITED $$ $(ts) SIGINT" > $ST; exit 130' INT
while true; do
  echo "=== launch $(ts) launcher pid $$"
  /usr/bin/caffeinate -dims "$PY" -I -B run_p2.py --execute --authorize GO-P2-RF03-d1a6585ac71b "$@"
  rc=$?
  echo "=== exit $rc $(ts)"
  [[ $rc -eq 2 ]] || break
  sleep 60
done
echo "EXITED $$ $(ts) rc=$rc" > $ST
echo "=== launcher finished rc=$rc $(ts)"
