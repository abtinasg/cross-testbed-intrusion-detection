#!/bin/zsh
# Headless T-b launcher for RF-03 (start from Terminal.app, never from the Claude app). Exit 2 = low-memory PAUSE
# (no fit started): wait 60 s and resume. Any other exit ends the loop (no retry after a real failure).
cd "${0:A:h}"
PY=${PY:-python3}   # release: set PY to your pinned Python 3.13.5 interpreter (original: local Anaconda path)
mkdir -p tb_run
ST=tb_run/LAUNCHER_STATUS
LOG=tb_console.log
ts() { date -u +%FT%TZ; }
exec > >(tee -a $LOG) 2>&1
if [[ -f $ST ]] && grep -q '^RUNNING' $ST; then
  old=$(awk '{print $2}' $ST)
  if kill -0 "$old" 2>/dev/null; then echo "=== REFUSED $(ts): launcher pid $old is still running"; exit 1; fi
  echo "=== NOTE $(ts): previous launcher (pid $old) vanished without recording an exit."
fi
echo "RUNNING $$ $(ts)" > $ST
trap 'echo "=== launcher pid $$ received SIGTERM $(ts)"; echo "EXITED $$ $(ts) SIGTERM" > $ST; exit 143' TERM
trap 'echo "=== launcher pid $$ received SIGINT $(ts)"; echo "EXITED $$ $(ts) SIGINT" > $ST; exit 130' INT
while true; do
  echo "=== launch $(ts) launcher pid $$"
  /usr/bin/caffeinate -dims "$PY" -I -B run_tb.py --execute --authorize GO-TB-RF03-P1-3ba9d56dfd26 "$@"
  rc=$?
  echo "=== exit $rc $(ts)"
  [[ $rc -eq 2 ]] || break
  sleep 60
done
if [[ $rc -eq 0 ]]; then "$PY" -I -B calibrate_tb.py; fi
echo "EXITED $$ $(ts) rc=$rc" > $ST
echo "=== launcher finished rc=$rc $(ts)"
