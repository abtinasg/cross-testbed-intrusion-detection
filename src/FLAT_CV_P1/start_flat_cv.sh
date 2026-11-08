#!/bin/zsh
# Headless Flat-CV launcher (start it from Terminal.app, never from the Claude app: the app's quit cleanup kills
# processes it started). Exit 2 from run_flat_cv.py = low-memory PAUSE (no fit started, no key consumed): wait 60 s and
# resume from the checkpoint. Any other exit (0 complete, 1 refused/stopped, 3 fit failure, 4 terminated) ends the loop:
# no retry after a real failure. Every start/exit/signal is written to the console log and cv_run/LAUNCHER_STATUS.
cd "${0:A:h}"
PY=${PY:-python3}   # release: set PY to your pinned Python 3.13.5 interpreter (original: local Anaconda path)
ST=cv_run/LAUNCHER_STATUS
ts() { date -u +%FT%TZ; }
if [[ -f $ST ]] && grep -q '^RUNNING' $ST; then
  old=$(awk '{print $2}' $ST)
  if kill -0 "$old" 2>/dev/null; then echo "=== REFUSED $(ts): launcher pid $old is still running"; exit 1; fi
  echo "=== NOTE $(ts): previous launcher (pid $old) vanished without recording an exit (killed by SIGKILL, logout or power loss). The runner records any fit in progress as INTERRUPTED."
fi
echo "RUNNING $$ $(ts)" > $ST
trap 'echo "=== launcher pid $$ received SIGTERM $(ts)"; echo "EXITED $$ $(ts) SIGTERM" > $ST; exit 143' TERM
trap 'echo "=== launcher pid $$ received SIGINT $(ts)"; echo "EXITED $$ $(ts) SIGINT" > $ST; exit 130' INT
while true; do
  echo "=== launch $(ts) launcher pid $$"
  /usr/bin/caffeinate -dims "$PY" -I -B run_flat_cv.py --execute --authorize GO-FLATCV-P1-3ba9d56dfd26 "$@"
  rc=$?
  echo "=== exit $rc $(ts)"
  [[ $rc -eq 2 ]] || break
  sleep 60
done
if [[ $rc -eq 0 ]]; then "$PY" -I -B summarize_flat_cv.py; fi
echo "EXITED $$ $(ts) rc=$rc" > $ST
echo "=== launcher finished rc=$rc $(ts)"
