#!/usr/bin/env bash
# Sequentially runs AIME and LiveBench-Math dynamic_fewshot experiments
# in the background. Designed to survive SSH disconnect.
#
# Usage:
#   ./scripts/run_dynamic_fewshot_overnight.sh            # launch in background
#   ./scripts/run_dynamic_fewshot_overnight.sh --fg       # run in foreground (debug)
#
# Logs:
#   logs/overnight-<timestamp>/
#     aime.log         full stdout/stderr of AIME run
#     livebench.log    full stdout/stderr of LiveBench run
#     status.txt       start/finish timestamps + exit codes
#     pid              parent wrapper PID
#
# Monitoring:
#   tail -f logs/overnight-<timestamp>/aime.log
#   tail -f logs/overnight-<timestamp>/livebench.log
#   cat  logs/overnight-<timestamp>/status.txt

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Reuse a pre-assigned log dir (set by the outer invocation before detaching)
# so the --fg child writes to the same place the user was told to watch.
if [[ -n "${OVERNIGHT_LOG_DIR:-}" ]]; then
    LOG_DIR="$OVERNIGHT_LOG_DIR"
else
    TS="$(date +%Y%m%d_%H%M%S)"
    LOG_DIR="logs/overnight-${TS}"
fi
mkdir -p "$LOG_DIR"

STATUS="$LOG_DIR/status.txt"
AIME_LOG="$LOG_DIR/aime.log"
LIVE_LOG="$LOG_DIR/livebench.log"

run_sequence() {
    {
        echo "=== overnight run started at $(date -Is) ==="
        echo "repo: $REPO_ROOT"
        echo "log_dir: $LOG_DIR"
    } >> "$STATUS"

    echo "[$(date -Is)] starting AIME dynamic_fewshot..." >> "$STATUS"
    uv run python -m examples.aime_math.dynamic_fewshot \
        > "$AIME_LOG" 2>&1
    aime_rc=$?
    echo "[$(date -Is)] AIME finished with exit=$aime_rc" >> "$STATUS"

    echo "[$(date -Is)] starting LiveBench-Math dynamic_fewshot..." >> "$STATUS"
    uv run python -m examples.livebench_math.dynamic_fewshot \
        > "$LIVE_LOG" 2>&1
    live_rc=$?
    echo "[$(date -Is)] LiveBench finished with exit=$live_rc" >> "$STATUS"

    echo "=== overnight run completed at $(date -Is) ===" >> "$STATUS"
    echo "AIME exit=$aime_rc, LiveBench exit=$live_rc" >> "$STATUS"
}

if [[ "${1:-}" == "--fg" ]]; then
    run_sequence
    exit $?
fi

# Detach: new session (no controlling TTY), stdin redirected, parent exits.
# Using setsid ensures SIGHUP from SSH disconnect does not reach the child.
# We re-invoke this same script with --fg inside the detached session, and
# pass the target log_dir via env so both invocations share it.
export OVERNIGHT_LOG_DIR="$LOG_DIR"
setsid nohup "$0" --fg \
    < /dev/null > "$LOG_DIR/wrapper.log" 2>&1 &

WRAPPER_PID=$!
disown "$WRAPPER_PID" 2>/dev/null || true
echo "$WRAPPER_PID" > "$LOG_DIR/pid"

cat <<EOF
Launched overnight run in background.

  log_dir : $LOG_DIR
  pid     : $WRAPPER_PID

Watch progress:
  tail -f $AIME_LOG
  tail -f $LIVE_LOG
  cat    $STATUS

Check if still running:
  ps -p $WRAPPER_PID -o pid,etime,cmd
EOF
