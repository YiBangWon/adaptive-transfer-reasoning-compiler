#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <session-name> <log-path> [benchmark args...]" >&2
  exit 1
fi

SESSION_NAME="$1"
shift
LOG_PATH="$1"
shift

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$PROJECT_ROOT/tools/run_transfer_benchmark_in_env.sh"

mkdir -p "$(dirname "$LOG_PATH")"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists" >&2
  exit 1
fi

quoted_args=()
for arg in "$@"; do
  quoted_args+=("$(printf '%q' "$arg")")
done

cmd="cd $(printf '%q' "$PROJECT_ROOT") && $(printf '%q' "$RUNNER") ${quoted_args[*]} 2>&1 | tee $(printf '%q' "$LOG_PATH")"
tmux new-session -d -s "$SESSION_NAME" "$cmd"

echo "Started tmux session: $SESSION_NAME"
echo "Log file: $LOG_PATH"
