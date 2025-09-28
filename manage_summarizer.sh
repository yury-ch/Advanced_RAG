#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_SCRIPT="$SCRIPT_DIR/YouTube_Summarizer.py"
PID_FILE="$SCRIPT_DIR/.youtube_summarizer.pid"
LOG_FILE="$SCRIPT_DIR/youtube_summarizer.log"
PYTHON_BIN="${PYTHON:-}"

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
        PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

usage() {
    cat <<'USAGE'
Usage: ./manage_summarizer.sh <command>

Commands:
  start   Launch the YouTube summarizer Gradio app in the background
  stop    Stop the running Gradio app
  status  Display whether the app is running
  logs    Tail the application log file
USAGE
}

ensure_app_exists() {
    if [[ ! -f "$APP_SCRIPT" ]]; then
        echo "Error: Could not find $APP_SCRIPT" >&2
        exit 1
    fi
}

is_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid="$(cat "$PID_FILE")"
        if kill -0 "$pid" >/dev/null 2>&1; then
            echo "$pid"
            return 0
        fi
    fi
    return 1
}

start_app() {
    if pid="$(is_running)"; then
        echo "App already running with PID $pid"
        exit 0
    fi

    ensure_app_exists

    echo "Starting YouTube summarizer..."
    nohup "$PYTHON_BIN" "$APP_SCRIPT" >"$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" >"$PID_FILE"
    echo "Started with PID $pid (logs: $LOG_FILE)"
}

stop_app() {
    if ! pid="$(is_running)"; then
        echo "App is not running"
        exit 0
    fi

    echo "Stopping PID $pid..."
    kill "$pid" || true

    for _ in {1..10}; do
        if ps -p "$pid" >/dev/null 2>&1; then
            sleep 0.5
        else
            break
        fi
    done

    if ps -p "$pid" >/dev/null 2>&1; then
        echo "Process did not terminate; sending SIGKILL"
        kill -9 "$pid" || true
    fi

    rm -f "$PID_FILE"
    echo "Stopped"
}

status_app() {
    if pid="$(is_running)"; then
        echo "Running (PID $pid)"
    else
        echo "Not running"
    fi
}

tail_logs() {
    if [[ ! -f "$LOG_FILE" ]]; then
        echo "No logs available at $LOG_FILE"
        exit 0
    fi
    tail -f "$LOG_FILE"
}

main() {
    local command=${1:-}

    case "$command" in
        start)
            start_app
            ;;
        stop)
            stop_app
            ;;
        status)
            status_app
            ;;
        logs)
            tail_logs
            ;;
        -h|--help|help|"")
            usage
            ;;
        *)
            echo "Unknown command: $command" >&2
            usage
            exit 1
            ;;
    esac
}

main "$@"
