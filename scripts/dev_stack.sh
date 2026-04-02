#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"
LOG_DIR="$RUN_DIR/logs"
PID_DIR="$RUN_DIR/pids"

LLM_PID_FILE="$PID_DIR/llmlingua_mcp.pid"
DOCKER_PID_FILE="$PID_DIR/docker_mcp.pid"
APP_PID_FILE="$PID_DIR/main_app.pid"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

LLM_HOST="${LLMLINGUA_MCP_HOST:-127.0.0.1}"
LLM_PORT="${LLMLINGUA_MCP_PORT:-8765}"
LLM_PATH="${LLMLINGUA_MCP_PATH:-/mcp}"
LLM_URL="${LLMLINGUA_MCP_SERVER_URL:-http://$LLM_HOST:$LLM_PORT$LLM_PATH}"

DOCKER_HOST="${DOCKER_MCP_HOST:-127.0.0.1}"
DOCKER_PORT="${DOCKER_MCP_PORT:-8766}"
DOCKER_PATH="${DOCKER_MCP_PATH:-/mcp}"
DOCKER_URL="http://$DOCKER_HOST:$DOCKER_PORT$DOCKER_PATH"

LLM_MODEL_NAME="${LLMLINGUA_MODEL_NAME:-Qwen/Qwen2-1.5B-Instruct}"
LLM_MODEL_PATH="${LLMLINGUA_MODEL_PATH:-./models/qwen-1.5b}"

APP_CMD_DEFAULT="$PYTHON_BIN -m app.ui.gui_chat"

mkdir -p "$LOG_DIR" "$PID_DIR"

usage() {
  cat <<'EOF'
Usage:
  scripts/dev_stack.sh up [--detach-app] [--no-docker-mcp] [--app-cmd "$PYTHON_BIN -m app.ui.gui_chat"]
  scripts/dev_stack.sh llm-up | llm-down
  scripts/dev_stack.sh docker-up | docker-down
  scripts/dev_stack.sh app-up [--app-cmd "$PYTHON_BIN -m app.ui.gui_chat"]
  scripts/dev_stack.sh app-down
  scripts/dev_stack.sh down
  scripts/dev_stack.sh force-down [--yes]
  scripts/dev_stack.sh restart [--detach-app] [--no-docker-mcp] [--app-cmd "..."]
  scripts/dev_stack.sh status
  scripts/dev_stack.sh logs [llm|docker|app|all]

Environment overrides:
  PYTHON_BIN
  LLMLINGUA_MCP_HOST LLMLINGUA_MCP_PORT LLMLINGUA_MCP_PATH LLMLINGUA_MCP_SERVER_URL
  LLMLINGUA_MODEL_NAME LLMLINGUA_MODEL_PATH
  DOCKER_MCP_HOST DOCKER_MCP_PORT DOCKER_MCP_PATH
EOF
}

start_llm_service() {
  if is_port_listening "$LLM_HOST" "$LLM_PORT"; then
    echo "[llmlingua_mcp] reusing existing external service at $LLM_URL"
    return 0
  fi

  start_process \
    "llmlingua_mcp" \
    "$LLM_PID_FILE" \
    "$LOG_DIR/llmlingua_mcp.log" \
    env \
    LLMLINGUA_MODEL_NAME="$LLM_MODEL_NAME" \
    LLMLINGUA_MODEL_PATH="$LLM_MODEL_PATH" \
    "$PYTHON_BIN" -m app.mcp.llmlingua_compression_server \
      --transport streamable-http --host "$LLM_HOST" --port "$LLM_PORT" --path "$LLM_PATH"
}

stop_llm_service() {
  stop_process "llmlingua_mcp" "$LLM_PID_FILE"
}

start_docker_service() {
  if is_port_listening "$DOCKER_HOST" "$DOCKER_PORT"; then
    echo "[docker_mcp] reusing existing external service at $DOCKER_URL"
    return 0
  fi

  start_process \
    "docker_mcp" \
    "$DOCKER_PID_FILE" \
    "$LOG_DIR/docker_mcp.log" \
    "$PYTHON_BIN" -m app.mcp.docker_sandbox_server \
      --transport streamable-http --host "$DOCKER_HOST" --port "$DOCKER_PORT" --path "$DOCKER_PATH"
}

stop_docker_service() {
  stop_process "docker_mcp" "$DOCKER_PID_FILE"
}

start_main_app_detached() {
  local app_cmd="$1"
  start_process \
    "main_app" \
    "$APP_PID_FILE" \
    "$LOG_DIR/main_app.log" \
    env \
    LLMLINGUA_MCP_ENABLED=true \
    LLMLINGUA_MCP_SERVER_URL="$LLM_URL" \
    bash -lc "cd '$ROOT_DIR' && $app_cmd"
}

stop_main_app() {
  stop_process "main_app" "$APP_PID_FILE"
}

kill_pid_hard() {
  local pid="$1"
  if ! is_pid_running "$pid"; then
    return 0
  fi

  kill "$pid" >/dev/null 2>&1 || true
  for _ in {1..10}; do
    if ! is_pid_running "$pid"; then
      return 0
    fi
    sleep 0.2
  done

  kill -9 "$pid" >/dev/null 2>&1 || true
}

kill_listeners_on_port() {
  local port="$1"
  local pids
  pids="$( (lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true) | tr '\n' ' ' )"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  echo "[force-down] killing listeners on port $port: $pids"
  for pid in $pids; do
    kill_pid_hard "$pid"
  done
}

kill_by_pattern() {
  local pattern="$1"
  local pids
  pids="$( (pgrep -f "$pattern" 2>/dev/null || true) | tr '\n' ' ' )"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  echo "[force-down] killing pattern '$pattern': $pids"
  for pid in $pids; do
    kill_pid_hard "$pid"
  done
}

force_down_all() {
  echo "[force-down] stopping managed processes first"
  stop_main_app || true
  stop_docker_service || true
  stop_llm_service || true

  echo "[force-down] stopping external listeners"
  kill_listeners_on_port "$LLM_PORT"
  kill_listeners_on_port "$DOCKER_PORT"

  echo "[force-down] stopping known app/module leftovers"
  kill_by_pattern "app.mcp.llmlingua_compression_server"
  kill_by_pattern "app.mcp.docker_sandbox_server"
  kill_by_pattern "app.ui.gui_chat"
  kill_by_pattern "app.cli.main"

  rm -f "$LLM_PID_FILE" "$DOCKER_PID_FILE" "$APP_PID_FILE"
  echo "[force-down] done"
}

confirm_force_down() {
  local bypass="${1:-}"
  if [[ "$bypass" == "--yes" ]]; then
    return 0
  fi

  if [[ ! -t 0 ]]; then
    echo "[force-down] non-interactive shell detected, use --yes to confirm"
    return 1
  fi

  local answer
  read -r -p "[force-down] This will kill related MCP/app processes. Continue? [y/N] " answer
  case "${answer:-}" in
    y|Y|yes|YES)
      return 0
      ;;
    *)
      echo "[force-down] cancelled"
      return 1
      ;;
  esac
}

is_pid_running() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 1
  fi
  kill -0 "$pid" >/dev/null 2>&1
}

is_port_listening() {
  local host="$1"
  local port="$2"
  if command -v nc >/dev/null 2>&1; then
    nc -z "$host" "$port" >/dev/null 2>&1 && return 0
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 && return 0
  fi
  return 1
}

read_pid() {
  local f="$1"
  if [[ -f "$f" ]]; then
    cat "$f"
  fi
}

start_process() {
  local name="$1"
  local pid_file="$2"
  local log_file="$3"
  shift 3

  local existing_pid
  existing_pid="$(read_pid "$pid_file")"
  if is_pid_running "$existing_pid"; then
    echo "[$name] already running pid=$existing_pid"
    return 0
  fi

  echo "[$name] starting..."
  nohup "$@" >"$log_file" 2>&1 &
  local new_pid=$!
  echo "$new_pid" >"$pid_file"
  sleep 1

  if is_pid_running "$new_pid"; then
    echo "[$name] started pid=$new_pid log=$log_file"
  else
    echo "[$name] failed to start, check log: $log_file"
    return 1
  fi
}

stop_process() {
  local name="$1"
  local pid_file="$2"

  local pid
  pid="$(read_pid "$pid_file")"
  if ! is_pid_running "$pid"; then
    echo "[$name] not running"
    rm -f "$pid_file"
    return 0
  fi

  echo "[$name] stopping pid=$pid"
  kill "$pid" >/dev/null 2>&1 || true

  for _ in {1..15}; do
    if ! is_pid_running "$pid"; then
      break
    fi
    sleep 0.2
  done

  if is_pid_running "$pid"; then
    echo "[$name] force kill pid=$pid"
    kill -9 "$pid" >/dev/null 2>&1 || true
  fi

  rm -f "$pid_file"
  echo "[$name] stopped"
}

show_status() {
  local llm_pid docker_pid app_pid
  llm_pid="$(read_pid "$LLM_PID_FILE")"
  docker_pid="$(read_pid "$DOCKER_PID_FILE")"
  app_pid="$(read_pid "$APP_PID_FILE")"

  if is_pid_running "$llm_pid"; then
    echo "llmlingua_mcp: running pid=$llm_pid url=$LLM_URL"
  elif is_port_listening "$LLM_HOST" "$LLM_PORT"; then
    echo "llmlingua_mcp: external-running host=$LLM_HOST port=$LLM_PORT url=$LLM_URL"
  else
    echo "llmlingua_mcp: stopped"
  fi

  if is_pid_running "$docker_pid"; then
    echo "docker_mcp: running pid=$docker_pid url=$DOCKER_URL"
  elif is_port_listening "$DOCKER_HOST" "$DOCKER_PORT"; then
    echo "docker_mcp: external-running host=$DOCKER_HOST port=$DOCKER_PORT url=$DOCKER_URL"
  else
    echo "docker_mcp: stopped"
  fi

  if is_pid_running "$app_pid"; then
    echo "main_app: running pid=$app_pid"
  else
    echo "main_app: stopped"
  fi
}

show_logs() {
  local target="${1:-all}"
  case "$target" in
    llm)
      tail -n 120 "$LOG_DIR/llmlingua_mcp.log"
      ;;
    docker)
      tail -n 120 "$LOG_DIR/docker_mcp.log"
      ;;
    app)
      tail -n 120 "$LOG_DIR/main_app.log"
      ;;
    all)
      echo "----- llmlingua_mcp.log -----"
      tail -n 60 "$LOG_DIR/llmlingua_mcp.log" 2>/dev/null || true
      echo "----- docker_mcp.log -----"
      tail -n 60 "$LOG_DIR/docker_mcp.log" 2>/dev/null || true
      echo "----- main_app.log -----"
      tail -n 60 "$LOG_DIR/main_app.log" 2>/dev/null || true
      ;;
    *)
      echo "Unknown log target: $target"
      usage
      return 1
      ;;
  esac
}

start_stack() {
  local detach_app="false"
  local start_docker_mcp="true"
  local app_cmd="$APP_CMD_DEFAULT"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --detach-app)
        detach_app="true"
        shift
        ;;
      --no-docker-mcp)
        start_docker_mcp="false"
        shift
        ;;
      --app-cmd)
        app_cmd="$2"
        shift 2
        ;;
      *)
        echo "Unknown option for up: $1"
        usage
        return 1
        ;;
    esac
  done

  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN"
    echo "Set PYTHON_BIN or create venv at .venv"
    return 1
  fi

  start_llm_service

  if [[ "$start_docker_mcp" == "true" ]]; then
    start_docker_service
  fi

  echo "Export these in your shell if needed:"
  echo "  export LLMLINGUA_MCP_ENABLED=true"
  echo "  export LLMLINGUA_MCP_SERVER_URL=$LLM_URL"

  if [[ "$detach_app" == "true" ]]; then
    start_main_app_detached "$app_cmd"
    show_status
    return 0
  fi

  echo "Starting main app in foreground: $app_cmd"
  echo "Press Ctrl+C to stop main app. MCP servers keep running."
  env \
    LLMLINGUA_MCP_ENABLED=true \
    LLMLINGUA_MCP_SERVER_URL="$LLM_URL" \
    bash -lc "cd '$ROOT_DIR' && $app_cmd"
}

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
  usage
  exit 1
fi
shift || true

case "$cmd" in
  up)
    start_stack "$@"
    ;;
  down)
    stop_main_app
    stop_docker_service
    stop_llm_service
    ;;
  force-down)
    confirm_force_down "${1:-}" || exit 1
    force_down_all
    ;;
  restart)
    stop_main_app
    stop_docker_service
    stop_llm_service
    start_stack "$@"
    ;;
  llm-up)
    start_llm_service
    ;;
  llm-down)
    stop_llm_service
    ;;
  docker-up)
    start_docker_service
    ;;
  docker-down)
    stop_docker_service
    ;;
  app-up)
    app_cmd="$APP_CMD_DEFAULT"
    if [[ "${1:-}" == "--app-cmd" ]]; then
      app_cmd="${2:-$APP_CMD_DEFAULT}"
    fi
    start_main_app_detached "$app_cmd"
    ;;
  app-down)
    stop_main_app
    ;;
  status)
    show_status
    ;;
  logs)
    show_logs "${1:-all}"
    ;;
  *)
    echo "Unknown command: $cmd"
    usage
    exit 1
    ;;
esac
