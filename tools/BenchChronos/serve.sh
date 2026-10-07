#!/usr/bin/env bash
# serve.sh — avvia/ferma i server OpenAI-compatibili per BenchChronos.
# Uso:  serve.sh {ds4|fmlx} {start|stop}
# Env (override):  MODEL, DS4_BIN, FMLX_BIN, DS4_PORT, FMLX_PORT
set -euo pipefail

MODEL="${MODEL:-/Users/gaetano/ds4/ds4flash.gguf}"
DS4_BIN="${DS4_BIN:-}"   # server ds4 standalone ritirato (motore unico LLMEngine/mlx/ds4)
FMLX_BIN="${FMLX_BIN:-/Users/gaetano/Documents/Projects/FastLLM/build-release/fastllm}"
DS4_PORT="${DS4_PORT:-8101}"
FMLX_PORT="${FMLX_PORT:-8102}"

engine="${1:-}"; action="${2:-}"
[ -n "$engine" ] && [ -n "$action" ] || { echo "uso: $0 {ds4|fmlx} {start|stop}"; exit 2; }

case "$engine/$action" in
  ds4/start)
    cd "$(dirname "$DS4_BIN")/../.." 2>/dev/null || true
    exec "$DS4_BIN" -m "$MODEL" --metal --ssd-streaming -c 16384 -n 256 --host 127.0.0.1 --port "$DS4_PORT" ;;
  fmlx/start)
    exec "$FMLX_BIN" serve --host 127.0.0.1 --port "$FMLX_PORT" ;;
  ds4/stop)  pkill -f "ds4-server.*--port $DS4_PORT" 2>/dev/null || true ;;
  fmlx/stop) pkill -f "fastllm serve.*--port $FMLX_PORT" 2>/dev/null || true ;;
  *) echo "uso: $0 {ds4|fmlx} {start|stop}"; exit 2 ;;
esac
