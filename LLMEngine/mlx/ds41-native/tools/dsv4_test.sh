#!/usr/bin/env bash
# ── MUTEX di build: i comandi di build (cc/clang/cmake/make/dotnet) prendono un lock ──
# esclusivo su mutex.txt SOLO per la durata del SINGOLO comando, poi lo rilasciano subito
# (lo script puo proseguire a lungo senza tenere il lock). Scadenza 15 min con heartbeat:
# un processo in hang non blocca le altre AI. Vedi tools/gmb_with_lock.sh e Knowledge/
# Build-Mutex-Lock-AI.md.
if [ -z "${GMB_BUILDBIN_ON:-}" ]; then
  export GMB_BUILDBIN_ON=1
  GMB_BUILDBIN="$(cd "$(dirname "$0")" && pwd)/buildbin"
  [ -d "$GMB_BUILDBIN" ] && export PATH="$GMB_BUILDBIN:$PATH"
fi

# dsv4_test.sh <test.c> [args...] — compila SEMPRE FRESCO il test DS4 contro lib+header CORRENTI.
#
# REGOLA (utente): non deve esistere il concetto di "stale". Ogni compilazione usa librerie fresche
# o RIFIUTA. I test DS4 sono .c compilati ad-hoc che includono header inline: se compilati prima di
# una modifica a un header condiviso (es. LLMCodec/LlmCodecGgufTensorInfo), il LAYOUT embedded nel
# binario diverge da quello della dylib -> mismatch ABI silenzioso ("manca blk..."). Questo wrapper
# elimina la classe di bug: ricompila la lib (incrementale) e il test PRIMA di eseguirlo.
#
# Uso:  tools/dsv4_test.sh llm_dsv4_batch_test.c <gguf> 40 16 f32
#       tools/dsv4_test.sh llm_dsv4_prefill_bench.c <gguf> 6474 4096 56000 /tmp/real.tok 0 0 0
#
# Env:  LLMENV (default /Users/gaetano/Documents/Projects/LLMEnv)
#       B      (default /Users/gaetano/Documents/Projects/FastLLM/build-release)
set -euo pipefail
LLMENV="${LLMENV:-/Users/gaetano/Documents/Projects/LLMEnv}"
B="${B:-/Users/gaetano/Documents/Projects/FastLLM/build-release}"
if [ "$#" -lt 1 ]; then echo "uso: $0 <test.c> [args...]" >&2; exit 2; fi
T="$1"; shift

# 1) lib fresca (incrementale: ricompila se sorgenti/header sono cambiati)
cmake --build "$B" --target llm_ai_mlx -j 3 >/dev/null

# 2) test SEMPRE ricompilato (mai riusare il binario: è la causa dello stale)
OUT="/tmp/dsv4_$(basename "$T" .c)"
clang -O1 -I "$LLMENV/LLMEngine/mlx" -I "$LLMENV/LLMEngine/Options" -I "$LLMENV/LLMCodec/include" \
      "$LLMENV/LLMEngine/mlx/tests/$T" -o "$OUT" \
      -L"$B/llmengine_mlx" -L"$B" -lllm_ai_mlx -lllmcodec_ggml \
      -Wl,-rpath,"$B/llmengine_mlx" -Wl,-rpath,"$B"

# 3) esegui
exec "$OUT" "$@"
