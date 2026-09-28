#!/usr/bin/env bash
# =============================================================================
# build_ai_libs.sh — costruisce le librerie runtime di LLMEnv e le staziona in
# <build-dir>/dist/<rid>/ come artefatti consumabili da un host esterno
# (dlopen / P-Invoke).
#
# PERCHÉ QUI. LLMEnv è la SORGENTE UNICA del mondo AI: questo script possiede la
# build dei motori (MLX, llama.cpp, MoE/streaming) + i plugin del codec + Apple
# Vision. Gli host (es. GMBasic) NON ricompilano nulla dei motori: consumano solo
# gli artefatti stazionati qui (poi li copiano nella propria Resources/libs/<rid>/).
#
# USA IL SUPERBUILD del repo (`CMakeLists.txt` con -DLLMENV_BUILD_ENGINE=ON):
# aggancia i layer in ordine e i target engine (plugin, moe, llama, mlx).
#
# USO:
#   tools/build_ai_libs.sh [--rid <rid>] [--build-dir <dir>] [--clean]
#
# ENV:
#   --build-dir  (default: <repo>/build)   # dist = <build-dir>/dist/<rid>  (gitignored)
#
# OFFLINE: il superbuild aggancia mlx-c/llama.cpp via FetchContent. mlx-c usa la
# cache locale <repo>/libs/mlx-deps se presente; per llama.cpp offline passare la
# cache sorgente:
#   FETCHCONTENT_SOURCE_DIR_LLAMA_CPP=<llama_cpp-src> tools/build_ai_libs.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

detect_rid() {
    local os arch
    case "$(uname -s)" in
        Darwin) os="osx" ;;
        Linux)  os="linux" ;;
        MINGW*|MSYS*|CYGWIN*) os="win" ;;
        *) echo "unsupported-$(uname -s)"; exit 1 ;;
    esac
    case "$(uname -m)" in
        arm64|aarch64) arch="arm64" ;;
        x86_64|amd64)  arch="x64" ;;
        *) echo "unsupported-$(uname -m)"; exit 1 ;;
    esac
    if [ "$os" = "win" ]; then echo "win-$arch"; else echo "$os-$arch"; fi
}

RID=""; BUILD_DIR=""; CLEAN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --rid) RID="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --clean) CLEAN=1; shift ;;
        *) echo "?opzione sconosciuta: $1"; exit 1 ;;
    esac
done
[ -z "$RID" ] && RID="$(detect_rid)"
[ -z "$BUILD_DIR" ] && BUILD_DIR="$ROOT/build"
DIST="$BUILD_DIR/dist/$RID"

ext() { case "$RID" in win-*) echo "dll";; osx-*) echo "dylib";; *) echo "so";; esac; }
EXT="$(ext)"

if [ "$CLEAN" = "1" ]; then
    rm -rf "$DIST"
    echo "== clean: $DIST"
    exit 0
fi

echo "== LLMEnv build_ai_libs: rid=$RID  build=$BUILD_DIR  dist=$DIST"

# Superbuild: layer base + engine (plugin, moe, llama, mlx). MLX è Apple-Silicon-only;
# su altre piattaforme il target mlx viene comunque configurato ma è l'unico che
# richiede Metal. Build dell'intero grafo.
cmake -S "$ROOT" -B "$BUILD_DIR" -DLLMENV_BUILD_ENGINE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j4

mkdir -p "$DIST"
found=0
while IFS= read -r f; do
    cp -f "$f" "$DIST/" && { echo "  stage: $(basename "$f")"; found=1; }
done < <(find "$BUILD_DIR" -type f \( \
    -name "libllm_ai_mlx.$EXT" -o -name "libllm_ai_llamacpp.$EXT" -o \
    -name "libllm_vision_apple.$EXT" -o -name "libllmengine_moe_stream.$EXT" -o \
    -name "libllmcodec_*.$EXT" -o \
    -name "libllama*.$EXT" -o -name "libggml*.$EXT" -o -name "libmtmd*.$EXT" \) | sort)

if [ "$found" = "0" ]; then
    echo "WARN: nessuna dylib trovata in $BUILD_DIR" >&2
    exit 1
fi
echo "== LLMEnv: $(ls "$DIST" | wc -l | tr -d ' ') librerie stazionate in $DIST"
