#!/usr/bin/env bash
# =============================================================================
# build_ai_libs.sh — costruisce le librerie runtime di LLMEnv e le staziona in
# <build-dir>/dist/<rid>/ come artefatti consumabili da un host esterno
# (dlopen / P-Invoke).
#
# PERCHÉ QUI. LLMEnv è la SORGENTE UNICA del mondo AI: questo script possiede la
# build dei motori (MLX, llama.cpp, MoE/streaming) + i plugin del codec + Apple
# Vision. Gli host NON ricompilano nulla dei motori: consumano solo
# gli artefatti stazionati qui (poi li copiano nella propria Resources/libs/<rid>/).
#
# USA IL SUPERBUILD del repo (`CMakeLists.txt` con -DLLMENV_BUILD_ENGINE=ON):
# aggancia i layer in ordine e i target engine (plugin, moe, llama, mlx).
#
# USO:
#   tools/build_ai_libs.sh [--rid <rid>] [--build-dir <dir>] [--deployment-target <macOS>] [--clean]
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

RID=""; BUILD_DIR=""; CLEAN=0; DEPLOY_TARGET=""
while [ $# -gt 0 ]; do
    case "$1" in
        --rid) RID="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --deployment-target) DEPLOY_TARGET="$2"; shift 2 ;;   # versione minima di macOS (es. 14.0, 26.2)
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

# Fix Metal-4.1 NAX (idempotente). I kernel NAX (tensor unit M5) di MLX 0.32.x NON compilano
# col compilatore Metal 4.1 (`32023.921`): errore address-space in `steel/{attn,gemm}/nax.h`.
# Il fix upstream (mlx #3963) è in 0.32.1+, ma 0.32.1+ cambia l'API `compile` (rompe mlx-c
# 0.6.0). Si applica il SOLO fix Metal alla cache mlx-src, mantenendo l'API 0.32.0.
MLXSRC="$ROOT/libs/mlx-deps/mlx-src"
NAXPATCH="$ROOT/LLMEngine/mlx/patches/mlx_metal41_nax.patch"
if [ -d "$MLXSRC/mlx/backend/metal/kernels" ] && [ -f "$NAXPATCH" ]; then
    if patch -p1 --forward --dry-run --silent -d "$MLXSRC" < "$NAXPATCH" >/dev/null 2>&1; then
        patch -p1 --forward -d "$MLXSRC" < "$NAXPATCH" >/dev/null && echo "== applicato fix Metal-4.1 NAX a mlx-src"
    else
        echo "== fix Metal-4.1 NAX gia' applicato (o non applicabile)"
    fi
fi

# Superbuild: layer base + engine (plugin, moe, llama, mlx). MLX è Apple-Silicon-only;
# su altre piattaforme il target mlx viene comunque configurato ma è l'unico che
# richiede Metal. Build dell'intero grafo.
# Il deployment target decide la variante: sotto 26.2 MLX NON costruisce i kernel NAX (tensor unit M5) e la libreria
# si carica anche su macOS 14+; da 26.2 in su i kernel NAX ci sono e la libreria richiede macOS 26.2.
TARGET_ARGS=()
if [ -n "$DEPLOY_TARGET" ]; then export MACOSX_DEPLOYMENT_TARGET="$DEPLOY_TARGET"; TARGET_ARGS=(-DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOY_TARGET"); fi
cmake -S "$ROOT" -B "$BUILD_DIR" -DLLMENV_BUILD_ENGINE=ON -DCMAKE_BUILD_TYPE=Release ${TARGET_ARGS[@]+"${TARGET_ARGS[@]}"}
cmake --build "$BUILD_DIR" -j4
# MLX Metal: la `mlx.metallib` (kernels Metal AOT) è un artefatto runtime necessario
# all'engine MLX (MLX la cerca colocated accanto alla libreria, o al METAL_PATH del build).
# Costruiscila esplicitamente se il target esiste (non è nel grafo di default).
# NB: il target puo' NON essere nel grafo di default; lo si costruisce SEMPRE (guardia sulla
# presenza del target, non su un match fragile di `--target help`). Il fallimento e' LOUD:
# una metallib assente = engine MLX inutilizzabile, e in passato e' passata in silenzio.
# (con pipefail `cmake | grep -q` fallisce per SIGPIPE quando grep esce al primo match: si legge prima l'elenco in una variabile)
TARGETS="$(cmake --build "$BUILD_DIR" --target help 2>/dev/null || true)"
if grep -qi 'mlx-metallib' <<<"$TARGETS"; then
    cmake --build "$BUILD_DIR" --target mlx-metallib -j4 || true
fi
if ! find "$BUILD_DIR" -name 'mlx.metallib' -print -quit | grep -q .; then
    echo "ERRORE: mlx.metallib NON prodotta (target mlx-metallib fallito) — l'engine MLX non carichera'. Vedi il log sopra." >&2
    exit 1
fi

mkdir -p "$DIST"
found=0
while IFS= read -r f; do
    cp -f "$f" "$DIST/" && { echo "  stage: $(basename "$f")"; found=1; }
done < <(find "$BUILD_DIR" -type f \( \
    -name "libllm*.$EXT" -o -name "mlx.metallib" -o \
    -name "libllama*.$EXT" -o -name "libggml*.$EXT" -o -name "libmtmd*.$EXT" \) | sort)

if [ "$found" = "0" ]; then
    echo "WARN: nessuna dylib trovata in $BUILD_DIR" >&2
    exit 1
fi
echo "== LLMEnv: $(ls "$DIST" | wc -l | tr -d ' ') librerie stazionate in $DIST"
