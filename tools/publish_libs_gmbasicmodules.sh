#!/usr/bin/env bash
# publish_libs_gmbasicmodules.sh — LLMEnv/FastLLM: pubblica il gruppo `engine` (C) su
# c-vision/GMBasicModules e aggiorna `libs-manifest.json`. Da eseguire SU COMANDO, dopo un build.
#
# PERCHÉ ESISTE. Chi PRODUCE una libreria la pubblica; chi la consuma non tiene copia. Le lib del
# motore AI (`libllm_ai_mlx`, `mlx.metallib`, `libllm_ai_llamacpp`, `libllama*`, `libggml*`,
# `libmtmd*`, `libllmcodec_*`, `libllmtoken`, `libllm_vision_apple`, `libllmengine_moe_stream`)
# le produce LLMEnv/FastLLM, non GMBasic: quindi è QUESTO script che le mette su GMBasicModules.
#
# COSA FA (in ordine, fermandosi al primo errore)
#   1. (opzionale) build incrementale delle lib AI    → `tools/build_ai_libs.sh`
#   2. STAGING: raccoglie gli artefatti dai build dir di LLMEnv (+ il dist di FastLLM se presente)
#      in un'unica cartella `build/dist/<rid>/` (niente duplicati, niente file di altri gruppi).
#      La SUDDIVISIONE in gruppi la fa il pubblicatore: `engine` = plumbing (codec/token) e
#      `llmmotor` = motori (MLX/llama/ggml/mtmd + wrapper): dipendenze diverse, pesi diversi.
#   3. VERIFICA: architettura giusta per il RID, motore MLX presente (`libllm_ai_mlx` + `mlx.metallib`),
#      sha256 di ogni file. NOTA: `libllm_ai_llamacpp` incorpora llama.cpp STATICAMENTE → le shared
#      libllama/libggml/libmtmd NON servono (nessun componente carica libgmb_ai_llamacpp).
#   4. PUBBLICA: chiama `GMBasic/tools/libs_publish_gh.sh engine <rid> --source build/dist/<rid>`
#      e, con `--push-manifest`, rigenera e pubblica `libs-manifest.json`
#
# USO
#   tools/publish_libs_gmbasicmodules.sh [<rid>] [opzioni]
#     --dist DIR           cartella di staging (default: build/dist/<rid>)
#     --fastllm DIR|auto|none   dist di FastLLM da unire (default: auto = ../FastLLM/dist/rel-*)
#     --skip-build         non lancia build_ai_libs.sh (usa ciò che è già compilato)
#     --allow-incomplete   pubblica anche senza il motore MLX (sconsigliato)
#     --dry-run            fa staging + verifica e stampa il piano, NON pubblica
#     --index              dopo l'upload rigenera il manifest locale
#     --push-manifest      dopo l'upload rigenera e PUBBLICA il manifest su main
#     --manifest FILE      manifest (default: <GMBasic>/Resources/libs-manifest.json)
#     --gmbasic DIR        checkout di GMBasic (default: ../GMBasicMonoRepo/GMBasic o ../GMBasic)
#     --repo OWNER/REPO    default: c-vision/GMBasicModules
#     --yes                non chiede conferma
#
# ESEMPI
#   tools/publish_libs_gmbasicmodules.sh                          # osx-arm64, build+stage+dry-run
#   tools/publish_libs_gmbasicmodules.sh --push-manifest           # pubblica e aggiorna l'indice
#   tools/publish_libs_gmbasicmodules.sh osx-x64 --skip-build --dry-run
set -euo pipefail

RID=""
DIST=""; FASTLLM="auto"; SKIP_BUILD=0; ALLOW_INCOMPLETE=0; DRY=0
INDEX_FLAG=""; PUSH_FLAG=""; MANIFEST=""; GMBASIC=""; REPO="c-vision/GMBasicModules"; YES=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dist) DIST="${2:-}"; shift 2;;
    --fastllm) FASTLLM="${2:-}"; shift 2;;
    --skip-build) SKIP_BUILD=1; shift;;
    --allow-incomplete) ALLOW_INCOMPLETE=1; shift;;
    --dry-run) DRY=1; shift;;
    --index) INDEX_FLAG="--index"; shift;;
    --push-manifest) PUSH_FLAG="--push-manifest"; shift;;
    --manifest) MANIFEST="${2:-}"; shift 2;;
    --gmbasic) GMBASIC="${2:-}"; shift 2;;
    --repo) REPO="${2:-}"; shift 2;;
    --yes) YES=1; shift;;
    -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    -*) echo "opzione sconosciuta: $1" >&2; exit 2;;
    *) RID="$1"; shift;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVROOT="$(cd "$HERE/.." && pwd)"

default_rid() {
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) echo "osx-arm64";; Darwin-x86_64) echo "osx-x64";;
    Linux-x86_64) echo "linux-x64";;  Linux-aarch64|Linux-arm64) echo "linux-arm64";;
    *) echo "osx-arm64";;
  esac
}
[ -n "$RID" ] || RID="$(default_rid)"

# ── checkout di GMBasic (contiene il pubblicatore generico + il manifest) ────────────────────────
if [ -z "$GMBASIC" ]; then
  for c in "$ENVROOT/../GMBasicMonoRepo/GMBasic" "$ENVROOT/../GMBasic" "$ENVROOT/../../GMBasic"; do
    [ -f "$c/tools/libs_publish_gh.sh" ] && { GMBASIC="$(cd "$c" && pwd)"; break; }
  done
fi
[ -n "$GMBASIC" ] || { echo "checkout GMBasic non trovato: passa --gmbasic <dir>" >&2; exit 1; }
PUBLISH="$GMBASIC/tools/libs_publish_gh.sh"
[ -f "$PUBLISH" ] || { echo "pubblicatore assente: $PUBLISH" >&2; exit 1; }
[ -n "$MANIFEST" ] || MANIFEST="$GMBASIC/Resources/libs-manifest.json"

[ -n "$DIST" ] || DIST="$ENVROOT/build/dist/$RID"

# ── 1. build (opzionale) ─────────────────────────────────────────────────────────────────────────
if [ "$SKIP_BUILD" = "0" ]; then
  if [ -x "$ENVROOT/tools/build_ai_libs.sh" ]; then
    echo "[1/4] build lib AI (tools/build_ai_libs.sh)"
    ( cd "$ENVROOT" && bash tools/build_ai_libs.sh ) || { echo "build fallita: fermo" >&2; exit 1; }
  else
    echo "[1/4] build saltata: tools/build_ai_libs.sh non eseguibile (usa --skip-build per silenziare)"
  fi
else
  echo "[1/4] build saltata (--skip-build)"
fi

# ── 2. staging ───────────────────────────────────────────────────────────────────────────────────
echo "[2/4] staging → $DIST"
mkdir -p "$DIST"
staged=0
add() {  # add <file>
  local src="$1" base; base="$(basename "$src")"
  [ -f "$src" ] || return 0
  case "$base" in
    libtmp_*) return 0;;                       # artefatti temporanei di build
    *.dylib|*.so|*.dll|*.metallib) ;;
    *) return 0;;
  esac
  case "$base" in
    libllm*|libgmb_ai_mlx*|libgmb_ai_llamacpp*|libgmb_ai_moestream*|libgmb_codec_*|mlx.metallib|libllama*|libggml*|libmtmd*) ;;
    *) return 0;;
  esac
  [ -f "$DIST/$base" ] || staged=$((staged+1))   # conta solo i NUOVI
  cp -f "$src" "$DIST/$base"
}

# artefatti del build CMake di LLMEnv
for f in "$ENVROOT"/build/*.dylib "$ENVROOT"/build/LLMToken/*.dylib \
         "$ENVROOT"/build/LLMEngine/*/*.dylib "$ENVROOT"/build/LLMEngine/*/*.metallib; do
  add "$f"
done
# (fallback) vecchie cartelle di build dentro il sorgente
for f in "$ENVROOT"/LLMEngine/*/build*/*.dylib "$ENVROOT"/LLMEngine/*/build*/*.metallib; do
  add "$f"
done

# FastLLM dist (motore Apache "inference-complete": codec + vision + moe_stream)
if [ "$FASTLLM" = "auto" ]; then
  FASTLLM="$(ls -d "$ENVROOT"/../FastLLM/dist/rel-* 2>/dev/null | sort | tail -1 || true)"
fi
if [ -n "$FASTLLM" ] && [ "$FASTLLM" != "none" ] && [ -d "$FASTLLM" ]; then
  for f in "$FASTLLM"/*.dylib "$FASTLLM"/*.metallib; do add "$f"; done
  echo "     + FastLLM dist: $FASTLLM"
else
  echo "     (nessun dist FastLLM)"
fi

# Cosa DEVE esserci: i motori veri. `libllm_ai_llamacpp` incorpora llama.cpp STATICAMENTE
# (237 simboli llama_* definiti, 0 non risolti) e nessuno carica `libgmb_ai_llamacpp`: quindi
# le shared `libllama/libggml/libmtmd` NON sono richieste (erano orfane, con rpath assoluti).
missing_motor=0
for want in libllm_ai_mlx.dylib mlx.metallib; do
  [ -f "$DIST/$want" ] || missing_motor=$((missing_motor+1))
done

echo "     file in staging: $(ls -1 "$DIST" | wc -l | tr -d ' ') (copiati ora: $staged)"

# ── 3. verifica ─────────────────────────────────────────────────────────────────────────────────
echo "[3/4] verifica"
if [ "$missing_motor" != "0" ]; then
  if [ "$ALLOW_INCOMPLETE" = "1" ]; then
    echo "     ATTENZIONE: mancano $missing_motor artefatti del motore MLX (libllm_ai_mlx/mlx.metallib) — pubblico INCOMPLETO" >&2
  else
    cat >&2 <<EOF
ERRORE: mancano $missing_motor artefatti del motore in $DIST (libllm_ai_mlx.dylib + mlx.metallib).
  Senza, il gruppo 'llmmotor' non fa partire l'inferenza locale (e nessuno se ne accorgerebbe:
  gli sha dei file presenti sarebbero corretti). Costruisci prima (tools/build_ai_libs.sh) o usa
  --allow-incomplete per pubblicare comunque.
EOF
    exit 1
  fi
fi
# llama.cpp NON è una dipendenza separata: `libllm_ai_llamacpp` è autosufficiente (statico).
[ -f "$DIST/libllm_ai_llamacpp.dylib" ] || echo "     (nota: libllm_ai_llamacpp assente — engine llama.cpp non incluso)" >&2
for f in "$DIST"/*.dylib; do
  [ -f "$f" ] || continue
  if ! file "$f" | grep -q "$(case "$RID" in *arm64*) echo arm64;; *x64*) echo x86_64;; *) echo .;; esac)"; then
    echo "     ATTENZIONE: $(basename "$f") non sembra $RID: $(file -b "$f" | cut -c1-60)" >&2
  fi
done
du -sh "$DIST" | awk '{print "     dimensione staging: "$1}'

if [ "$DRY" = "1" ]; then
  echo "[4/4] dry-run: nessuna pubblicazione. Piano:"
  bash "$PUBLISH" engine "$RID" --source "$DIST" --repo "$REPO" --dry-run
  exit 0
fi

# ── 4. pubblica (+ indice) ───────────────────────────────────────────────────────────────────────
# DUE gruppi, per dipendenza (non per proprietario):
#   engine   = PLUMBING che ogni dylib nostra linka (libllmcodec_*, libllmtoken) → ~0.5 MB, sempre installato
#   llmmotor = motori pesanti (MLX + metallib, llama.cpp/ggml/mtmd + wrapper libgmb_ai_*) → ~64 MB,
#              serve solo per l'inferenza locale
echo "[4/4] pubblicazione su $REPO (gruppi: engine, llmmotor)"
for g in engine llmmotor; do
  args=("$g" "$RID" --source "$DIST" --repo "$REPO" --manifest "$MANIFEST")
  [ "$YES" = "1" ] && args+=(--yes)
  bash "$PUBLISH" "${args[@]}"
done
if [ -n "$INDEX_FLAG" ]; then
  bash "$GMBASIC/tools/libs_manifest_gh.sh" build --rid "$RID" --repo "$REPO" --manifest "$MANIFEST"
  [ -n "$PUSH_FLAG" ] && bash "$GMBASIC/tools/libs_manifest_gh.sh" push --manifest "$MANIFEST" --repo "$REPO"
fi
