#!/usr/bin/env bash
# build_bench.sh — compila l'harness dei kernel Metal propri (gemb).
# Output in LLMEngine/metal/build/ (artefatto, non versionato): eseguibile + .metal accanto.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="$DIR/build"
mkdir -p "$BUILD"
cp "$DIR/gemb_kernel.metal" "$BUILD/gemb_kernel.metal"
clang -O2 -fobjc-arc -framework Foundation -framework Metal -o "$BUILD/gemb_bench" "$DIR/gemb_bench.m"
echo "built: $BUILD/gemb_bench"
