#!/bin/sh
# build_metallibs.sh — compila le metallib di oQ A8 (NAX) coi flag OBBLIGATORI.
# Uso: ./build_metallibs.sh <mlx-src-dir> <out-dir>
# I flag -std=metal4.0 (definisce __HAVE_TENSOR__) e -mmacosx-version-min=26.2 (deployment floor
# MPP/NAX) sono INDISPENSABILI: senza il floor il kernel si carica ma usa una convenzione NAX
# invalida (risultati sbagliati in silenzio). Vedi Knowledge/oQ-A8-INT8-Activation-GEMM-M5-NAX.md.
set -e
MLX="${1:?uso: $0 <mlx-src-dir> <out-dir>}"
OUT="${2:?uso: $0 <mlx-src-dir> <out-dir>}"
CS="$(cd "$(dirname "$0")" && pwd)/csrc"
mkdir -p "$OUT"
for k in oq_a8_qmm_nax oq_a8_quantize oq_a8_support; do
  xcrun -sdk macosx metal -I "$MLX" -I "$CS" -std=metal4.0 -fno-fast-math -mmacosx-version-min=26.2 \
        -c "$CS/$k.metal" -o "$OUT/$k.air"
  xcrun -sdk macosx metal -mmacosx-version-min=26.2 "$OUT/$k.air" -o "$OUT/$k.metallib"
done
rm -f "$OUT"/*.air
echo "OK: $OUT/oq_a8_qmm_nax.metallib + $OUT/oq_a8_quantize.metallib"
