#!/bin/sh
# build_metallibs.sh — compila la metallib del bridge MPP q4 (kernel per-gruppo + tile).
# Uso: ./build_metallibs.sh <out-dir>
# Flag OBBLIGATORI: -std=metal4.0 (tensor-op MPP/NAX) e -mmacosx-version-min=26.2 (deployment
# floor MPP). Senza, il kernel si carica ma le tensor-op sono no-op silenziosi.
set -e
OUT="${1:?uso: $0 <out-dir>}"
CS="$(cd "$(dirname "$0")" && pwd)/csrc"
mkdir -p "$OUT"
for k in mpp_q4 q4tile; do
  xcrun -sdk macosx metal -I "$CS" -std=metal4.0 -mmacosx-version-min=26.2 \
        -c "$CS/$k.metal" -o "$OUT/$k.air"
done
xcrun -sdk macosx metal -mmacosx-version-min=26.2 "$OUT/mpp_q4.air" "$OUT/q4tile.air" \
      -o "$OUT/mpp_q4.metallib"
rm -f "$OUT"/*.air
echo "OK: $OUT/mpp_q4.metallib (mpp_q4_* + q4t_*)"
