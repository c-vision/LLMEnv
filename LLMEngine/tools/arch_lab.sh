#!/bin/bash
# arch_lab.sh — build+run del "architecture lab" (OpenArch A3): test per-modulo con golden,
# funzioni reali del motore (ds4, C) vs riferimento CPU double. Offline, deterministico.
set -e
R="$(cd "$(dirname "$0")/../.." && pwd)"
INC=(-I "$R/LLMEngine/mlx/ds4" -I "$R/LLMEngine/mlx/ds41-native" -I "$R/LLMEngine/mlx" -I "$R/LLMCodec/include" -I "$R/LLMEngine/include" -I "$R/LLMEngine/Options")
clang -O2 "${INC[@]}" "$R/LLMEngine/tools/arch_lab.c" -lm -o /tmp/arch_lab
exec /tmp/arch_lab
