#!/bin/zsh
# dsv41_run_tests.sh — costruisce (CMake, dipendenze tracciate) ed esegue TUTTI i test ds41-native.
# Fixture generate in C (tools/dsv41_fixtures.sh), nessun Python. Uso: tools/dsv41_run_tests.sh [filtro]
# Richiede una build configurata in <repo>/build (cmake -S . -B build -DLLMENV_BUILD_ENGINE=ON).
set -u
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
zsh "$(dirname "$0")/dsv41_fixtures.sh" >/dev/null || exit 1
zsh "$(dirname "$0")/dsv41_fixtures.sh" --no-probes --out /tmp/dsv41_st_np >/dev/null || exit 1
cmake -S "$ROOT" -B "$ROOT/build" -DLLM_BUILD_DSV4_TESTS=ON >/dev/null || exit 1
T=$(cmake --build "$ROOT/build" --target help | grep -o "llm_dsv41_[a-z0-9_]*_test" | sort -u | grep "${1:-.}" | tr '\n' ' ')
cmake --build "$ROOT/build" -j8 --target ${=T} -- -k 2>&1 | grep -E "error:" 
pass=0; fail=0
for t in ${=T}; do
  b="$ROOT/build/LLMEngine/mlx/$t"
  if [[ -x $b ]] && timeout 300 $b >"${TMPDIR:-/tmp}/$t.out" 2>&1; then pass=$((pass+1)); else fail=$((fail+1)); echo "FAIL $t  (${TMPDIR:-/tmp}/$t.out)"; fi
done
echo "ds41-native: $pass passati, $fail falliti"
[[ $fail -eq 0 ]]
