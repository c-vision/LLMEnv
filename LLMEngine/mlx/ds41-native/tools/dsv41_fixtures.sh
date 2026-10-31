#!/bin/zsh
# dsv41_fixtures.sh — genera i checkpoint sintetici tiny CSA2 per i test ds41-native (C puro, no Python).
# Uso: tools/dsv41_fixtures.sh [--out <dir>] [--no-probes]   (default: /tmp/dsv41_st)
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="${TMPDIR:-/tmp}/llm_dsv41_fixture_gen"
/usr/bin/cc -std=gnu11 -O1 -Wall -Wno-unused-function "$HERE/llm_dsv41_fixture_gen.c" -lm -o "$BIN"
"$BIN" "$@"
