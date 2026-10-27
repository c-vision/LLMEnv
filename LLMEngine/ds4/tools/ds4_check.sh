#!/usr/bin/env bash
# Verifica che vendor/ corrisponda al MANIFEST (nessuna modifica manuale).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE/vendor"
shasum -a 256 -c MANIFEST.sha256 --quiet
echo "ds4 vendor: manifest OK ($(wc -l < MANIFEST.sha256 | tr -d ' ') file)"
