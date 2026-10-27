#!/usr/bin/env bash
# Aggiorna vendor/ da ds4 upstream e riapplica le patch locali.
# Uso: ds4_sync.sh <upstream-dir | git-url> [ref]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VEND="$HERE/vendor"; PATCHES="$HERE/patches"
[ $# -ge 1 ] || { echo "uso: $0 <upstream-dir|git-url> [ref]" >&2; exit 2; }
SRC="$1"; REF="${2:-}"; TMP=""

# 1) il vendor attuale deve essere pulito (le modifiche vanno in patches/)
"$HERE/tools/ds4_check.sh" >/dev/null || { echo "vendor/ modificato: spostare le modifiche in patches/ e ripristinare" >&2; exit 3; }

# 2) sorgente upstream
if [ -d "$SRC" ] && [ -d "$SRC/.git" ]; then
  # snapshot PULITO dal commit (mai dal working tree: puo' avere debug locali)
  TMP="$(mktemp -d)"; git -C "$SRC" archive ${REF:-HEAD} | tar -x -C "$TMP" -f -
  UP="$TMP"
elif [ -d "$SRC" ]; then UP="$SRC"
else
  TMP="$(mktemp -d)"; git clone --quiet ${REF:+--branch "$REF"} --depth 1 "$SRC" "$TMP/up" >/dev/null
  UP="$TMP/up"
fi
[ -f "$UP/ds4.c" ] || { echo "non sembra ds4: $UP/ds4.c manca" >&2; exit 4; }

# 3) sostituzione secca (solo i file vendorizzati)
find "$VEND" -maxdepth 1 -type f \( -name '*.c' -o -name '*.h' -o -name '*.m' -o -name '*.inc' \) -delete
rm -rf "$VEND/metal" "$VEND/licenses" "$VEND/third_party"
cp "$UP"/LICENSE "$VEND/"
cp -R "$UP/licenses" "$VEND/"
cp -R "$UP/metal" "$VEND/"
[ -d "$UP/third_party" ] && cp -R "$UP/third_party" "$VEND/"
for f in "$UP"/*.c "$UP"/*.h "$UP"/*.m "$UP"/*.inc; do [ -e "$f" ] && cp "$f" "$VEND/"; done

# 4) patch locali
shopt -s nullglob
for p in "$PATCHES"/*.patch; do
  echo "[ds4_sync] patch $(basename "$p")"
  ( cd "$VEND" && patch -p1 --forward < "$p" )
done

# 5) manifest + provenienza
( cd "$VEND" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 shasum -a 256 > MANIFEST.sha256 )
if git -C "$UP" rev-parse HEAD >/dev/null 2>&1; then
  SHA="$(git -C "$UP" rev-parse HEAD)"; DATE="$(git -C "$UP" log -1 --format=%cd --date=iso)"
  sed -i '' -e "s/^- Commit: .*/- Commit: \`$SHA\`/" -e "s/^- Data: .*/- Data: $DATE/" "$HERE/UPSTREAM.md"
fi
[ -n "$TMP" ] && rm -rf "$TMP"
echo "[ds4_sync] ok: $(wc -l < "$VEND/MANIFEST.sha256" | tr -d ' ') file, patch applicate"
