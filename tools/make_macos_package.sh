#!/usr/bin/env bash
# make_macos_package.sh — impacchetta le librerie di LLMEnv per macOS (Apple Silicon):
#   - llmenv-<ver>-macos-<arch>.tar.gz   (lib/llmenv, include/llmenv, share/doc/llmenv)
#   - llmenv-<ver>-macos-<arch>.pkg      (installa in /usr/local)
#   - SHA256SUMS
#   - Formula/llmenv.rb                  (Homebrew, aggiornata con url e sha256 del tar.gz)
#
# USO: tools/make_macos_package.sh [--version X.Y.Z] [--dist DIR] [--out DIR] [--repo owner/name]
#   --dist  cartella con le dylib gia' costruite (default: build/dist/osx-<arch>, prodotta da tools/build_ai_libs.sh)
#   --out   cartella di output (default: build/package)
#
# Le dylib costruite in locale hanno rpath assoluti verso le cartelle di build: qui vengono rimossi, si aggiunge
# @loader_path e si rifirma ad-hoc (su arm64 una dylib modificata senza firma valida non si carica).
# Dipendenza esterna: libllm_ai_llamacpp usa OpenSSL 3 (brew install openssl@3).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

VERSION="$(tr -d '[:space:]' < VERSION)"; REPO="c-vision/LLMEnv"; OUT="$ROOT/build/package"; DIST=""
while [ $# -gt 0 ]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --dist) DIST="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --repo) REPO="$2"; shift 2 ;;
        *) echo "opzione sconosciuta: $1" >&2; exit 1 ;;
    esac
done
[ "$(uname -s)" = "Darwin" ] || { echo "solo macOS" >&2; exit 1; }
case "$(uname -m)" in arm64) ARCH=arm64 ;; *) echo "solo Apple Silicon (arm64)" >&2; exit 1 ;; esac
[ -n "$DIST" ] || DIST="$ROOT/build/dist/osx-$ARCH"
[ -f "$DIST/libllm_ai_mlx.dylib" ] && [ -f "$DIST/mlx.metallib" ] || { echo "manca libllm_ai_mlx.dylib/mlx.metallib in $DIST (esegui tools/build_ai_libs.sh)" >&2; exit 1; }

export COPYFILE_DISABLE=1   # niente file ._* (attributi estesi) nei tar/pkg
NAME="llmenv-$VERSION-macos-$ARCH"
STAGE="$OUT/stage/$NAME"
rm -rf "$OUT/stage" "$OUT/$NAME.tar.gz" "$OUT/$NAME.pkg"
mkdir -p "$STAGE/lib/llmenv" "$STAGE/include/llmenv" "$STAGE/share/doc/llmenv"

# librerie del motore: solo le nostre (llama.cpp e' incorporato staticamente in libllm_ai_llamacpp)
for f in "$DIST"/libllm*.dylib "$DIST"/mlx.metallib; do cp -f "$f" "$STAGE/lib/llmenv/"; done

# rpath assoluti -> @loader_path; firma ad-hoc
for f in "$STAGE"/lib/llmenv/*.dylib; do
    while IFS= read -r rp; do
        case "$rp" in @*) ;; *) install_name_tool -delete_rpath "$rp" "$f" 2>/dev/null || true ;; esac
    done < <(otool -l "$f" | awk '/LC_RPATH/{r=1} r&&/ path /{print $2; r=0}')
    otool -l "$f" | grep -q 'path @loader_path ' || install_name_tool -add_rpath @loader_path "$f"
    codesign --force -s - "$f" >/dev/null 2>&1
done
# nessun percorso assoluto di build deve restare
if otool -l "$STAGE"/lib/llmenv/*.dylib | awk '/LC_RPATH/{r=1} r&&/ path /{print $2; r=0}' | grep -q '^/'; then
    echo "ERRORE: rpath assoluti residui" >&2; exit 1
fi

# descrittori dei modelli (.gmbarch): il motore li cerca in ABI/ accanto alle librerie (llmdesc_abi_dir risale dalla dylib)
mkdir -p "$STAGE/lib/llmenv/ABI"
cp -f ABI/*.gmbarch ABI/ABI_VERSION "$STAGE/lib/llmenv/ABI/"

# header pubblici
cp -f LLMEngine/include/*.h LLMToken/llmtoken.h LLMCodec/include/*.h LLMDesc/llmdesc.h LLMJinja/llmjinja.h "$STAGE/include/llmenv/"
cp -f LICENSE NOTICE README.md "$STAGE/share/doc/llmenv/"
echo "$VERSION" > "$STAGE/share/doc/llmenv/VERSION"
xattr -cr "$STAGE" 2>/dev/null || true

# tar.gz
tar -C "$OUT/stage" -czf "$OUT/$NAME.tar.gz" "$NAME"
# pkg (installa in /usr/local: lib/llmenv, include/llmenv, share/doc/llmenv)
pkgbuild --root "$STAGE" --install-location /usr/local --identifier "io.github.c-vision.llmenv" --version "$VERSION" \
    "$OUT/$NAME.pkg" >/dev/null

( cd "$OUT" && shasum -a 256 "$NAME.tar.gz" "$NAME.pkg" > SHA256SUMS )
SHA="$(shasum -a 256 "$OUT/$NAME.tar.gz" | awk '{print $1}')"

# formula Homebrew (tap: brew tap c-vision/llmenv https://github.com/$REPO)
mkdir -p "$ROOT/Formula"
cat > "$ROOT/Formula/llmenv.rb" <<RB
class Llmenv < Formula
  desc "C libraries for running large language models locally on Apple Silicon"
  homepage "https://github.com/$REPO"
  url "https://github.com/$REPO/releases/download/v$VERSION/$NAME.tar.gz"
  sha256 "$SHA"
  version "$VERSION"
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on :macos
  depends_on "openssl@3"

  def install
    prefix.install "lib", "include", "share"
  end

  test do
    assert_predicate lib/"llmenv/libllm_ai_mlx.dylib", :exist?
    assert_predicate lib/"llmenv/mlx.metallib", :exist?
  end
end
RB

echo "== pacchetto: $OUT"; ls -la "$OUT"/*.tar.gz "$OUT"/*.pkg; cat "$OUT/SHA256SUMS"
