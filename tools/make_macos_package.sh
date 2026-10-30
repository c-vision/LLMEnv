#!/usr/bin/env bash
# make_macos_package.sh — impacchetta le librerie di LLMEnv per macOS (Apple Silicon), UNA variante per volta:
#   - llmenv-<ver>-macos<14|26>-arm64.tar.gz   (lib/llmenv, include/llmenv, share/doc/llmenv)
#   - llmenv-<ver>-macos<14|26>-arm64.pkg      (installa in /usr/local)
#   - SHA256SUMS-<ver>                         (somma di tutte le varianti gia' presenti in --out)
#   - install-llmenv.sh, LATEST                (script di installazione idempotente e versione corrente)
#   - Formula/llmenv.rb                        (Homebrew: sceglie la variante dalla versione di macOS; completa solo
#                                               quando in --out ci sono ENTRAMBE le varianti)
#
# Varianti: compat = costruita con deployment target 14.0 (macOS 14+, senza i kernel NAX di MLX);
#           nax    = costruita con deployment target 26.2 (macOS 26.2+, con i kernel NAX / tensor unit M5).
#
# USO: tools/make_macos_package.sh --variant compat|nax [--version X.Y.Z] [--dist DIR] [--out DIR] [--repo owner/name]
#   --dist  cartella con le dylib gia' costruite (tools/build_ai_libs.sh --build-dir B --deployment-target T:
#           default build/compat14/dist/osx-<arch> per compat, build/nax262/dist/osx-<arch> per nax)
#   --out   cartella di output (default: build/package)
#
# Le dylib costruite in locale hanno rpath assoluti verso le cartelle di build: qui vengono rimossi, si aggiunge
# @loader_path e si rifirma ad-hoc (su arm64 una dylib modificata senza firma valida non si carica).
# Dipendenza esterna: libllm_ai_llamacpp usa OpenSSL 3 (brew install openssl@3).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

VERSION="$(tr -d '[:space:]' < VERSION)"; REPO="c-vision/LLMEnv"; OUT="$ROOT/build/package"; DIST=""; VARIANT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --dist) DIST="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --repo) REPO="$2"; shift 2 ;;
        *) echo "opzione sconosciuta: $1" >&2; exit 1 ;;
    esac
done
[ "$(uname -s)" = "Darwin" ] || { echo "solo macOS" >&2; exit 1; }
case "$(uname -m)" in arm64) ARCH=arm64 ;; *) echo "solo Apple Silicon (arm64)" >&2; exit 1 ;; esac
case "$VARIANT" in
    compat) MIN_MACOS=14.0; TAG=14; BDIR=compat14 ;;
    nax)    MIN_MACOS=26.2; TAG=26; BDIR=nax262 ;;
    *) echo "serve --variant compat|nax" >&2; exit 1 ;;
esac
[ -n "$DIST" ] || DIST="$ROOT/build/$BDIR/dist/osx-$ARCH"
[ -f "$DIST/libllm_ai_mlx.dylib" ] && [ -f "$DIST/mlx.metallib" ] || { echo "manca libllm_ai_mlx.dylib/mlx.metallib in $DIST (esegui tools/build_ai_libs.sh --build-dir build/$BDIR --deployment-target $MIN_MACOS)" >&2; exit 1; }
# la libreria deve davvero richiedere il macOS dichiarato (un build col target sbagliato non va pubblicato)
MINOS="$(otool -l "$DIST/libllm_ai_mlx.dylib" | awk '/LC_BUILD_VERSION/{r=1} r&&/minos/{print $2; exit}')"
[ "$MINOS" = "$MIN_MACOS" ] || { echo "ERRORE: libllm_ai_mlx.dylib ha minos $MINOS, la variante $VARIANT vuole $MIN_MACOS" >&2; exit 1; }

export COPYFILE_DISABLE=1   # niente file ._* (attributi estesi) nei tar/pkg
NAME="llmenv-$VERSION-macos$TAG-$ARCH"
STAGE="$OUT/stage/$NAME"
mkdir -p "$OUT"
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

# LLMENV_VERSION: lo legge install-llmenv.sh (idempotenza) e FastLLM (controllo di compatibilita' all'avvio)
{
    echo "version=$VERSION"
    echo "variant=$VARIANT"
    echo "min_macos=$MIN_MACOS"
    echo "nax=$([ "$VARIANT" = nax ] && echo 1 || echo 0)"
    echo "abi_schema=$(tr -d '[:space:]' < ABI/ABI_VERSION | head -c 40)"
} > "$STAGE/lib/llmenv/LLMENV_VERSION"
xattr -cr "$STAGE" 2>/dev/null || true

# tar.gz
tar -C "$OUT/stage" -czf "$OUT/$NAME.tar.gz" "$NAME"
# pkg (installa in /usr/local: lib/llmenv, include/llmenv, share/doc/llmenv). Idempotente: sovrascrive la stessa versione.
pkgbuild --root "$STAGE" --install-location /usr/local --identifier "io.github.c-vision.llmenv" --version "$VERSION" \
    "$OUT/$NAME.pkg" >/dev/null

# asset di installazione e versione corrente
cp -f tools/install-llmenv.sh "$OUT/install-llmenv.sh"; chmod +x "$OUT/install-llmenv.sh"
echo "$VERSION" > "$OUT/LATEST"
# SHA256SUMS-<ver>: tutte le varianti di questa versione presenti in --out
( cd "$OUT" && shasum -a 256 llmenv-"$VERSION"-macos*-arm64.tar.gz llmenv-"$VERSION"-macos*-arm64.pkg install-llmenv.sh > "SHA256SUMS-$VERSION" )

# formula Homebrew (tap: brew tap c-vision/llmenv https://github.com/$REPO). Completa solo con ENTRAMBE le varianti.
C14="$OUT/llmenv-$VERSION-macos14-$ARCH.tar.gz"; C26="$OUT/llmenv-$VERSION-macos26-$ARCH.tar.gz"
if [ -f "$C14" ] && [ -f "$C26" ]; then
    SHA14="$(shasum -a 256 "$C14" | awk '{print $1}')"; SHA26="$(shasum -a 256 "$C26" | awk '{print $1}')"
    mkdir -p "$ROOT/Formula"
    cat > "$ROOT/Formula/llmenv.rb" <<RB
class Llmenv < Formula
  desc "C libraries for running large language models locally on Apple Silicon"
  homepage "https://github.com/$REPO"
  version "$VERSION"
  license "Apache-2.0"

  # Two builds of the same engine: macOS 26.2 and later gets the one with the NAX kernels (M5 tensor units),
  # every other supported macOS (14 and later) gets the compatible one.
  if MacOS.full_version >= "26.2"
    url "https://github.com/$REPO/releases/download/v$VERSION/llmenv-$VERSION-macos26-$ARCH.tar.gz"
    sha256 "$SHA26"
  else
    url "https://github.com/$REPO/releases/download/v$VERSION/llmenv-$VERSION-macos14-$ARCH.tar.gz"
    sha256 "$SHA14"
  end

  depends_on arch: :arm64
  depends_on :macos
  depends_on "openssl@3"

  def install
    prefix.install "lib", "include", "share"
  end

  test do
    assert_predicate lib/"llmenv/libllm_ai_mlx.dylib", :exist?
    assert_predicate lib/"llmenv/mlx.metallib", :exist?
    assert_predicate lib/"llmenv/LLMENV_VERSION", :exist?
  end
end
RB
    echo "== formula Homebrew aggiornata (entrambe le varianti)"
else
    echo "== formula NON aggiornata: serve anche l'altra variante in $OUT"
fi

echo "== pacchetto: $OUT"; ls -la "$OUT"/*.tar.gz "$OUT"/*.pkg; cat "$OUT/SHA256SUMS-$VERSION"
