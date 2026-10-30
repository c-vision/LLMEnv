#!/usr/bin/env bash
# install-llmenv.sh — installs the LLMEnv engine (macOS, Apple Silicon). IDEMPOTENT.
#
#   curl -fsSL https://github.com/c-vision/LLMEnv/releases/latest/download/install-llmenv.sh | bash
#
# If a suitable copy of LLMEnv is already on the machine (Homebrew, the .pkg, a previous run of this script, $LLMENV_HOME)
# it does NOTHING and says so. Otherwise it downloads the variant that fits this macOS, verifies the SHA-256 and installs it
# without sudo in ~/.local, swapping the version atomically.
#
# Usage: install-llmenv.sh [--version X.Y.Z|latest] [--prefix DIR] [--variant auto|compat|nax]
#                        [--from DIR|URL] [--force] [--check] [--uninstall] [--quiet]
#   --prefix     where to install (default ~/.local): lib/llmenv, include/llmenv, opt/llmenv/<version>-<variant>
#   --variant    compat = macOS 14+ without the NAX kernels; nax = macOS 26.2+ with the NAX kernels (M5 tensor units).
#                auto picks nax on 26.2 and later, compat otherwise
#   --from       local folder or base URL with the release files (offline tests; default GitHub)
#   --force      reinstall even if a suitable version is already there
#   --check      do not install: say where LLMEnv is and its version (exit 0 = suitable, 1 = needs install)
#   --uninstall  remove the installation made by this script from the prefix
# Environment: LLMENV_HOME (lib/llmenv folder to check first).
set -euo pipefail

REPO="c-vision/LLMEnv"
VERSION="latest"; PREFIX="${HOME}/.local"; VARIANT="auto"; FROM=""; FORCE=0; CHECK=0; UNINSTALL=0; QUIET=0
while [ $# -gt 0 ]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --from) FROM="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --check) CHECK=1; shift ;;
        --uninstall) UNINSTALL=1; shift ;;
        --quiet) QUIET=1; shift ;;
        -h|--help) sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "install-llmenv: unknown option: $1" >&2; exit 2 ;;
    esac
done
say() { [ "$QUIET" = 1 ] || echo "$@"; }
die() { echo "install-llmenv: $*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "macOS only"
[ "$(uname -m)" = "arm64" ] || die "an Apple Silicon Mac (arm64) is required"
OSVER="$(sw_vers -productVersion)"

# version compare "a.b.c": returns 0 if $1 >= $2
ver_ge() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -t. -k1,1n -k2,2n -k3,3n | head -1)" = "$2" ]; }

# --- variant ---
case "$VARIANT" in
    auto) if ver_ge "$OSVER" "26.2"; then VARIANT=nax; else VARIANT=compat; fi ;;
    compat|nax) ;;
    *) die "unknown variant: $VARIANT (auto|compat|nax)" ;;
esac
if [ "$VARIANT" = nax ] && ! ver_ge "$OSVER" "26.2"; then die "the nax variant needs macOS 26.2 or later (this is $OSVER)"; fi
if ver_ge "$OSVER" "14.0"; then :; else die "macOS 14 or later is required (this is $OSVER)"; fi
MIN_TAG=$([ "$VARIANT" = nax ] && echo 26 || echo 14)

# --- LLMENV_VERSION: reading an existing install ---
# prints "version variant" if the lib/llmenv folder has a readable LLMENV_VERSION file
read_install() {
    local f="$1/LLMENV_VERSION"; [ -f "$f" ] || return 1
    local v var
    v="$(sed -n 's/^version=//p' "$f" | head -1)"; var="$(sed -n 's/^variant=//p' "$f" | head -1)"
    [ -n "$v" ] || return 1
    echo "$v ${var:-compat}"
}
candidates() {
    [ -n "${LLMENV_HOME:-}" ] && echo "$LLMENV_HOME"
    echo "$PREFIX/lib/llmenv"
    if command -v brew >/dev/null 2>&1; then
        local p; p="$(brew --prefix llmenv 2>/dev/null || true)"; [ -n "$p" ] && echo "$p/lib/llmenv"
    fi
    echo "/opt/homebrew/opt/llmenv/lib/llmenv"
    echo "/usr/local/lib/llmenv"
}

# --- uninstall ---
if [ "$UNINSTALL" = 1 ]; then
    for l in "$PREFIX/lib/llmenv" "$PREFIX/include/llmenv"; do [ -L "$l" ] && rm -f "$l"; done
    rm -rf "$PREFIX/opt/llmenv"
    say "LLMEnv removed from $PREFIX"; exit 0
fi

# --- version resolution ---
fetch() { # $1 = URL o percorso -> stdout
    case "$1" in http*) curl -fsSL --retry 3 "$1" ;; *) cat "$1" ;; esac
}
if [ "$VERSION" = latest ]; then
    if [ -n "$FROM" ]; then
        VERSION="$(fetch "$FROM/LATEST" 2>/dev/null | tr -d '[:space:]')"
        [ -n "$VERSION" ] || die "with --from the LATEST file (or --version X.Y.Z) is needed"
    else
        VERSION="$(curl -fsSL --retry 3 "https://api.github.com/repos/$REPO/releases/latest" | sed -n 's/.*"tag_name": *"v\{0,1\}\([^"]*\)".*/\1/p' | head -1)"
        [ -n "$VERSION" ] || die "cannot read the latest release of $REPO (network?)"
    fi
fi

# --- idempotence: is a suitable version already there? ---
found=""
while IFS= read -r c; do
    [ -n "$c" ] || continue
    if info="$(read_install "$c")"; then
        iv="${info% *}"; ivar="${info#* }"
        if [ "$iv" = "$VERSION" ]; then found="$c ($iv, $ivar)"; FOUND_PATH="$c"; break; fi
    fi
done < <(candidates)
if [ "$CHECK" = 1 ]; then
    if [ -n "$found" ]; then echo "LLMEnv $VERSION already present: $found"; exit 0; fi
    echo "LLMEnv $VERSION not found"; exit 1
fi
if [ -n "$found" ] && [ "$FORCE" = 0 ]; then
    say "LLMEnv $VERSION is already installed: $found. Nothing to do."; exit 0
fi

# --- download and verification ---
NAME="llmenv-$VERSION-macos$MIN_TAG-arm64"
BASE="${FROM:-https://github.com/$REPO/releases/download/v$VERSION}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
say "== LLMEnv $VERSION (variant $VARIANT, macOS $OSVER) from $BASE"
fetch "$BASE/$NAME.tar.gz" > "$TMP/$NAME.tar.gz" || die "download failed: $BASE/$NAME.tar.gz"
fetch "$BASE/SHA256SUMS-$VERSION" > "$TMP/SHA256SUMS" || die "download failed: SHA256SUMS-$VERSION"
want="$(awk -v n="$NAME.tar.gz" '$2==n || $2=="*"n {print $1}' "$TMP/SHA256SUMS" | head -1)"
[ -n "$want" ] || die "$NAME.tar.gz is not in SHA256SUMS-$VERSION"
have="$(shasum -a 256 "$TMP/$NAME.tar.gz" | awk '{print $1}')"
[ "$want" = "$have" ] || die "SHA-256 mismatch (expected $want, got $have): the download is corrupt or was altered"
say "   SHA-256 verified"

# --- install with an atomic swap ---
ID="$VERSION-$VARIANT"
DEST="$PREFIX/opt/llmenv/$ID"
mkdir -p "$PREFIX/opt/llmenv" "$PREFIX/lib" "$PREFIX/include"
tar -C "$TMP" -xzf "$TMP/$NAME.tar.gz"
[ -f "$TMP/$NAME/lib/llmenv/libllm_ai_mlx.dylib" ] || die "the package does not contain libllm_ai_mlx.dylib"
rm -rf "$DEST.new"; mv "$TMP/$NAME" "$DEST.new"
xattr -cr "$DEST.new" 2>/dev/null || true       # no quarantine flag: curl does not set it, a browser download does
rm -rf "$DEST"; mv "$DEST.new" "$DEST"
# rename(2) is atomic even on a symlink to a folder (mv is not); perl ships with every macOS
swap() { ln -s "$1" "$2.tmp.$$" && perl -e 'rename($ARGV[0],$ARGV[1]) or die "rename: $!"' "$2.tmp.$$" "$2"; }
swap "$ID" "$PREFIX/opt/llmenv/current"
swap "../opt/llmenv/current/lib/llmenv" "$PREFIX/lib/llmenv"
swap "../opt/llmenv/current/include/llmenv" "$PREFIX/include/llmenv"
# keeps the previous version for a manual rollback, removes older ones
ls -1dt "$PREFIX"/opt/llmenv/*/ 2>/dev/null | sed 's:/$::' | grep -v -- "/$ID\$" | tail -n +2 | while IFS= read -r old; do rm -rf "$old"; done

say "== LLMEnv $VERSION installed in $PREFIX/lib/llmenv"
if ! command -v otool >/dev/null 2>&1; then :; fi
exit 0
