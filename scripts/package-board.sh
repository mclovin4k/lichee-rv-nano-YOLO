#!/bin/sh
# package-board.sh -- build the board package that is published in a release.
#
#   scripts/package-board.sh <yolo_stream|yolo_face> <binary> <out-dir>
#   scripts/package-board.sh <project> <binary> <toolchain> <out-dir>   (legacy)
#   scripts/package-board.sh <project> <binary> <out-dir> --with-vendor <dir>
#
# The published package contains only this project's code: the application
# binary, the start/run scripts and the README. Vendor libraries and models are
# proprietary, are not redistributed here, and are therefore not part of the
# package; `install.sh` copies the libraries from the board firmware itself and
# uploads whatever models you provide.
#
# `--with-vendor <dir>` additionally embeds <dir>/lib and <dir>/models (as
# filled by scripts/fetch-vendor.sh). Use it for a private, self-contained
# archive; do not publish that archive.
set -eu

PROJECT=${1:-}
BIN=${2:-}
ARG3=${3:-}
ARG4=${4:-}

die() { printf 'package-board: %s\n' "$*" >&2; exit 1; }
say() { printf '\n>>> %s\n' "$*"; }

[ -n "$PROJECT" ] && [ -n "$BIN" ] && [ -n "$ARG3" ] || \
    die "usage: $0 <yolo_stream|yolo_face> <binary> <out-dir> [--with-vendor <dir>]"

# Accept the old four-argument form, where the third argument was the
# cross toolchain (it is no longer needed).
case "$ARG3" in
    */riscv64-*|*riscv64*linux*musl*|*/bin)
        OUT=$ARG4
        [ -n "$OUT" ] || die "missing output directory"
        printf 'package-board: the toolchain argument is no longer used\n' >&2
        ;;
    *)  OUT=$ARG3 ;;
esac

VENDOR=
shift 3
while [ $# -gt 0 ]; do
    case "$1" in
        --with-vendor) VENDOR=${2:-}; shift 2 ;;
        *) die "unknown option: $1" ;;
    esac
done

[ -f "$BIN" ] || die "binary not found: $BIN"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SRC="$ROOT/$PROJECT/dist/board"
[ -d "$SRC" ] || die "missing package skeleton: $SRC"

case "$PROJECT" in
    yolo_stream) APP=stream_yolo; MODEL_DIR=models ;;
    yolo_face)   APP=face_stream; MODEL_DIR=models/face ;;
    *) die "unknown project: $PROJECT" ;;
esac

STAGE=$(mktemp -d "${TMPDIR:-/tmp}/lichee-${PROJECT}-pkg.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT INT TERM

say "staging $PROJECT"
# Scripts and README only: the runtime/ and models/ directories in the working
# tree are local vendor copies and are never packaged from there.
for f in "$SRC"/*.sh "$SRC/README.md"; do
    [ -f "$f" ] && cp -a "$f" "$STAGE/"
done
mkdir -p "$STAGE/$MODEL_DIR"
install -m 0755 "$BIN" "$STAGE/$APP"
chmod 0755 "$STAGE"/*.sh

if [ -n "$VENDOR" ]; then
    [ -d "$VENDOR" ] || die "no such vendor directory: $VENDOR"
    say "embedding vendor files from $VENDOR (do not publish this archive)"
    if [ -d "$VENDOR/lib" ]; then
        mkdir -p "$STAGE/runtime"
        cp -a "$VENDOR/lib/." "$STAGE/runtime/"
    fi
    if [ -d "$VENDOR/models" ]; then
        cp -a "$VENDOR/models/." "$STAGE/$MODEL_DIR/"
    fi
fi

say "checksums"
(
    cd "$STAGE"
    find . -type f ! -name md5sums.txt -printf '%P\0' \
        | sort -z \
        | xargs -0 md5sum > md5sums.txt
)

mkdir -p "$OUT"
ASSET="$OUT/$PROJECT-board.tar.gz"
tar -C "$STAGE" --owner=0 --group=0 --numeric-owner -czf "$ASSET" .
(
    cd "$OUT"
    sha256sum "$PROJECT-board.tar.gz" > "$PROJECT-board.tar.gz.sha256"
)

say "result"
ls -lh "$ASSET" "$ASSET.sha256"
printf '\nContents (%s):\n' "$APP"
tar -tzf "$ASSET" | sed -n '1,25p'
