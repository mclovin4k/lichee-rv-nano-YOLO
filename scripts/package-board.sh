#!/bin/sh
# package-board.sh -- build a complete, self-contained board package.
#
#   scripts/package-board.sh <yolo_stream|yolo_face> <binary> <toolchain> <out-dir>
#
# The package contains the application binary, start/run scripts, models and
# every runtime library the binary needs (musl loader + libc, libgcc, libatomic
# plus the vendor libraries already stored in <project>/dist/board/runtime).
# A release tarball is written to <out-dir>/<project>-board.tar.gz together
# with a SHA256 file.
set -eu

PROJECT=${1:-}
BIN=${2:-}
TOOLCHAIN=${3:-}
OUT=${4:-}

die() { printf 'package-board: %s\n' "$*" >&2; exit 1; }
say() { printf '\n>>> %s\n' "$*"; }

[ -n "$PROJECT" ] && [ -n "$BIN" ] && [ -n "$TOOLCHAIN" ] && [ -n "$OUT" ] || \
    die "usage: $0 <yolo_stream|yolo_face> <binary> <toolchain> <out-dir>"
[ -f "$BIN" ] || die "binary not found: $BIN"
[ -d "$TOOLCHAIN" ] || die "toolchain not found: $TOOLCHAIN"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SRC="$ROOT/$PROJECT/dist/board"
[ -d "$SRC" ] || die "missing package skeleton: $SRC"

case "$PROJECT" in
    yolo_stream) APP=stream_yolo ;;
    yolo_face)   APP=face_stream ;;
    *) die "unknown project: $PROJECT" ;;
esac

# The musl.cc and musl-cross archives use slightly different directory names.
SYSROOT=
for d in \
    "$TOOLCHAIN/riscv64-unknown-linux-musl/sysroot" \
    "$TOOLCHAIN/riscv64-linux-musl/sysroot"; do
    [ -d "$d" ] && { SYSROOT=$d; break; }
done
[ -n "$SYSROOT" ] || die "no musl sysroot under $TOOLCHAIN"

find_lib() { # find_lib <name>
    for d in "$SYSROOT/lib" "$SYSROOT/usr/lib"; do
        [ -e "$d/$1" ] && { printf '%s\n' "$d/$1"; return 0; }
    done
    return 1
}

STAGE=$(mktemp -d "${TMPDIR:-/tmp}/lichee-${PROJECT}-pkg.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT INT TERM

say "staging $PROJECT"
cp -a "$SRC/." "$STAGE/"
rm -f "$STAGE/$APP"
install -m 0755 "$BIN" "$STAGE/$APP"

mkdir -p "$STAGE/runtime"
for name in ld-musl-riscv64.so.1 libc.so libatomic.so.1 libgcc_s.so.1 \
            libstdc++.so.6; do
    src=$(find_lib "$name") || die "missing runtime library in toolchain: $name"
    cp -L "$src" "$STAGE/runtime/$name"
done

# The toolchain ships an unstripped libstdc++ (16 MB); only the dynamic
# symbols are needed at runtime.
STRIP=
for candidate in "$TOOLCHAIN"/bin/*-strip; do
    [ -x "$candidate" ] && { STRIP=$candidate; break; }
done
if [ -n "$STRIP" ]; then
    "$STRIP" --strip-unneeded "$STAGE/runtime/libstdc++.so.6" || true
fi

# libcviruntime/libcvikernel are built from third_party/cviruntime. Add them
# when the runtime skeleton does not already ship its own copy (yolo_face
# does, yolo_stream does not).
for lib in "$ROOT/$PROJECT/third_party/cviruntime/lib/"*.so; do
    [ -f "$lib" ] || continue
    name=$(basename "$lib")
    [ -e "$STAGE/runtime/$name" ] || cp -L "$lib" "$STAGE/runtime/$name"
done

# Keep the loader executable: some file systems lose the bit after archiving.
chmod 0755 "$STAGE/runtime/ld-musl-riscv64.so.1"
chmod 0755 "$STAGE/$APP" "$STAGE"/*.sh 2>/dev/null || true

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
