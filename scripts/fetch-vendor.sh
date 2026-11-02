#!/bin/sh
# fetch-vendor.sh -- collect the vendor libraries and models that the board
# applications need.
#
# The repository contains no Cvitek/Sophgo binaries: they are proprietary and
# are not redistributed here. This script copies them from a source you already
# have (your board, or a vendor SDK/image you downloaded yourself) and can
# additionally download the parts Sophgo publishes on GitHub.
#
#   scripts/fetch-vendor.sh --board 192.0.2.100   # from a running board (best)
#   scripts/fetch-vendor.sh --sdk ~/cvitek-tdl-sdk-sg200x
#   scripts/fetch-vendor.sh --models ~/models/my-detector
#   scripts/fetch-vendor.sh --download            # upstream part only
#
# Layout of the destination (default: ./vendor, ignored by git):
#
#   vendor/lib/*.so                 vendor libraries used at run time
#   vendor/models/*.cvimodel        models
#   vendor/cviruntime/include/*.h   cviruntime headers (needed to build)
#   vendor/cviruntime/lib/*.so      libcviruntime/libcvikernel (needed to build)
#
# `scripts/build.sh`, `install.sh` and the packaging scripts use this directory.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DEST=${DEST:-"$ROOT/vendor"}
BOARD_USER=${BOARD_USER:-root}
SSH_OPTS=${SSH_OPTS:-"-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"}

# Libraries the applications load at run time. The names are the exact sonames
# used by the board firmware and the vendor SDK.
LIBS="libcviruntime.so libcvikernel.so libvenc.so libsys.so libisp.so
libisp_algo.so libae.so libawb.so libsns_gc4653.so libcvi_bin.so
libcvi_bin_isp.so libcvi_tdl.so libcvimath.so libcnpy.so libvpu_tdl.so
libcvi_ive.so libopencv_core.so.3.2 libopencv_imgproc.so.3.2
libopencv_imgcodecs.so.3.2"

# Required to link the applications; the rest is used at run time only.
LINK_LIBS="libcviruntime.so libcvikernel.so"

SDK_REPO=${SDK_REPO:-https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x.git}
SDK_COMMIT=${SDK_COMMIT:-02dfb9091d390a7b0d1177fe7b651db1b99854bc}
CVIRT_REPO=${CVIRT_REPO:-https://github.com/sophgo/cviruntime.git}
CVIRT_COMMIT=${CVIRT_COMMIT:-ef8044988c2b4a5d491125d13e6f048b5f8a1389}

MODE=
BOARD_IP=
SDK_DIR=
MODELS_DIR=

usage() {
    sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --board <IP>      copy libraries and models from a running board over SSH
  --sdk <DIR>       copy from a locally extracted cvitek TDL SDK or image
  --models <DIR>    copy your own exported models (*.cvimodel)
  --download        download the parts Sophgo publishes on GitHub
  --dest <DIR>      output directory (default: <repo>/vendor)
  -h, --help        this help

Options may be combined; the script reports whatever is still missing.
EOF
}

say() { printf '\n>>> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --board)    MODE="$MODE board";   BOARD_IP=${2:-};  shift 2 ;;
        --sdk)      MODE="$MODE sdk";     SDK_DIR=${2:-};   shift 2 ;;
        --models)   MODE="$MODE models";  MODELS_DIR=${2:-}; shift 2 ;;
        --download) MODE="$MODE download"; shift ;;
        --dest)     DEST=${2:-}; shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *)          die "unknown option: $1 (try --help)" ;;
    esac
done

[ -n "$MODE" ] || { usage; exit 1; }
for m in $MODE; do
    case "$m" in
        board)  [ -n "$BOARD_IP" ] || die "--board needs an address" ;;
        sdk)    [ -d "$SDK_DIR" ] || die "not a directory: $SDK_DIR" ;;
        models) [ -d "$MODELS_DIR" ] || die "not a directory: $MODELS_DIR" ;;
    esac
done

mkdir -p "$DEST/lib" "$DEST/models" \
         "$DEST/cviruntime/include" "$DEST/cviruntime/lib"

copy_lib() { # copy_lib <file>
    [ -f "$1" ] || return 1
    name=$(basename "$1")
    [ -e "$DEST/lib/$name" ] && return 0
    cp -L "$1" "$DEST/lib/$name"
    printf '  lib/%s\n' "$name"
}

copy_models() { # copy_models <dir>: flatten every model below <dir>
    [ -d "$1" ] || return 0
    find "$1" -type f \( -name '*.cvimodel' -o -name '*.names' \) | \
    while IFS= read -r f; do
        name=$(basename "$f")
        [ -e "$DEST/models/$name" ] && continue
        cp -L "$f" "$DEST/models/$name"
        printf '  models/%s\n' "$name"
    done
}

copy_headers() { # copy_headers <dir>
    [ -d "$1" ] || return 0
    for h in cviruntime.h cvitpu_debug.h; do
        [ -f "$1/$h" ] || continue
        [ -e "$DEST/cviruntime/include/$h" ] && continue
        cp -f "$1/$h" "$DEST/cviruntime/include/$h"
        printf '  cviruntime/include/%s\n' "$h"
    done
}

copy_tree_libs() { # copy_tree_libs <dir>
    [ -d "$1" ] || return 0
    for n in $LIBS; do
        copy_lib "$1/$n" || true
    done
}

# ---------------------------------------------------------------- from board
fetch_board() {
    say "board: $BOARD_USER@$BOARD_IP"
    TMP=$(mktemp -d "${TMPDIR:-/tmp}/fetch-vendor.XXXXXX")
    trap 'rm -rf "$TMP"' EXIT INT TERM

    ssh $SSH_OPTS "$BOARD_USER@$BOARD_IP" true 2>/dev/null || \
        die "cannot reach $BOARD_USER@$BOARD_IP over SSH"

    ssh $SSH_OPTS "$BOARD_USER@$BOARD_IP" sh -s > "$TMP/vendor.tar" <<EOF || \
        die "copy from the board failed"
set -e
T=\$(mktemp -d /tmp/vendor.XXXXXX)
mkdir -p "\$T/lib" "\$T/models"
for n in $(printf '%s' "$LIBS"); do
    for d in /mnt/system/lib /mnt/system/usr/lib /mnt/system/usr/lib/3rd \
             /usr/lib /lib; do
        if [ -f "\$d/\$n" ]; then cp -L "\$d/\$n" "\$T/lib/\$n"; break; fi
    done
done
for d in /mnt/system/usr/bin/ai/models /root/models \
         /root/yolo_stream/models /root/yolo_face/models; do
    [ -d "\$d" ] || continue
    find "\$d" -type f -name '*.cvimodel' -exec cp -f {} "\$T/models/" \; 2>/dev/null || true
    find "\$d" -type f -name '*.names' -exec cp -f {} "\$T/models/" \; 2>/dev/null || true
done
cd "\$T"
tar -cf - lib models
cd /
rm -rf "\$T"
EOF

    [ -s "$TMP/vendor.tar" ] || die "nothing was copied from the board"
    tar -xf "$TMP/vendor.tar" -C "$TMP"
    for f in "$TMP"/lib/*; do [ -f "$f" ] && copy_lib "$f"; done
    copy_models "$TMP/models"
    rm -rf "$TMP"
    trap - EXIT INT TERM
}

# ------------------------------------------------------------------ from SDK
fetch_sdk() {
    say "SDK: $SDK_DIR"
    for d in "$SDK_DIR/sample/3rd/middleware/v2/lib" \
             "$SDK_DIR/sample/3rd/middleware/v2/lib/3rd" \
             "$SDK_DIR/sample/3rd/tpu/lib" \
             "$SDK_DIR/sample/3rd/opencv/lib" \
             "$SDK_DIR/lib" \
             "$SDK_DIR/mnt/system/lib" \
             "$SDK_DIR/mnt/system/usr/lib"; do
        copy_tree_libs "$d"
    done
    for d in "$SDK_DIR/cvimodel" "$SDK_DIR/sample/3rd/models" \
             "$SDK_DIR/mnt/system/usr/bin/ai/models"; do
        copy_models "$d"
    done
    for d in "$SDK_DIR/sample/3rd/middleware/v2/include" \
             "$SDK_DIR/sample/3rd/tpu/include" "$SDK_DIR/include"; do
        copy_headers "$d"
    done
}

# ---------------------------------------------------------------- own models
fetch_models() {
    say "models: $MODELS_DIR"
    copy_models "$MODELS_DIR"
}

# ----------------------------------------------------------------- upstream
fetch_download() {
    say "upstream: cvitek-tdl-sdk-sg200x @ $SDK_COMMIT"
    TMP=$(mktemp -d "${TMPDIR:-/tmp}/fetch-vendor-dl.XXXXXX")
    trap 'rm -rf "$TMP"' EXIT INT TERM
    git -C "$TMP" init -q
    git -C "$TMP" remote add origin "$SDK_REPO"
    git -C "$TMP" config core.sparseCheckout true
    mkdir -p "$TMP/.git/info"
    cat > "$TMP/.git/info/sparse-checkout" <<'EOF'
lib/
cvimodel/
sample/3rd/middleware/v2/lib/
sample/3rd/tpu/lib/
sample/3rd/opencv/lib/
sample/3rd/middleware/v2/include/
EOF
    if git -C "$TMP" fetch -q --depth 1 origin "$SDK_COMMIT" 2>/dev/null && \
            git -C "$TMP" checkout -q FETCH_HEAD 2>/dev/null; then
        for d in "$TMP/sample/3rd/middleware/v2/lib" \
                 "$TMP/sample/3rd/tpu/lib" "$TMP/sample/3rd/opencv/lib" \
                 "$TMP/lib"; do
            copy_tree_libs "$d"
        done
        copy_models "$TMP/cvimodel"
        copy_headers "$TMP/sample/3rd/middleware/v2/include"
    else
        warn "cannot fetch $SDK_REPO"
    fi
    rm -rf "$TMP"
    trap - EXIT INT TERM

    say "upstream: sophgo/cviruntime @ $CVIRT_COMMIT (headers)"
    TMP=$(mktemp -d "${TMPDIR:-/tmp}/fetch-vendor-hdr.XXXXXX")
    trap 'rm -rf "$TMP"' EXIT INT TERM
    git -C "$TMP" init -q
    git -C "$TMP" remote add origin "$CVIRT_REPO"
    if git -C "$TMP" fetch -q --depth 1 origin "$CVIRT_COMMIT" 2>/dev/null && \
            git -C "$TMP" checkout -q FETCH_HEAD 2>/dev/null; then
        copy_headers "$TMP/include"
    else
        warn "cannot fetch $CVIRT_REPO"
    fi
    rm -rf "$TMP"
    trap - EXIT INT TERM
}

# --------------------------------------------------------------------- main
for m in $MODE; do
    case "$m" in
        board)    fetch_board ;;
        sdk)      fetch_sdk ;;
        models)   fetch_models ;;
        download) fetch_download ;;
    esac
done

if [ ! -f "$DEST/cviruntime/include/cviruntime.h" ]; then
    fetch_download
fi

# Libraries found on the board or in the SDK are also the link-time ones.
for n in $LINK_LIBS; do
    if [ -f "$DEST/lib/$n" ] && [ ! -e "$DEST/cviruntime/lib/$n" ]; then
        cp -L "$DEST/lib/$n" "$DEST/cviruntime/lib/$n"
    fi
done

say "result: $DEST"
missing=
for n in $LIBS; do
    [ -e "$DEST/lib/$n" ] || missing="$missing $n"
done
[ -n "$missing" ] && warn "not found:$missing"

link_missing=
for n in $LINK_LIBS; do
    [ -e "$DEST/cviruntime/lib/$n" ] || link_missing="$link_missing $n"
done
if [ -n "$link_missing" ]; then
    cat >&2 <<EOF

warning: $link_missing not found.
These two libraries are only distributed inside the board firmware and the
vendor SDK, so they cannot be downloaded from GitHub:

  scripts/fetch-vendor.sh --board <IP>          # from your board
  scripts/fetch-vendor.sh --sdk <extracted SDK>

Without them the applications still run on a board (the libraries are already
there), but they cannot be cross-compiled.
EOF
fi

cat <<EOF

Vendor files: $DEST
  lib/         $(ls -1 "$DEST/lib" 2>/dev/null | wc -l) files
  models/      $(ls -1 "$DEST/models" 2>/dev/null | wc -l) files
  cviruntime/  $(ls -1 "$DEST/cviruntime/lib" 2>/dev/null | wc -l) libs, \
$(ls -1 "$DEST/cviruntime/include" 2>/dev/null | wc -l) headers

Next:
  cd yolo_stream && ./scripts/build.sh     # or yolo_face
  ./install.sh <IP>                        # uploads binary + libraries + models
EOF
