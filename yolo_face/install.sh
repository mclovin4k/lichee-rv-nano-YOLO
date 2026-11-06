#!/bin/sh
# install.sh -- install face_stream on a LicheeRV Nano.
#
#   ./install.sh                        # download the release, ask for the IP
#   BOARD_IP=<IP> ./install.sh
#   ./install.sh <IP>
#   MODELS_DIR=~/models/face ./install.sh <IP>          # upload your models
#   ./install.sh --build <IP>           # cross-compile here instead of downloading
#   RELEASE=v1.0.1 ./install.sh <IP>    # pin a release
#
# The release package contains this project's code only (binary + scripts).
# The vendor libraries are proprietary and are not redistributed, so install.sh
# keeps the copies already on the board, uploads local copies when available
# and downloads the parts Sophgo publishes on GitHub for anything still missing.
# Models are uploaded from MODELS_DIR, ../vendor/models or the local package;
# the stock detector is downloaded when it is absent, while the recognition and
# liveness models must come from the board/SDK or from MODELS_DIR.
#
# The board address is not fixed: USB/RNDIS, Wi-Fi and Ethernet differ.
# Nothing is started automatically.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT=yolo_face
APP=face_stream
ASSET=yolo_face-board.tar.gz
DEST=/root/yolo_face
BOARD_USER=${BOARD_USER:-root}
SSH_OPTS=${SSH_OPTS:-"-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"}
RELEASE=${RELEASE:-latest}
REPO_URL=${REPO_URL:-https://github.com/mclovin4k/lichee-rv-nano-YOLO}
PAYLOAD_DIR=${PAYLOAD_DIR:-"$ROOT/dist/board"}
MODELS_DIR=${MODELS_DIR:-}
VENDOR_DIR=${VENDOR_DIR:-"$ROOT/../vendor"}
BOARD_TMP=/tmp/yolo_face_install.$$
BUILD=0
TMP=

# Libraries the application loads at run time. Missing ones are restored from
# an existing local copy or downloaded from the pinned upstream SDK. The TDL
# stack needs libcvi_tdl and its shared OpenCV 3.2 dependencies; libvpu_tdl.so
# is optional (a patched middleware build), the application falls back to the
# system libvpu.so when it is absent.
LIBS="libcviruntime.so libcvikernel.so libvenc.so libsys.so libisp.so
libisp_algo.so libae.so libawb.so libsns_gc4653.so libcvi_bin.so
libcvi_bin_isp.so libatomic.so.1 libstdc++.so.6 libgcc_s.so.1 libcvi_tdl.so libcvimath.so libcnpy.so libcvi_ive.so
libopencv_core.so.3.2 libopencv_imgproc.so.3.2 libopencv_imgcodecs.so.3.2
libvpu_tdl.so libz.so.1"

# Files published by Sophgo in cvitek-tdl-sdk-sg200x. The OpenCV names differ:
# the upstream symlinks point at .so.3.2.0, while the board expects .so.3.2.
SDK_COMMIT=${SDK_COMMIT:-02dfb9091d390a7b0d1177fe7b651db1b99854bc}
SDK_RAW=${SDK_RAW:-"https://raw.githubusercontent.com/milkv-duo/cvitek-tdl-sdk-sg200x/$SDK_COMMIT"}

# The only face model Sophgo publishes is the SCRFD detector; it is the same
# 768x432 network used by face_stream. Recognition and liveness models are not
# in the SDK and cannot be downloaded.
STOCK_FD_MODEL=scrfd_768_432_int8_1x.cvimodel
FD_MODEL=scrfd_det_face_432_768_INT8_cv181x.cvimodel
FACE_MODELS="$FD_MODEL feature_cviface_112_112_INT8_cv181x.cvimodel
cls_rgbliveness_256_256_INT8_cv181x.cvimodel"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
say() { printf '\n>>> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }

cleanup() {
    [ -n "$TMP" ] && [ -d "$TMP" ] && rm -rf "$TMP"
    return 0
}
trap cleanup EXIT INT TERM

ensure_tmp() {
    [ -n "$TMP" ] || TMP=$(mktemp -d "${TMPDIR:-/tmp}/yolo_face_install.XXXXXX")
}

fetch() { # fetch <url> <dest>
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$2" "$1"
    else
        die "curl or wget is required to download the release package"
    fi
}

verify_sha256() { # verify_sha256 <dir> <file.sha256>
    if command -v sha256sum >/dev/null 2>&1; then
        (cd "$1" && sha256sum -c "$2")
    elif command -v shasum >/dev/null 2>&1; then
        (cd "$1" && shasum -a 256 -c "$2")
    else
        warn "sha256sum/shasum not found; checksum not verified"
    fi
}

upstream_lib_path() { # upstream_lib_path <soname>
    case "$1" in
        libcvi_tdl.so)          printf '%s\n' 'lib/libcvi_tdl.so' ;;
        libcvikernel.so|libcvimath.so|libcnpy.so|libcviruntime.so|libz.so.1)
            printf 'sample/3rd/tpu/lib/%s\n' "$1" ;;
        libopencv_core.so.3.2)
            printf '%s\n' 'sample/3rd/opencv/lib/libopencv_core.so.3.2.0' ;;
        libopencv_imgproc.so.3.2)
            printf '%s\n' 'sample/3rd/opencv/lib/libopencv_imgproc.so.3.2.0' ;;
        libopencv_imgcodecs.so.3.2)
            printf '%s\n' 'sample/3rd/opencv/lib/libopencv_imgcodecs.so.3.2.0' ;;
        libcvi_ive.so|libvenc.so|libsys.so|libisp.so|libisp_algo.so|libae.so|\
        libawb.so|libsns_gc4653.so|libcvi_bin.so|libcvi_bin_isp.so)
            printf 'sample/3rd/middleware/v2/lib/%s\n' "$1" ;;
        *) return 1 ;;
    esac
}

stage_lib() { # stage_lib <soname>: copy a local vendor library into $TMP/libs
    for d in "$PAYLOAD_DIR/runtime" "$VENDOR_DIR/lib" \
             "$ROOT/dist/board/runtime" "$ROOT/runtime"; do
        if [ -f "$d/$1" ]; then
            cp -L "$d/$1" "$TMP/libs/$1"
            return 0
        fi
    done
    return 1
}

restore_runtime_libs() {
    ensure_tmp
    mkdir -p "$TMP/libs"
    say "restoring runtime libraries in $DEST/runtime"
    have=$($RSSH "mkdir -p $DEST/runtime && cd $DEST/runtime && ls -1 2>/dev/null") || \
        die "cannot inspect $DEST/runtime on the board"

    wanted=
    for n in $LIBS; do
        printf '%s\n' "$have" | grep -Fxq "$n" && continue
        if stage_lib "$n"; then
            printf '  local  %s\n' "$n"
        else
            wanted="$wanted $n"
        fi
    done

    for n in $wanted; do
        path=$(upstream_lib_path "$n" || true)
        [ -n "$path" ] || continue
        printf '  web    %s\n' "$n"
        if ! fetch "$SDK_RAW/$path" "$TMP/libs/$n"; then
            rm -f "$TMP/libs/$n"
            warn "could not download $n"
        fi
    done

    if [ -n "$(ls -A "$TMP/libs" 2>/dev/null)" ]; then
        tar -C "$TMP/libs" -cf - . | \
            $RSSH "tar -xf - -C $DEST/runtime" || \
            die "cannot upload the runtime libraries"
    fi

    missing=$($RSSH "cd $DEST/runtime && for n in $LIBS; do \
        [ \"\$n\" = libvpu_tdl.so ] && continue; \
        [ -f \"\$n\" ] || printf '%s ' \"\$n\"; done") || true
    if [ -n "$missing" ]; then
        warn "libraries still missing on the board:$missing"
        warn "copy them from the firmware/SDK with scripts/fetch-vendor.sh, or set VENDOR_DIR"
    else
        echo "  all $LIBS present"
    fi
}

build_payload() {
    say "building $APP locally"
    ( cd "$ROOT" && ./scripts/fetch-deps.sh && ./scripts/build-opencv.sh && \
      ./scripts/build.sh ) || die "build failed"
    ensure_tmp
    ( cd "$ROOT/.." && scripts/package-board.sh "$PROJECT" \
        "$ROOT/build/bin/$APP" "$TMP" ) || die "packaging failed"
    mkdir -p "$TMP/payload"
    tar -xzf "$TMP/$ASSET" -C "$TMP/payload" || die "cannot unpack $ASSET"
    PAYLOAD_DIR="$TMP/payload"
}

stage_models() { # stage_models <src> <board-dir>
    ensure_tmp
    rm -rf "$TMP/models"
    mkdir -p "$TMP/models"
    find "$1" -type f \( -name '*.cvimodel' -o -name '*.names' \) \
        -exec cp -f {} "$TMP/models/" \;
    if [ -z "$(ls -A "$TMP/models" 2>/dev/null)" ]; then
        warn "no .cvimodel or .names files under $1"
        return 1
    fi
    tar -C "$TMP/models" -cf - . | $RSSH "mkdir -p $2 && tar -xf - -C $2"
}

prepare_payload() {
    if [ "$BUILD" = "1" ]; then
        build_payload
        return 0
    fi
    if [ -x "$PAYLOAD_DIR/$APP" ]; then
        say "using local package: $PAYLOAD_DIR"
        return 0
    fi
    case "$RELEASE" in
        latest) BASE="$REPO_URL/releases/latest/download" ;;
        *)      BASE="$REPO_URL/releases/download/$RELEASE" ;;
    esac
    ensure_tmp
    say "downloading $BASE/$ASSET"
    fetch "$BASE/$ASSET" "$TMP/$ASSET" || \
        die "cannot download $BASE/$ASSET (use --build, RELEASE or PAYLOAD_DIR)"
    if fetch "$BASE/$ASSET.sha256" "$TMP/$ASSET.sha256"; then
        verify_sha256 "$TMP" "$ASSET.sha256" || die "checksum verification failed"
    else
        warn "checksum file not available; skipping verification"
    fi
    mkdir -p "$TMP/payload"
    tar -xzf "$TMP/$ASSET" -C "$TMP/payload" || die "cannot unpack $ASSET"
    PAYLOAD_DIR="$TMP/payload"
    [ -x "$PAYLOAD_DIR/$APP" ] || die "package does not contain $APP"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --build)     BUILD=1; shift ;;
        -h|--help)   sed -n '2,/^set -eu/{/^set -eu/d;s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
        *)           break ;;
    esac
done

IP=${BOARD_IP:-${1:-}}
if [ -z "$IP" ]; then
    printf 'Board IP (or host name): '
    read -r IP || true
fi
[ -n "$IP" ] || die "board address is empty"
case "$IP" in
    *[!A-Za-z0-9.:_-]*) die "unexpected characters in board address: $IP" ;;
esac

RSSH="ssh $SSH_OPTS $BOARD_USER@$IP"

prepare_payload

say "checking $BOARD_USER@$IP"
$RSSH true || die "cannot connect over SSH"

say "stopping old stream/face instances"
$RSSH '
for s in /root/yolo_face/start_face.sh /root/start_face.sh \
         /root/yolo_stream/start_stream.sh /root/start_stream.sh; do
    if [ -x "$s" ]; then "$s" stop >/dev/null 2>&1 || true; fi
done
' || true

say "copying files to $DEST"
# cp -a merges into the existing tree: runtime/, models/ and the face gallery
# are preserved. Only files that the package actually contains are replaced.
$RSSH "rm -rf $BOARD_TMP && mkdir -p $BOARD_TMP $DEST/log"
tar -C "$PAYLOAD_DIR" -cf - . | $RSSH "tar -xf - -C $BOARD_TMP"
$RSSH "
    mkdir -p $DEST
    cp -a $BOARD_TMP/. $DEST/
    rm -rf $BOARD_TMP
    chmod +x $DEST/$APP $DEST/start_face.sh $DEST/run_face.sh
"

restore_runtime_libs

MODEL_SRC=
if [ -n "$MODELS_DIR" ] && [ -d "$MODELS_DIR" ]; then
    MODEL_SRC=$MODELS_DIR
elif [ -n "$(find "$VENDOR_DIR/models" -type f \
        \( -name '*.cvimodel' -o -name '*.names' \) 2>/dev/null | head -n 1)" ]; then
    MODEL_SRC=$VENDOR_DIR/models
elif [ -n "$(find "$PAYLOAD_DIR/models" -type f \
        \( -name '*.cvimodel' -o -name '*.names' \) 2>/dev/null | head -n 1)" ]; then
    MODEL_SRC=$PAYLOAD_DIR/models
elif [ -n "$(find "$ROOT/dist/board/models" -type f \
        \( -name '*.cvimodel' -o -name '*.names' \) 2>/dev/null | head -n 1)" ]; then
    MODEL_SRC=$ROOT/dist/board/models
fi

if [ -n "$MODEL_SRC" ]; then
    say "uploading models from $MODEL_SRC"
    stage_models "$MODEL_SRC" "$DEST/models/face" || true
    $RSSH "ls -1 $DEST/models/face | head -20"
else
    say "models"
    $RSSH "ls -1 $DEST/models/face 2>/dev/null | head -20" || true
fi

say "checking face models on the board"
$RSSH '
set -e
DEST='$DEST'
mkdir -p "$DEST/models/face"
for n in '$FACE_MODELS'; do
    [ -f "$DEST/models/face/$n" ] && continue
    for d in /mnt/system/usr/bin/ai/models \
             /mnt/system/usr/bin/ai/models/face \
             /root/models; do
        if [ -f "$d/$n" ]; then
            cp -L "$d/$n" "$DEST/models/face/$n"
            echo "  $n (from $d)"
            break
        fi
    done
done
' || warn "could not check the stock face models on the board"

# Sophgo publishes the SCRFD detector in the TDL SDK. It is the same 768x432
# network used by face_stream, so install it under its upstream name when no
# detector is present. The recognition and liveness models are not published.
if ! $RSSH "[ -f $DEST/models/face/$FD_MODEL ] || \
             [ -f $DEST/models/face/$STOCK_FD_MODEL ]"; then
    say "downloading the stock face detector"
    ensure_tmp
    rm -rf "$TMP/face"
    mkdir -p "$TMP/face"
    if fetch "$SDK_RAW/cvimodel/$STOCK_FD_MODEL" \
            "$TMP/face/$STOCK_FD_MODEL"; then
        tar -C "$TMP/face" -cf - . | \
            $RSSH "mkdir -p $DEST/models/face && tar -xf - -C $DEST/models/face" || \
            warn "cannot upload the stock face detector"
        echo "  $STOCK_FD_MODEL"
    else
        warn "could not download $STOCK_FD_MODEL"
    fi
fi

missing=$($RSSH "cd $DEST/models/face && \
    { [ -f $FD_MODEL ] || [ -f $STOCK_FD_MODEL ]; } || printf '%s ' $FD_MODEL; \
    [ -f feature_cviface_112_112_INT8_cv181x.cvimodel ] || \
        printf '%s ' feature_cviface_112_112_INT8_cv181x.cvimodel; \
    [ -f cls_rgbliveness_256_256_INT8_cv181x.cvimodel ] || \
        printf '%s ' cls_rgbliveness_256_256_INT8_cv181x.cvimodel") || true
if [ -n "$missing" ]; then
    warn "face models still missing on the board:$missing"
    cat <<EOF

The recognition (feature_cviface) and liveness (cls_rgbliveness) models are
not published by Sophgo, so they cannot be downloaded automatically. Copy them
from your board/SDK, or install your own set:

  MODELS_DIR=~/models/face ./install.sh $IP

The directory must contain $FD_MODEL,
feature_cviface_112_112_INT8_cv181x.cvimodel and
cls_rgbliveness_256_256_INT8_cv181x.cvimodel.
EOF
fi

if [ -f "$PAYLOAD_DIR/md5sums.txt" ]; then
    say "verifying checksums"
    $RSSH "cd $DEST && md5sum -c md5sums.txt --quiet" || \
        warn "the package checksums do not match on the board"
fi


say "camera configuration"
$RSSH '
if [ -f /mnt/data/sensor_cfg.ini ]; then
    echo "sensor_cfg.ini found:"
    grep -E "^(I2C|i2c|sns_i2c|lane|mipi)" /mnt/data/sensor_cfg.ini 2>/dev/null | head -10 || true
else
    echo "warning: /mnt/data/sensor_cfg.ini not found; configure the sensor first"
fi
'

cat <<EOF

Installed to $DEST on $BOARD_USER@$IP.

  ./start_face.sh start       # stops stream_yolo (shared camera and port 7777)
  ./start_face.sh status
  ./start_face.sh stop

Run over SSH:
  ssh $BOARD_USER@$IP '$DEST/start_face.sh start'
  http://$IP:7777/            # web UI

The application uses the system musl loader and libc of the firmware;
run_face.sh puts the copied vendor libraries in front of the firmware ones.
The gallery stays in /root/faces. No autostart entry was added.
EOF
