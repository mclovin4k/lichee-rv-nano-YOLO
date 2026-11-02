#!/bin/sh
# install.sh -- install stream_yolo on a LicheeRV Nano.
#
#   ./install.sh                        # download the release, ask for the IP
#   BOARD_IP=<IP> ./install.sh
#   ./install.sh <IP>
#   MODELS_DIR=~/models/my-detector ./install.sh <IP>    # upload your models
#   ./install.sh --build <IP>           # cross-compile here instead of downloading
#   RELEASE=v1.0.1 ./install.sh <IP>    # pin a release
#   TPU_PATCH=1 ./install.sh <IP>       # also apply the TPU timeout fix
#
# The release package contains this project's code only (binary + scripts).
# The vendor libraries are proprietary and are not redistributed, so they are
# copied from the board firmware itself (they are already on the image).
# Models are uploaded from MODELS_DIR or from ../vendor/models when present;
# otherwise the models already on the board are kept.
#
# The board address is not fixed: USB/RNDIS, Wi-Fi and Ethernet differ.
# Nothing is started automatically.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT=yolo_stream
APP=stream_yolo
ASSET=yolo_stream-board.tar.gz
DEST=/root/yolo_stream
BOARD_USER=${BOARD_USER:-root}
SSH_OPTS=${SSH_OPTS:-"-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"}
RELEASE=${RELEASE:-latest}
REPO_URL=${REPO_URL:-https://github.com/mclovin4k/lichee-rv-nano-YOLO}
PAYLOAD_DIR=${PAYLOAD_DIR:-"$ROOT/dist/board"}
MODELS_DIR=${MODELS_DIR:-}
VENDOR_DIR=${VENDOR_DIR:-"$ROOT/../vendor"}
BOARD_TMP=/tmp/yolo_stream_install.$$
TPU_REF=${TPU_REF:-master}
BUILD=0
TPU_PATCH=${TPU_PATCH:-0}
TMP=

# Libraries the application loads at run time; copied from the board firmware.
LIBS="libcviruntime.so libcvikernel.so libvenc.so libsys.so libisp.so
libisp_algo.so libae.so libawb.so libsns_gc4653.so libcvi_bin.so
libcvi_bin_isp.so"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
say() { printf '\n>>> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }

cleanup() {
    [ -n "$TMP" ] && [ -d "$TMP" ] && rm -rf "$TMP"
    return 0
}
trap cleanup EXIT INT TERM

ensure_tmp() {
    [ -n "$TMP" ] || TMP=$(mktemp -d "${TMPDIR:-/tmp}/yolo_stream_install.XXXXXX")
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
        --tpu-patch) TPU_PATCH=1; shift ;;
        -h|--help)   sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
for s in /root/yolo_stream/start_stream.sh /root/start_stream.sh \
         /root/yolo_face/start_face.sh /root/start_face.sh; do
    if [ -x "$s" ]; then "$s" stop >/dev/null 2>&1 || true; fi
done
' || true

say "copying files to $DEST"
$RSSH "rm -rf $BOARD_TMP && mkdir -p $BOARD_TMP $DEST/log"
tar -C "$PAYLOAD_DIR" -cf - . | $RSSH "tar -xf - -C $BOARD_TMP"
$RSSH "
    mkdir -p $DEST
    cp -a $BOARD_TMP/. $DEST/
    rm -rf $BOARD_TMP
    chmod +x $DEST/$APP $DEST/start_stream.sh $DEST/run_stream.sh
"

say "copying vendor libraries from the board firmware"
$RSSH "sh -s" <<EOF || die "cannot copy the vendor libraries on the board"
set -e
DEST=$DEST
mkdir -p "\$DEST/runtime"
missing=
for n in $(printf '%s' "$LIBS"); do
    found=
    for d in /mnt/system/lib /mnt/system/usr/lib /mnt/system/usr/lib/3rd \
             /usr/lib /lib; do
        if [ -f "\$d/\$n" ]; then
            cp -L "\$d/\$n" "\$DEST/runtime/\$n"
            found=1
            break
        fi
    done
    [ -n "\$found" ] || missing="\$missing \$n"
done
if [ -n "\$missing" ]; then
    echo "warning: not found in the firmware:\$missing" >&2
fi
ls -1 "\$DEST/runtime" | wc -l
EOF

MODEL_SRC=
if [ -n "$MODELS_DIR" ] && [ -d "$MODELS_DIR" ]; then
    MODEL_SRC=$MODELS_DIR
elif [ -d "$VENDOR_DIR/models" ] && \
        [ -n "$(ls -A "$VENDOR_DIR/models" 2>/dev/null)" ]; then
    MODEL_SRC=$VENDOR_DIR/models
fi

if [ -n "$MODEL_SRC" ]; then
    say "uploading models from $MODEL_SRC"
    $RSSH "mkdir -p $DEST/models"
    tar -C "$MODEL_SRC" -cf - . | $RSSH "tar -xf - -C $DEST/models"
    $RSSH "ls -1 $DEST/models | head -20"
else
    say "models"
    $RSSH "ls -1 $DEST/models 2>/dev/null | head -20" || true
    printf 'no models uploaded (MODELS_DIR is not set); models already on the\n'
    printf 'board are used, and stock models live in /mnt/system/usr/bin/ai/models.\n'
fi

if [ -f "$PAYLOAD_DIR/md5sums.txt" ]; then
    say "verifying checksums"
    $RSSH "cd $DEST && md5sum -c md5sums.txt --quiet" || \
        warn "the package checksums do not match on the board"
fi

if [ "$TPU_PATCH" = "1" ]; then
    say "applying TPU timeout fix"
    ensure_tmp
    if [ -f "$ROOT/../tpu-fix/patch_tpu.sh" ]; then
        mkdir -p "$TMP/tpu-fix"
        cp "$ROOT/../tpu-fix/patch_tpu.sh" "$ROOT/../tpu-fix/README.md" \
            "$TMP/tpu-fix/"
    else
        mkdir -p "$TMP/tpu-fix"
        fetch "$REPO_URL/raw/$TPU_REF/tpu-fix/patch_tpu.sh" \
            "$TMP/tpu-fix/patch_tpu.sh" || die "cannot download patch_tpu.sh"
        fetch "$REPO_URL/raw/$TPU_REF/tpu-fix/README.md" \
            "$TMP/tpu-fix/README.md" || true
    fi
    tar -C "$TMP/tpu-fix" -cf - . | \
        $RSSH "rm -rf /root/tpu-fix && mkdir -p /root/tpu-fix && tar -xf - -C /root/tpu-fix"
    $RSSH "sh /root/tpu-fix/patch_tpu.sh"
    say "reboot is required for the TPU module fix"
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

  ./start_stream.sh auto      # stop face_stream if needed and start detection
  ./start_stream.sh status
  ./start_stream.sh stop

Run over SSH:
  ssh $BOARD_USER@$IP '$DEST/start_stream.sh auto'
  vlc http://$IP:7777/

The application uses the system musl loader and libc of the firmware;
run_stream.sh puts the copied vendor libraries in front of the firmware ones.
No autostart entry was added.
EOF
