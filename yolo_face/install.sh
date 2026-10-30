#!/bin/sh
# install.sh -- install the ready-to-run face_stream package on a LicheeRV Nano.
#
#   ./install.sh                    # download the latest release, ask for IP
#   BOARD_IP=<IP> ./install.sh
#   ./install.sh <IP>
#   RELEASE=v1.0.0 ./install.sh <IP>
#   PAYLOAD_DIR=build/board ./install.sh <IP>   # use a locally built package
#   TPU_PATCH=1 ./install.sh <IP>              # also patch soph_tpu.ko
#   ./install.sh --tpu-patch <IP>
#
# The release package already contains the binary, models and every required
# runtime library.  If a local dist/board with a binary is present, it is used
# instead of downloading.  The board address is not fixed: USB/RNDIS, Wi-Fi
# and Ethernet use different addresses.  Nothing is started automatically.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BOARD_USER=${BOARD_USER:-root}
SSH_OPTS=${SSH_OPTS:-"-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"}
RELEASE=${RELEASE:-latest}
PROJECT=yolo_face
APP=face_stream
ASSET=yolo_face-board.tar.gz
DEST=/root/yolo_face
PAYLOAD_DIR=${PAYLOAD_DIR:-"$ROOT/dist/board"}
BOARD_TMP=/tmp/yolo_face_install.$$
TMP=
TPU_REF=${TPU_REF:-master}
REPO_URL=${REPO_URL:-https://github.com/mclovin4k/lichee-rv-nano-YOLO}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
say() { printf '\n>>> %s\n' "$*"; }

cleanup() {
    if [ -n "$TMP" ] && [ -d "$TMP" ]; then
        rm -rf "$TMP"
    fi
}
trap cleanup EXIT INT TERM

ensure_tmp() {
    if [ -z "$TMP" ]; then
        TMP=$(mktemp -d "${TMPDIR:-/tmp}/yolo_face_install.XXXXXX")
    fi
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
        printf 'warning: sha256sum/shasum not found; checksum not verified\n' >&2
    fi
}

prepare_payload() {
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
        die "cannot download $BASE/$ASSET (set RELEASE or PAYLOAD_DIR)"
    if fetch "$BASE/$ASSET.sha256" "$TMP/$ASSET.sha256"; then
        verify_sha256 "$TMP" "$ASSET.sha256" || die "checksum verification failed"
    else
        printf 'warning: checksum file not available; skipping verification\n' >&2
    fi
    mkdir -p "$TMP/payload"
    tar -xzf "$TMP/$ASSET" -C "$TMP/payload" || die "cannot unpack $ASSET"
    PAYLOAD_DIR="$TMP/payload"
    [ -x "$PAYLOAD_DIR/$APP" ] || die "package does not contain $APP"
}

TPU_PATCH=${TPU_PATCH:-0}
if [ "${1:-}" = "--tpu-patch" ]; then
    TPU_PATCH=1
    shift
fi
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

say "stopping old face/stream instances"
$RSSH '
for s in /root/yolo_face/start_face.sh /root/start_face.sh \
         /root/yolo_stream/start_stream.sh /root/start_stream.sh; do
    [ -x "$s" ] && "$s" stop >/dev/null 2>&1 || true
done
true
' || true

say "copying files to $DEST"
$RSSH "rm -rf $BOARD_TMP && mkdir -p $BOARD_TMP $DEST/log /root/faces"
tar -C "$PAYLOAD_DIR" -cf - . | $RSSH "tar -xf - -C $BOARD_TMP"
$RSSH "
    rm -rf $DEST/runtime $DEST/models &&
    mkdir -p $DEST &&
    cp -a $BOARD_TMP/. $DEST/ &&
    rm -rf $BOARD_TMP &&
    chmod +x $DEST/$APP $DEST/start_face.sh $DEST/run_face.sh
"

if [ -f "$PAYLOAD_DIR/md5sums.txt" ]; then
    say "verifying checksums"
    $RSSH "cd $DEST && md5sum -c md5sums.txt" || \
        die "checksum verification failed on the board"
fi

if [ "$TPU_PATCH" = "1" ]; then
    say "applying TPU timeout fix"
    ensure_tmp
    if [ -f "$ROOT/../tpu-fix/patch_tpu.sh" ]; then
        mkdir -p "$TMP/tpu-fix"
        cp "$ROOT/../tpu-fix/patch_tpu.sh" "$ROOT/../tpu-fix/README.md" "$TMP/tpu-fix/"
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

  ./start_face.sh start     # stop stream_yolo if needed and start face_stream
  ./start_face.sh status
  ./start_face.sh stop

Run over SSH:
  ssh $BOARD_USER@$IP '$DEST/start_face.sh start'
  vlc http://$IP:7777/

The package includes runtime/ with the musl loader and all libraries.
run_face.sh sets LD_LIBRARY_PATH and uses that loader by default
(SY_BUNDLED_LOADER=0 switches to the system loader).

The gallery is kept in /root/faces and is not overwritten by this script.
No autostart entry was added.
EOF
