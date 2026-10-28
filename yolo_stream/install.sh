#!/bin/sh
# install.sh -- copy the ready-to-run stream_yolo package to a LicheeRV Nano.
#
#   ./install.sh                 # asks for the board IP
#   BOARD_IP=<IP> ./install.sh
#   ./install.sh <IP>
#   TPU_PATCH=1 ./install.sh <IP>    # also patch soph_tpu.ko
#   ./install.sh --tpu-patch <IP>    # same, explicit flag
#
# The board address is not fixed: USB/RNDIS, Wi-Fi and Ethernet networks use
# different addresses.  Nothing is started automatically; the script only
# copies files and (optionally) applies the TPU timeout fix.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BOARD_USER=${BOARD_USER:-root}
SSH_OPTS=${SSH_OPTS:-"-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

TPU_PATCH=${TPU_PATCH:-0}
if [ "${1:-}" = "--tpu-patch" ]; then
    TPU_PATCH=1
    shift
fi
IP=${BOARD_IP:-${1:-}}
if [ -z "$IP" ]; then
    printf 'Board IP (or host name): '
    read -r IP
fi
[ -n "$IP" ] || die "board address is empty"
case "$IP" in
    *[!A-Za-z0-9.:_-]*) die "unexpected characters in board address: $IP" ;;
esac

RSSH="ssh $SSH_OPTS $BOARD_USER@$IP"
DEST=/root/yolo_stream
BOARD_TMP=/tmp/yolo_stream_install.$$

say() { printf '\n>>> %s\n' "$*"; }

[ -d "$ROOT/dist/board" ] || die "dist/board is missing"
[ -x "$ROOT/dist/board/stream_yolo" ] || die "dist/board/stream_yolo is missing"

say "checking $BOARD_USER@$IP"
$RSSH true || die "cannot connect over SSH"

say "stopping old stream/face instances"
$RSSH '
for s in /root/yolo_stream/start_stream.sh /root/start_stream.sh \
         /root/yolo_face/start_face.sh /root/start_face.sh; do
    [ -x "$s" ] && "$s" stop >/dev/null 2>&1 || true
done
true
' || true

say "copying files to $DEST"
$RSSH "rm -rf $BOARD_TMP && mkdir -p $BOARD_TMP $DEST/log $DEST/models"
tar -C "$ROOT/dist/board" -cf - . | $RSSH "tar -xf - -C $BOARD_TMP"
$RSSH "
    rm -rf $DEST/runtime &&
    mkdir -p $DEST &&
    cp -a $BOARD_TMP/. $DEST/ &&
    rm -rf $BOARD_TMP &&
    chmod +x $DEST/stream_yolo $DEST/start_stream.sh $DEST/run_stream.sh
"

if [ -f "$ROOT/dist/md5sums.txt" ]; then
    say "verifying checksums"
    tar -C "$ROOT/dist" -cf - md5sums.txt | $RSSH "tar -xf - -C $DEST"
    $RSSH "cd $DEST && md5sum -c md5sums.txt" || \
        die "checksum verification failed on the board"
fi

if [ "$TPU_PATCH" = "1" ]; then
    say "applying TPU timeout fix"
    [ -f "$ROOT/../tpu-fix/patch_tpu.sh" ] || die "tpu-fix/patch_tpu.sh not found"
    tar -C "$ROOT/../tpu-fix" -cf - patch_tpu.sh README.md | \
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

Existing models in $DEST/models are preserved.  No autostart entry was added.
EOF
