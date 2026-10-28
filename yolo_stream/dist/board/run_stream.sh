#!/bin/sh
# run_stream.sh -- one stream_yolo run (environment setup + exec).
#
# Used by start_stream.sh, but can be run by hand:
#   ./run_stream.sh /path/to/model.cvimodel auto 960 540
#
# The system musl loader is used by default.  SY_BUNDLED_LOADER=1 forces the
# loader from runtime/ (fallback for unusual images).
set -eu

SELF=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP=${APP:-"$SELF/stream_yolo"}
RUNTIME=${RUNTIME:-"$SELF/runtime"}
APP_PIDFILE=${SY_STREAM_APP_PIDFILE:-/tmp/yolo_stream.app.pid}

die() { printf 'run_stream: %s\n' "$*" >&2; exit 1; }

[ -x "$APP" ] || die "no executable: $APP"
[ -d "$RUNTIME" ] || die "no runtime directory: $RUNTIME"

export LD_LIBRARY_PATH="$RUNTIME:/mnt/system/usr/lib:/mnt/system/usr/lib/3rd:/mnt/system/lib:/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf '%s\n' "$$" > "$APP_PIDFILE"

if [ "${SY_BUNDLED_LOADER:-0}" = "1" ] || [ ! -x /lib/ld-musl-riscv64.so.1 ]; then
    [ -x "$RUNTIME/ld-musl-riscv64.so.1" ] || \
        die "no system loader and no bundled loader"
    exec "$RUNTIME/ld-musl-riscv64.so.1" "$APP" "$@"
fi

exec "$APP" "$@"
