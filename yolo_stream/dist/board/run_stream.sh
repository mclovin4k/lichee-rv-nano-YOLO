#!/bin/sh
# run_stream.sh -- one stream_yolo run (environment setup + exec).
#
# Used by start_stream.sh, but can be run by hand:
#   ./run_stream.sh /path/to/model.cvimodel auto 960 540
#
# The application is a musl binary and uses the loader and the C library of the
# board firmware. Vendor libraries are searched in runtime/ first (copied from
# the board by install.sh), then in the firmware directories.
set -eu

SELF=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP=${APP:-"$SELF/stream_yolo"}
RUNTIME=${RUNTIME:-"$SELF/runtime"}
APP_PIDFILE=${SY_STREAM_APP_PIDFILE:-/tmp/yolo_stream.app.pid}

die() { printf 'run_stream: %s\n' "$*" >&2; exit 1; }

[ -x "$APP" ] || die "no executable: $APP"
[ -d "$RUNTIME" ] || printf 'run_stream: warning: no runtime directory %s\n' \
    "$RUNTIME" >&2

export LD_LIBRARY_PATH="$RUNTIME:/mnt/system/usr/lib:/mnt/system/usr/lib/3rd:/mnt/system/lib:/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf '%s\n' "$$" > "$APP_PIDFILE"

# A bundled loader is only used when someone put one into runtime/ on purpose.
if [ "${SY_BUNDLED_LOADER:-0}" != "0" ] && \
        [ -x "$RUNTIME/ld-musl-riscv64.so.1" ]; then
    exec "$RUNTIME/ld-musl-riscv64.so.1" "$APP" "$@"
fi

exec "$APP" "$@"
