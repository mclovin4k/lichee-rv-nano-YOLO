#!/bin/sh
# run_face.sh -- one face_stream run (environment setup + exec).
#
# Used by start_face.sh, but can be run by hand:
#   ./run_face.sh [width] [height]
#
# The application is a musl binary and uses the loader and the C library of the
# board firmware. Vendor libraries are searched in runtime/ first (copied from
# the board by install.sh), then in the firmware directories.
set -eu

SELF=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP=${APP:-"$SELF/face_stream"}
RUNTIME=${RUNTIME:-"$SELF/runtime"}
MODEL_DIR=${MODEL_DIR:-"$SELF/models/face"}
FACES_DIR=${FACES_DIR:-/root/faces}
APP_PIDFILE=${FACE_APP_PIDFILE:-/tmp/yolo_face.app.pid}

die() { printf 'run_face: %s\n' "$*" >&2; exit 1; }

[ -x "$APP" ] || die "no executable: $APP"
[ -d "$RUNTIME" ] || printf 'run_face: warning: no runtime directory %s\n' \
    "$RUNTIME" >&2

find_model() { # find_model <primary> <fallback...>
    for name in "$@"; do
        if [ -n "$name" ] && [ -f "$name" ]; then
            printf '%s\n' "$name"
            return 0
        fi
    done
    return 1
}

FD_MODEL=${SY_FD_MODEL:-}
if [ -z "$FD_MODEL" ]; then
    FD_MODEL=$(find_model \
        "$MODEL_DIR/scrfd_det_face_432_768_INT8_cv181x.cvimodel" \
        "$MODEL_DIR/scrfd_768_432_int8_1x.cvimodel") || \
        die "face detector model not found in $MODEL_DIR"
fi
FR_MODEL=${SY_FR_MODEL:-}
if [ -z "$FR_MODEL" ]; then
    FR_MODEL=$(find_model \
        "$MODEL_DIR/feature_cviface_112_112_INT8_cv181x.cvimodel") || \
        die "face recognition model not found in $MODEL_DIR"
fi
LIVE_MODEL=${SY_LIVE_MODEL:-}
if [ -z "$LIVE_MODEL" ]; then
    LIVE_MODEL=$(find_model \
        "$MODEL_DIR/cls_rgbliveness_256_256_INT8_cv181x.cvimodel") || \
        LIVE_MODEL=
fi

export LD_LIBRARY_PATH="$RUNTIME:/mnt/system/usr/lib:/mnt/system/usr/lib/3rd:/mnt/system/lib:/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -f "$RUNTIME/libcvi_tdl.so" ] && \
    export SY_TDL_LIB=${SY_TDL_LIB:-"$RUNTIME/libcvi_tdl.so"}
# Optional: a patched middleware build for TDL. Without it the application
# falls back to the system libvpu.so.
[ -f "$RUNTIME/libvpu_tdl.so" ] && \
    export SY_VPU_LIB=${SY_VPU_LIB:-"$RUNTIME/libvpu_tdl.so"}
export SY_FD_MODEL="$FD_MODEL"
export SY_FR_MODEL="$FR_MODEL"
export SY_LIVE_MODEL="$LIVE_MODEL"
export SY_FACES_DIR="$FACES_DIR"
export SY_LIVE=${SY_LIVE:-0}
export SY_ROTATE=${SY_ROTATE:-0}
export SY_ROT_CW=${SY_ROT_CW:-1}

mkdir -p "$FACES_DIR"
printf '%s\n' "$$" > "$APP_PIDFILE"

# A bundled loader is only used when someone put one into runtime/ on purpose.
if [ "${SY_BUNDLED_LOADER:-0}" != "0" ] && \
        [ -x "$RUNTIME/ld-musl-riscv64.so.1" ]; then
    exec "$RUNTIME/ld-musl-riscv64.so.1" "$APP" "$@"
fi

exec "$APP" "$@"
