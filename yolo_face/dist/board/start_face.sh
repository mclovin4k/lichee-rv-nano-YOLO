#!/bin/sh
# start_face.sh [start|stop|restart|status|log] -- MJPEG + face recognition.
#
#   /root/yolo_face/start_face.sh start     # start (default)
#   /root/yolo_face/start_face.sh stop
#   /root/yolo_face/start_face.sh restart
#   /root/yolo_face/start_face.sh status
#
# stream_yolo and face_stream share the camera and port 7777, so starting one
# stops the other.  No autostart is installed.  Camera geometry (I2C address,
# MIPI lane_id) comes from /mnt/data/sensor_cfg.ini; this script does not touch
# pinmux or /dev/mem.
#
# Optional environment:
#   WIDTH, HEIGHT         capture size (default 960x540)
#   SY_LIVE=1             enable the liveness model (off by default)
#   SY_ROTATE=90          rotate the frame for the detector
#   FACES_DIR=/root/faces gallery: <name>/feature.bin
#   SY_BUNDLED_LOADER=1   use runtime/ld-musl-riscv64.so.1 (default: system)
#   SY_KEEP_LOG=1         do not rotate the log on start
set -eu

SELF=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP=${APP:-"$SELF/face_stream"}
MODEL_DIR=${MODEL_DIR:-"$SELF/models/face"}
FACES_DIR=${FACES_DIR:-/root/faces}
LOG=${LOG:-"$SELF/log/face.log"}
WIDTH=${WIDTH:-960}
HEIGHT=${HEIGHT:-540}

SUP_PID=${SUP_PID:-/tmp/yolo_face.pid}
APP_PID=${APP_PID:-/tmp/yolo_face.app.pid}
LEGACY_SUP_PID=/tmp/face_supervisor.pid

pid_alive() {
    [ -n "${1:-}" ] || return 1
    [ -r "/proc/$1/stat" ] || return 1
    [ "$(awk '{print $3}' "/proc/$1/stat" 2>/dev/null)" != "Z" ]
}

find_proc_pid() { # find_proc_pid <basename>
    for d in /proc/[0-9]*; do
        [ -r "$d/cmdline" ] || continue
        p=${d#/proc/}
        pid_alive "$p" || continue
        tr '\0' '\n' < "$d/cmdline" 2>/dev/null | while IFS= read -r arg; do
            if [ "${arg##*/}" = "$1" ]; then
                echo "$p"
                break
            fi
        done | head -n 1 | grep . && return 0
    done
    return 1
}

app_pid() {
    if pid_alive "$(cat "$APP_PID" 2>/dev/null || true)"; then
        cat "$APP_PID"
        return 0
    fi
    find_proc_pid face_stream
}

sup_pid() {
    for f in "$SUP_PID" "$LEGACY_SUP_PID"; do
        [ -f "$f" ] || continue
        p=$(cat "$f" 2>/dev/null || true)
        if pid_alive "$p"; then
            echo "$p"
            return 0
        fi
    done
    return 1
}

stop_other_stream() {
    P=$(find_proc_pid stream_yolo) || return 0
    echo "stopping stream_yolo (shared camera and port 7777)..."
    for ctl in "$SELF/../yolo_stream/start_stream.sh" \
               /root/yolo_stream/start_stream.sh /root/start_stream.sh; do
        if [ -x "$ctl" ]; then
            "$ctl" stop >/dev/null 2>&1 || true
            break
        fi
    done
    n=0
    while [ "$n" -lt 30 ] && pid_alive "$P"; do
        sleep 1
        n=$((n + 1))
    done
    if pid_alive "$P"; then
        kill -INT "$P" 2>/dev/null || true
        n=0
        while [ "$n" -lt 10 ] && pid_alive "$P"; do
            sleep 1
            n=$((n + 1))
        done
    fi
    if pid_alive "$P"; then
        kill -9 "$P" 2>/dev/null || true
        sleep 1
    fi
    if pid_alive "$P"; then
        echo "could not stop stream_yolo"
        return 1
    fi
    return 0
}

stop_app() {
    S=$(sup_pid) || S=
    [ -z "$S" ] || kill -INT "$S" 2>/dev/null || true

    P=$(app_pid) || P=
    if [ -z "$P" ]; then
        rm -f "$SUP_PID" "$APP_PID" "$LEGACY_SUP_PID"
        echo "stopped"
        return 0
    fi

    kill -INT "$P" 2>/dev/null || true
    n=0
    while [ "$n" -lt 15 ] && pid_alive "$P"; do
        sleep 1
        n=$((n + 1))
    done
    if pid_alive "$P"; then
        kill -9 "$P" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$SUP_PID" "$APP_PID" "$LEGACY_SUP_PID"
    if pid_alive "$P"; then
        echo "face_stream did not exit (pid $P)"
        return 1
    fi
    echo "stopped"
}

start_app() {
    [ -x "$APP" ] || { echo "no executable: $APP"; exit 1; }
    [ -x "$SELF/run_face.sh" ] || { echo "no run_face.sh"; exit 1; }
    P=$(app_pid) || P=
    if [ -n "$P" ]; then
        echo "already running (pid $P)"
        return 0
    fi
    stop_other_stream || exit 1

    mkdir -p "$FACES_DIR" "$(dirname "$LOG")"
    if [ "${SY_KEEP_LOG:-0}" != "1" ] && [ -f "$LOG" ]; then
        mv -f "$LOG" "$LOG.1"
    fi
    : > "$LOG"

    export FACES_DIR MODEL_DIR
    export SY_FACES_DIR="$FACES_DIR"
    export SY_LIVE=${SY_LIVE:-0}
    export SY_ROTATE=${SY_ROTATE:-0}
    export FACE_APP_PIDFILE="$APP_PID"

    rm -f "$SUP_PID" "$APP_PID"
    setsid sh -c '
        echo $$ > "$1"
        while :; do
            "$2" "$3" "$4" >> "$5" 2>&1
            rc=$?
            [ "$rc" -eq 0 ] && break
            echo "[supervisor] face_stream exited rc=$rc, restarting" >> "$5"
            sleep 1
        done
        rm -f "$1" "$6"
    ' sh "$SUP_PID" "$SELF/run_face.sh" "$WIDTH" "$HEIGHT" "$LOG" "$APP_PID" \
        </dev/null >>"$LOG" 2>&1 &
    echo $! > "$SUP_PID"

    n=0
    while [ "$n" -lt 5 ]; do
        sleep 1
        P=$(app_pid) && break
        n=$((n + 1))
    done
    if [ -n "${P:-}" ] && pid_alive "$P"; then
        echo "started: UI http://<IP>:7777/ , stream /stream"
        echo "  pid $P, log $LOG"
    else
        echo "did not start; see $LOG"
        tail -n 20 "$LOG" 2>/dev/null || true
        exit 1
    fi
}

status_app() {
    P=$(app_pid) || P=
    S=$(sup_pid) || S=
    if [ -n "$P" ]; then
        echo "face_stream: running (pid $P${S:+, supervisor $S})"
        grep -E '^\[face\]' "$LOG" 2>/dev/null | tail -n 1 || true
        echo "UI: http://<IP>:7777/"
    else
        echo "face_stream: not running (log: $LOG)"
    fi
}

case "${1:-start}" in
start) start_app ;;
stop) stop_app ;;
restart) stop_app; start_app ;;
status) status_app ;;
log) tail -f "$LOG" ;;
*)
    echo "usage: $0 [start|stop|restart|status|log]"
    exit 1
    ;;
esac
