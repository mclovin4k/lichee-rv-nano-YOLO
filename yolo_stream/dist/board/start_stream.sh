#!/bin/sh
# start_stream.sh -- MJPEG + NPU detection on LicheeRV Nano.
#
#   ./start_stream.sh                interactive model menu
#   ./start_stream.sh auto           default model, no menu
#   ./start_stream.sh <substring>    specific model (file name or substring)
#   ./start_stream.sh stop|restart|status|log
#
# Optional environment:
#   SY_MODEL=<substring>   model for "auto" (default coco80)
#   SY_HP="W H"            capture size (default "960 540")
#   SY_BUNDLED_LOADER=1    run through runtime/ld-musl-riscv64.so.1
#   SY_KEEP_LOG=1          do not rotate the log on start
#   plus the detection variables understood by stream_yolo (SY_DET_THR, ...)
#
# stream_yolo and face_stream share the camera and port 7777: starting one
# stops the other.  No autostart is installed.  Camera geometry comes from
# /mnt/data/sensor_cfg.ini.
set -eu

SELF=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP=${APP:-"$SELF/stream_yolo"}
LOG=${LOG:-"$SELF/log/stream.log"}
MODEL_DIRS="$SELF/models /root/models /mnt/system/usr/bin/ai/models"
ORDER_FILE=${ORDER_FILE:-/root/models/models.order}
APP_PID=${APP_PID:-/tmp/yolo_stream.app.pid}

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
    find_proc_pid stream_yolo
}

face_pid() {
    find_proc_pid face_stream
}

stop_app() {
    P=$(app_pid) || P=
    if [ -z "$P" ]; then
        rm -f "$APP_PID"
        echo "stopped"
        return 0
    fi
    kill -INT "$P" 2>/dev/null || true
    n=0
    while [ "$n" -lt 25 ] && pid_alive "$P"; do
        sleep 1
        n=$((n + 1))
    done
    if pid_alive "$P"; then
        kill -9 "$P" 2>/dev/null || true
        # A thread stuck in a TPU ioctl ignores SIGKILL until the driver
        # timeout expires (~60 s on the stock module, ~1 s with tpu-fix).
        n=0
        while [ "$n" -lt 70 ] && pid_alive "$P"; do
            sleep 1
            n=$((n + 1))
        done
    fi
    rm -f "$APP_PID"
    if pid_alive "$P"; then
        echo "stream_yolo did not exit (pid $P)"
        return 1
    fi
    echo "stopped"
}

stop_face() {
    P=$(face_pid) || P=
    [ -n "$P" ] || return 0
    echo "stopping face_stream (shared camera and port 7777)..."
    # Stop supervisors first so they do not restart the app we are killing.
    for f in /tmp/yolo_face.pid /tmp/face_supervisor.pid; do
        [ -f "$f" ] && kill -INT "$(cat "$f" 2>/dev/null || true)" 2>/dev/null || true
    done
    for ctl in "$SELF/../yolo_face/start_face.sh" \
               /root/yolo_face/start_face.sh /root/start_face.sh; do
        if [ -x "$ctl" ]; then
            "$ctl" stop >/dev/null 2>&1 || true
            break
        fi
    done
    n=0
    while [ "$n" -lt 20 ] && pid_alive "$P"; do
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
    rm -f /tmp/yolo_face.app.pid
    if pid_alive "$P"; then
        echo "could not stop face_stream"
        return 1
    fi
    return 0
}

# --- model list (duplicates from different directories are shown once)
list_models() {
    seen=""
    for d in $MODEL_DIRS; do
        [ -d "$d" ] || continue
        for f in "$d"/*.cvimodel; do
            [ -f "$f" ] || continue
            b=${f##*/}
            case "$seen" in *"|$b|"*) continue ;; esac
            seen="$seen|$b|"
            echo "$b"
        done
    done
    if [ -f "$ORDER_FILE" ]; then
        while IFS= read -r b; do
            [ -n "$b" ] || continue
            case "$seen" in *"|$b|"*) continue ;; esac
            for d in $MODEL_DIRS; do
                if [ -f "$d/$b" ]; then
                    seen="$seen|$b|"
                    echo "$b"
                    break
                fi
            done
        done < "$ORDER_FILE"
    fi
}

find_model() { # $1: substring (empty = first available)
    for d in $MODEL_DIRS; do
        [ -d "$d" ] || continue
        for f in "$d"/*.cvimodel; do
            [ -f "$f" ] || continue
            case "${f##*/}" in *"$1"*) echo "$f"; return 0 ;; esac
        done
    done
    return 1
}

kind() {
    case "$1" in
    *coco80*) echo "COCO-80, 640x640" ;;
    *person_vehicle*) echo "человек/транспорт (7), 640x384" ;;
    *head_hardhat*) echo "голова/каска (2), 960x576" ;;
    *traffic_light*) echo "светофор (5), 640x384" ;;
    *license_plate*) echo "номер (1), 640x384" ;;
    *fire_smoke*) echo "огонь/дым (2), 640x384" ;;
    *drone*) echo "квадрокоптер (1), 640x640" ;;
    *det*) echo "YOLO-det (классы и размер авто)" ;;
    *) echo "не YOLO-det: бинарь откажется с сообщением" ;;
    esac
}

start_app() { # $1: model path
    MODEL=$1
    [ -x "$APP" ] || { echo "no executable: $APP"; exit 1; }
    [ -x "$SELF/run_stream.sh" ] || { echo "no run_stream.sh"; exit 1; }

    stop_app >/dev/null || exit 1
    stop_face || exit 1

    mkdir -p "$SELF/log" "$(dirname "$LOG")"
    if [ "${SY_KEEP_LOG:-0}" != "1" ] && [ -f "$LOG" ]; then
        mv -f "$LOG" "$LOG.1"
    fi
    : > "$LOG"
    printf '%s\n' "$MODEL" > "$SELF/log/last_model"

    HP=${SY_HP:-"960 540"}
    # shellcheck disable=SC2086
    set -- $HP
    W=${1:-960}
    H=${2:-540}

    SY_WEDGE_MS=${SY_WEDGE_MS:-25000} \
        setsid "$SELF/run_stream.sh" "$MODEL" auto "$W" "$H" \
        >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$APP_PID"

    sleep 3
    P=$(app_pid) || P=
    if [ -n "$P" ]; then
        echo "started: ${MODEL##*/}"
        echo "  MJPEG http://<IP>:${SY_HTTP_PORT:-7777}/ , log $LOG (pid $P)"
        return 0
    fi
    echo "did not start; see $LOG"
    tail -n 20 "$LOG" 2>/dev/null || true
    exit 1
}

run_path() {
    start_app "$1"
}

case "${1:-}" in
stop)
    stop_app
    exit $?
    ;;
status)
    P=$(app_pid) || P=
    if [ -n "$P" ]; then
        echo "stream_yolo: running (pid $P)"
        grep "^\[stream\]" "$LOG" 2>/dev/null | tail -1 || true
        echo "MJPEG: http://<IP>:${SY_HTTP_PORT:-7777}/"
    else
        echo "stream_yolo: not running (log: $LOG)"
    fi
    exit 0
    ;;
log)
    tail -f "$LOG"
    ;;
restart)
    M=$(cat "$SELF/log/last_model" 2>/dev/null || true)
    [ -n "$M" ] && [ -f "$M" ] || M=$(find_model "${SY_MODEL:-coco80}") || M=$(find_model "")
    [ -n "$M" ] || { echo "no models in: $MODEL_DIRS"; exit 1; }
    run_path "$M"
    ;;
auto)
    M=$(find_model "${SY_MODEL:-coco80}") || M=$(find_model "")
    [ -n "$M" ] || { echo "no models in: $MODEL_DIRS"; exit 1; }
    run_path "$M"
    ;;
"")
    echo "=== models (*.cvimodel) ==="
    i=0
    for b in $(list_models); do
        i=$((i + 1))
        echo "  $i) $b  --  $(kind "$b")"
    done
    if [ "$i" = "0" ]; then echo "no models in: $MODEL_DIRS"; exit 1; fi
    echo "  0) exit"
    printf "Choice: "
    read -r sel
    case "$sel" in
    "" | *[!0-9]*) echo "a number is required"; exit 1 ;;
    esac
    [ "$sel" = "0" ] && exit 0
    if [ "$sel" -lt 1 ] || [ "$sel" -gt "$i" ]; then echo "no such item"; exit 1; fi
    b=$(list_models | sed -n "${sel}p")
    M=$(find_model "${b%.cvimodel}") || { echo "model not found"; exit 1; }
    run_path "$M"
    ;;
*)
    M=$(find_model "$1") || { echo "model '$1' not found in: $MODEL_DIRS"; exit 1; }
    run_path "$M"
    ;;
esac
