#!/bin/sh
# start_stream.sh -- запуск/перезапуск потока stream_yolo (MJPEG + NPU-детекция).
#
#   ./start_stream.sh                интерактивное меню выбора модели
#   ./start_stream.sh auto           без меню, модель по умолчанию (автозапуск)
#   ./start_stream.sh <часть имени>  запуск конкретной модели (файл или подстрока)
#   ./start_stream.sh stop|restart|status|log
#
# Ручки (необязательно): SY_MODEL=<подстрока>  SY_HP="W H" (по умолчанию "960 540")
#   SY_DET_THR, SY_KEEP_LOG=1, SY_DUMP_NV21=<файл> и другие из main.cpp.
#
# Модель -- любой YOLOv8-det cvimodel: число классов и размер входа бинарь
# определяет по тензорам сам (аргумент "auto"). Модели ищутся в models/ рядом
# со скриптом, затем в /root/models и /mnt/system/usr/bin/ai/models.

SELF_DIR=$(cd "$(dirname "$0")" && pwd)
APP=$SELF_DIR/stream_yolo
LOG=$SELF_DIR/log/stream.log
MODEL_DIRS="$SELF_DIR/models /root/models /mnt/system/usr/bin/ai/models"
# Модели из этого файла показываются в конце меню (в порядке добавления).
ORDER_FILE=${ORDER_FILE:-/root/models/models.order}
# face_stream использует ту же камеру и порт 7777 — перед стартом останавливаем.
FACE_CTL=${FACE_CTL:-/root/start_face.sh}
LDP=/root/mixboard1:/root/libs_patch/middleware_v2:/root/libs_patch/middleware_v2_3rd:/mnt/system/usr/lib:/mnt/system/usr/lib/3rd:/mnt/system/lib:/usr/lib

# --- остановка: SIGINT, через 25 с SIGKILL; залипший в ioctl тред умирает ~60 с
stop_app() {
    P=$(pidof stream_yolo)
    [ -n "$P" ] || return 0
    kill -INT $P 2>/dev/null
    n=0
    while [ $n -lt 25 ] && pidof stream_yolo > /dev/null; do sleep 1; n=$((n + 1)); done
    pidof stream_yolo > /dev/null || return 0
    kill -9 $(pidof stream_yolo) 2>/dev/null
    n=0
    while [ $n -lt 65 ] && pidof stream_yolo > /dev/null; do sleep 1; n=$((n + 1)); done
    pidof stream_yolo > /dev/null && return 1
    return 0
}

# pidof находит и зомби, а они не держат камеру — берём только живые.
face_pid() {
    for p in $(pidof face_stream 2>/dev/null); do
        st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null)
        [ "$st" = "Z" ] && continue
        echo "$p"
        return 0
    done
    return 1
}

# face_stream занимает ту же камеру и порт 7777: перед стартом убираем его.
# У face_stream есть супервизор, поэтому pidof может показать только зомби,
# а живой супервизор перезапустит поток — stop вызываем и по pid-файлу.
stop_face() {
    if [ -x "$FACE_CTL" ]; then
        if [ -n "$(pidof face_stream 2>/dev/null)" ] || [ -f /tmp/face_supervisor.pid ]; then
            echo "останавливаю face_stream (общая камера и порт 7777)..."
        fi
        # stop_face.sh сам разбирается с супервизором, зомби и pid-файлом.
        "$FACE_CTL" stop >/dev/null 2>&1
    elif [ -n "$(face_pid)" ]; then
        echo "останавливаю face_stream (общая камера и порт 7777)..."
        kill -INT $(face_pid) 2>/dev/null
    else
        return 0
    fi
    n=0
    while [ $n -lt 30 ] && [ -n "$(pidof face_stream 2>/dev/null)" ]; do
        sleep 1; n=$((n + 1))
    done
    if [ -n "$(face_pid)" ]; then
        kill -9 $(face_pid) 2>/dev/null
        sleep 1
    fi
    if [ -n "$(face_pid)" ]; then
        echo "не удалось остановить face_stream"
        return 1
    fi
    return 0
}

# --- список моделей (дубликаты по имени из разных каталогов не повторяются)
# Модели из ORDER_FILE печатаются в конце — так новая модель, залитая утилитой,
# оказывается последней в меню.
list_models() {
    seen=""
    for d in $MODEL_DIRS; do
        [ -d "$d" ] || continue
        for f in "$d"/*.cvimodel; do
            [ -f "$f" ] || continue
            b=${f##*/}
            if [ -f "$ORDER_FILE" ] && grep -qxF "$b" "$ORDER_FILE"; then
                continue
            fi
            echo "$seen" | grep -q "|$b|" && continue
            seen="$seen|$b|"
            echo "$b"
        done
    done
    if [ -f "$ORDER_FILE" ]; then
        while IFS= read -r b; do
            [ -n "$b" ] || continue
            echo "$seen" | grep -q "|$b|" && continue
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

find_model() { # $1: подстрока имени (пустая = первая доступная)
    for d in $MODEL_DIRS; do
        [ -d "$d" ] || continue
        for f in "$d"/*.cvimodel; do
            [ -f "$f" ] || continue
            case "${f##*/}" in *"$1"*) echo "$f"; return 0 ;; esac
        done
    done
    return 1
}

kind() { # короткая подпись модели для меню
    case "$1" in
    *coco80*) echo "COCO-80, 640x640" ;;
    *person_vehicle*) echo "человек/транспорт (7), 640x384" ;;
    *head_hardhat*) echo "голова/каска (2), 960x576" ;;
    *traffic_light*) echo "светофор (5), 640x384" ;;
    *license_plate*) echo "номер (1), 640x384" ;;
    *fire_smoke*) echo "огонь/дым (2), 640x384" ;;
    *drone*) echo "квадрокоптер (1), 640x640" ;;
    *det*) echo "YOLOv8-det (классы и размер авто)" ;;
    *) echo "не YOLOv8-det: бинарь откажется с сообщением" ;;
    esac
}

start_app() { # $1: полный путь к модели
    stop_face || return 1
    mkdir -p "$SELF_DIR/log"
    if [ "${SY_KEEP_LOG:-0}" != "1" ] && [ -f "$LOG" ]; then mv -f "$LOG" "$LOG.1"; fi
    cd "$SELF_DIR" || return 1
    SY_WEDGE_MS=${SY_WEDGE_MS:-25000} LD_LIBRARY_PATH=$LDP \
        setsid "$APP" "$1" auto ${SY_HP:-960 540} >> "$LOG" 2>&1 < /dev/null &
    sleep 2
    pidof stream_yolo > /dev/null
}

run_path() { # $1: полный путь к модели
    echo "остановка прошлого экземпляра..."
    stop_app || { echo "не смог остановить stream_yolo"; exit 1; }
    if start_app "$1"; then
        echo "запущено: ${1##*/}"
        echo "поток MJPEG: http://192.0.2.1:${SY_HTTP_PORT:-7777}/   лог: $LOG"
    else
        echo "старт не удался, смотри $LOG"
        exit 1
    fi
}

case "$1" in
stop)
    stop_app
    exit $?
    ;;
status)
    P=$(pidof stream_yolo)
    if [ -n "$P" ]; then
        echo "stream_yolo: работает (pid $P)"
        grep "MJPEG on" "$LOG" 2>/dev/null | tail -1
        grep "^\[stream\]" "$LOG" 2>/dev/null | tail -1
        echo "поток MJPEG: http://192.0.2.1:7777/"
    else
        echo "stream_yolo: не запущен (лог: $LOG)"
    fi
    exit 0
    ;;
log)
    tail -f "$LOG"
    ;;
restart | auto)
    M=$(find_model "${SY_MODEL:-coco80}") || M=$(find_model "")
    [ -n "$M" ] || { echo "нет моделей в: $MODEL_DIRS"; exit 1; }
    run_path "$M"
    ;;
"")
    echo "=== Модели (*.cvimodel) ==="
    i=0
    for b in $(list_models); do
        i=$((i + 1))
        echo "  $i) $b  --  $(kind "$b")"
    done
    if [ "$i" = "0" ]; then echo "нет моделей в: $MODEL_DIRS"; exit 1; fi
    echo "  0) выход"
    printf "Выбор: "
    read sel
    case "$sel" in
    "" | *[!0-9]*) echo "нужен номер пункта"; exit 1 ;;
    esac
    [ "$sel" = "0" ] && exit 0
    if [ "$sel" -lt 1 ] || [ "$sel" -gt "$i" ]; then echo "нет такого пункта"; exit 1; fi
    b=$(list_models | sed -n "${sel}p")
    M=$(find_model "${b%.cvimodel}") || { echo "модель не найдена"; exit 1; }
    run_path "$M"
    ;;
*)
    M=$(find_model "$1") || { echo "модель '$1' не найдена в: $MODEL_DIRS"; exit 1; }
    run_path "$M"
    ;;
esac
