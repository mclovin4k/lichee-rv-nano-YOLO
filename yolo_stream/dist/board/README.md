# Плата: установка, автозапуск, управление

Состав: `stream_yolo` (бинарь), `start_stream.sh` (меню моделей + управление),
`s.sh` (MIPI-клок, pinmux, USB-сеть), `models/` (6 моделей YOLOv8n-det),
`libs_patch/` (middleware-библиотеки SDK, на них отлажен бинарь),
`S99z-cam-init.sh` / `S99z-stream.sh` / `S99z-swap.sh` (init-скрипты).

## Установка на чистый образ

```sh
# 1. файлы на плату
scp -r board/* root@192.0.2.1:/root/
# 2. автозапуск
cp /root/S99z-cam-init.sh /root/S99z-stream.sh /root/S99z-swap.sh /etc/init.d/
chmod +x /etc/init.d/S99z-*.sh /root/s.sh /root/start_stream.sh
# 3. swap (по желанию; иначе S99z-swap молча пропустит)
dd if=/dev/zero of=/swapfile bs=1M count=1024 && mkswap /swapfile
# 4. фикс TPU (см. ../tpu-fix/README.md), затем
reboot
```

`board/libs_patch/` копируется вместе со всем остальным в `/root/libs_patch`
(отладочная связка middleware SDK). Если её не копировать, `start_stream.sh`
возьмёт библиотеки из системных каталогов — вариант `mixboard1` не нужен,
`libvpu.so` есть на любом образе в `/mnt/system/usr/lib`.

## Управление

```sh
/root/start_stream.sh             # меню: выбрать модель из /root/models и запустить
/root/start_stream.sh auto        # модель по умолчанию (coco80), без меню
/root/start_stream.sh <подстрока> # конкретная модель, напр. fire
/root/start_stream.sh status      # pid + последние строки fps
/root/start_stream.sh stop
```

- Поток: `http://192.0.2.1:7777/` (MJPEG)
- Лог: `/root/log/stream.log` (при старте прошлый уезжает в `.1`)
- Ручки: `SY_DET_THR` (порог), `SY_HP="W H"` (разрешение), `SY_MODEL` (модель для
  `auto`), остальные — в `main.cpp`.

## Модели

Положить файл `*.cvimodel` в `/root/models` — он появится в меню. Подходят
любые YOLOv8/YOLO11-det экспорты (голова 3 масштаба x 64 DFL + N классов,
int8 или fp32): число классов и размер входа бинарь определяет по тензорам
сам. Модели других голов (SCRFD, позы, классификация и т.п.) бинарь отвергнет
с понятным сообщением в логе. Штатные 6 моделей уже в `models/`.

## Что ожидать (норма)

- `[stream] frames=... fps=30.0` — поток держит 30 к/с.
- `yolo8: auto-detect: <N> classes` — классы прочитаны из модели.
- `[stream] ... det=60..90ms ... boxes=N det_age=0.0s` — детекция ~12-15 Гц,
  рамки свежие.

Известные мелочи: клин TPU (`run dmabuf timeout`) приложение переживает само
(рамки пропадают на секунды, поток не рвётся; с фиксом TPU клин ограничен ~1 с);
шумные строки `isp_clut_ctrl_postprocess` в логе — безвредны. Если поток всё же
встал намертво — `/root/start_stream.sh restart`.

## Как это работает при загрузке

```
rcS -> S99z-cam-init.sh   (source /root/s.sh: MIPI 0x03001174/0x0300118C, usb0 192.0.2.1)
    -> S99z-swap.sh       (swapon /swapfile, если есть)
    -> S99z-stream.sh     (start_stream.sh auto -> stream_yolo)
```
