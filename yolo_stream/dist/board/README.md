# Плата: установка, автозапуск, управление

Состав: `stream_yolo` (бинарь), `start_stream.sh` (меню моделей + управление),
`s.sh` (MIPI-клок, pinmux, USB-сеть), `models/` (7 моделей YOLOv8n/YOLO11n-det),
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

`face_stream` (комплект `lichee-face-stream`) использует ту же камеру и порт
7777: `start_stream.sh` сам останавливает его перед стартом, а
`start_face.sh start` сам останавливает `stream_yolo`.

- Поток: `http://192.0.2.1:7777/` (MJPEG)
- Лог: `/root/log/stream.log` (при старте прошлый уезжает в `.1`)
- Ручки: `SY_DET_THR` (порог), `SY_HP="W H"` (разрешение), `SY_MODEL` (модель для
  `auto`), `SY_MAX_BOXES` (анти-всплеск: не публиковать результат, где рамок
  больше N; по умолчанию 16, 0 — выключить), `SY_NAMES_FILE` (sidecar с
  именами классов), остальные — в `main.cpp`.

## Модели

Положить файл `*.cvimodel` в `/root/models` — он появится в меню. Подходят
любые YOLOv8/YOLO11-det экспорты (голова 3 масштаба x 64 DFL + N классов,
int8 или fp32): число классов и размер входа бинарь определяет по тензорам
сам. Модели других голов (SCRFD, позы, классификация и т.п.) бинарь отвергнет
с понятным сообщением в логе. Штатные модели уже в `models/`, включая
детектор квадрокоптеров `yolov11n_det_drone_640_640` (YOLO11n, 1 класс;
пайплайн сборки — `scripts/model-drone/`). Это **v2**: дообучена на кадрах
конкретного дрона (в руке, на столе, на полке вдали) + жёстких негативах
(шнурок худи, сцена без дрона), поэтому уверенно ловит его и вблизи, и вдали
(INT8 ~0.84–0.90), а на сцене без дрона молчит. Рядом лежит
`yolov11n_det_drone_v1_generic.cvimodel.bak` — прежняя «общая» модель
(обучена на полётных кадрах kc34251): хуже на этом дроне (вблизи ~0.3,
вдали ~0.0), но лучше обобщает на другие. Вернуться к ней:
`cp /root/models/yolov11n_det_drone_v1_generic.cvimodel.bak /root/models/yolov11n_det_drone_640_640_INT8_cv181x.cvimodel && /root/start_stream.sh drone`.

Имена классов в `.cvimodel` не хранятся. Рядом можно положить
`<имя_модели>.names` (по одному имени в строке) — тогда рамки подписываются
именами, иначе остаётся `class:N`. yolo_studio создаёт и заливает этот файл
вместе с моделью.

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
