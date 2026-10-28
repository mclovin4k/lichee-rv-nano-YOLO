# Плата: stream_yolo

Состав: `stream_yolo` (бинарь), `run_stream.sh` (окружение + loader),
`start_stream.sh` (меню моделей и управление), `runtime/` (библиотеки и
musl loader), `models/` (7 моделей YOLOv8n/YOLO11n-det).

## Установка

Скопировать всё в `/root/yolo_stream` (установщик `../../install.sh` делает
это сам), затем:

```sh
/root/yolo_stream/start_stream.sh auto
```

Камера настраивается штатным образом по `/mnt/data/sensor_cfg.ini`.
Автозапуск не ставится.

## Управление

```sh
/root/yolo_stream/start_stream.sh             # меню: выбрать модель и запустить
/root/yolo_stream/start_stream.sh auto        # модель по умолчанию (coco80)
/root/yolo_stream/start_stream.sh <подстрока> # конкретная модель, напр. fire
/root/yolo_stream/start_stream.sh status
/root/yolo_stream/start_stream.sh stop
/root/yolo_stream/start_stream.sh restart
```

- Поток: `http://<IP>:7777/` (MJPEG)
- Лог: `/root/yolo_stream/log/stream.log` (при старте прошлый уезжает в `.1`)
- Модели: `models/` рядом со скриптом, затем `/root/models`
- PID: `/tmp/yolo_stream.app.pid`

`stream_yolo` и `face_stream` из соседнего проекта `yolo_face` используют
одну камеру и порт 7777: `start_stream.sh` сам останавливает `face_stream`,
а `start_face.sh start` сам останавливает `stream_yolo`.

## Runtime

`runtime/` содержит middleware-библиотеки камеры/ISP/VENC/VPSS,
`libvenc`/`libsys`, `libstdc++`/`libgcc`/`libatomic` и musl loader.
`libcviruntime.so`/`libcvikernel.so` и системные `libvpu.so`/`libz.so`
берутся с платы (`/mnt/system/lib`, `/mnt/system/usr/lib`).

Для запуска через встроенный загрузчик:

```sh
SY_BUNDLED_LOADER=1 /root/yolo_stream/start_stream.sh restart
```

## Модели

Положить файл `*.cvimodel` в `models/` или `/root/models` — он появится в
меню. Подходят любые YOLOv8/YOLO11-det экспорты (голова 3 масштаба × 64 DFL
+ N классов, int8 или fp32): число классов и размер входа бинарь определяет
по тензорам сам. Модели других голов (SCRFD, позы, классификация и т.п.)
бинарь отвергнет с понятным сообщением в логе.

Дрон-модель `yolov11n_det_drone_640_640` — одна общая версия (YOLO11n,
1 класс). Она производна от Ultralytics YOLO11n (AGPL-3.0); лицензии и
источники — в `../../../THIRD_PARTY_NOTICES.md`.

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
(рамки пропадают на секунды, поток не рвётся; с TPU-патчем клин ограничен
~1 с); шумные строки `isp_clut_ctrl_postprocess` в логе безвредны. Если
поток всё же встал намертво — `/root/yolo_stream/start_stream.sh restart`.

## Загрузка платы

```
rcS -> штатные S* скрипты образа, в т.ч. настройка MIPI по /mnt/data/sensor_cfg.ini
    -> sshd/usb0 по настройкам образа
```

Поток и `face_stream` автозапуском не поднимаются: после загрузки выберите
режим вручную (`start_stream.sh` или `yolo_face/start_face.sh start`).
