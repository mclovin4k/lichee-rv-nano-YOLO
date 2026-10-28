# Плата: face_stream

Состав: `face_stream`, `run_face.sh`, `start_face.sh`, `runtime/`,
`models/face/`.

## Установка

Скопировать всё в `/root/yolo_face` (установщик `../install.sh` делает это
сам), затем:

```sh
/root/yolo_face/start_face.sh start
```

Камера настраивается штатным образом по `/mnt/data/sensor_cfg.ini`.
Автозапуск не ставится.

## Управление

```sh
/root/yolo_face/start_face.sh start|stop|restart|status|log
```

- UI: `http://<IP>:7777/`
- MJPEG: `http://<IP>:7777/stream`
- Лог: `/root/yolo_face/log/face.log`
- Галерея: `/root/faces/<имя>/`
- PID: `/tmp/yolo_face.pid` (супервизор), `/tmp/yolo_face.app.pid`

`start_face.sh start` останавливает `stream_yolo` (общая камера и порт).
Обратно `yolo_stream/start_stream.sh` останавливает `face_stream`.

## Runtime

`runtime/` содержит `libcvi_tdl.so`, её зависимости (`libcnpy`, `libcvi_ive`,
shared OpenCV 3.2, `libz`), middleware-библиотеки камеры/ISP/VENC,
`libvpu_tdl.so` для TDL, `libstdc++`/`libgcc`/`libatomic` и musl loader.
Системная `libvpu.so` берётся с платы (`/mnt/system/usr/lib`).

Для запуска через встроенный загрузчик:

```sh
SY_BUNDLED_LOADER=1 /root/yolo_face/start_face.sh restart
```
