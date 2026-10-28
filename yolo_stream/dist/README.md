# Поставка: поток MJPEG + NPU-детекция YOLOv8 (LicheeRV Nano, SG2002)

Аппарат: LicheeRV Nano с камерой GC4653. Поток MJPEG 960×540 @30 fps,
детекция YOLOv8n/YOLO11n на NPU (~70 мс/инференс). Всё уже собрано:
бинарь, runtime-библиотеки, загрузчик, модели и скрипты запуска — кросс-
компиляция для запуска не нужна.

| Что | Где |
|---|---|
| Плата: бинарь, runtime, модели, меню запуска | `board/README.md` |
| Установщик по SSH | `../install.sh` |
| Фикс залипания TPU (патчер, без `.ko`) | `../../tpu-fix/` |
| Исходники и сборка | `../` |
| Контрольные суммы | `md5sums.txt` |

Быстро: `../install.sh <IP>` разложит пакет в `/root/yolo_stream` и по
желанию применит TPU-патч. Автозапуск не ставится; камеру штатный образ
настраивает сам по `/mnt/data/sensor_cfg.ini` (адрес матрицы на I2C и
`lane_id` MIPI-пар).

- Поток: `http://<IP>:7777/`
- Управление: `/root/yolo_stream/start_stream.sh` (меню моделей)
- Лог: `/root/yolo_stream/log/stream.log`
- Детектор дронов: `board/models/yolov11n_det_drone_640_640...cvimodel` —
  одна общая YOLO11n-модель (1 класс, md5 `d479ed5849723b9137d71fd18cc3d4ba`).
