# LicheeRV Nano: камера, NPU и распознавание лиц

Рабочий набор для LicheeRV Nano (SG2002 / CV181x) с камерой GC4653:
MJPEG-поток, детекция YOLO на NPU, распознавание лиц, утилита для съёмки и
обучения моделей. В репозитории четыре самостоятельных проекта; готовые
бинари и runtime-библиотеки уже включены, кросс-компиляция для запуска на
плате не нужна.

| Проект | Что это | Быстрый старт |
|---|---|---|
| [`yolo_stream/`](yolo_stream/README.md) | MJPEG 960×540 + YOLOv8/YOLO11 на NPU, 7 моделей, порт 7777 | `yolo_stream/install.sh <IP>` → `start_stream.sh auto` |
| [`yolo_face/`](yolo_face/README.md) | MJPEG + детекция/распознавание лиц (SCRFD + cviface) | `yolo_face/install.sh <IP>` → `start_face.sh start` |
| [`yolo_studio/`](yolo_studio/README.md) | Настольная утилита: съёмка датасета, разметка, обучение, экспорт `.cvimodel` | `pip install -r yolo_studio/requirements.txt` → `python yolo_studio/app.py` |
| [`tpu-fix/`](tpu-fix/README.md) | Патчер таймаута TPU (`soph_tpu.ko`) без распространения `.ko` | `tpu-fix/patch_tpu.sh` на плате |

`yolo_stream` и `yolo_face` используют одну камеру и порт 7777, поэтому
взаимно останавливают друг друга. Автозапуск не устанавливается: после
загрузки режим выбирается вручную.

## Установка на плату

Адрес платы не фиксирован (USB/RNDIS, Wi-Fi и Ethernet дают разные адреса).
Установщики спрашивают IP, копируют пакет в `/root/yolo_stream` или
`/root/yolo_face`, проверяют контрольные суммы и не трогают камеру:

```sh
cd yolo_stream && ./install.sh              # спросит IP
# или
BOARD_IP=<IP> ./install.sh
```

Камера настраивается штатным образом по `/mnt/data/sensor_cfg.ini`
(I2C-адрес матрицы и `lane_id` MIPI-пар). Скрипты не меняют pinmux и не
пишут в `/dev/mem`.

- Поток: `http://<IP>:7777/` (VLC/ffplay)
- Лог потока: `/root/yolo_stream/log/stream.log`
- Лог лиц: `/root/yolo_face/log/face.log`

## Сборка

Каждый проект собирается своими скриптами; общие зависимости (RISC-V musl
toolchain, SDK, OpenCV) скачиваются в локальный `.deps/`:

```sh
cd yolo_stream
./scripts/fetch-deps.sh
./scripts/build-opencv.sh
./scripts/build.sh
```

Аналогично для `yolo_face`. `scripts/fetch-deps.sh` по умолчанию скачивает
[musl-cross 20261001](https://github.com/musl-cross/musl-cross/releases/tag/20261001)
(`riscv64-unknown-linux-musl`, SHA256 зафиксирован); поддерживаются также
`musl.cc` `riscv64-linux-musl-cross` (GCC 11.2.1) и Sipeed/Sophgo. SDK —
[milkv-duo/cvitek-tdl-sdk-sg200x](https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x),
OpenCV — официальный [opencv-mobile v36](https://github.com/nihui/opencv-mobile/releases/tag/v36)
(4.13.0, Apache-2.0) с патчем CVI-бэкенда. Точные версии и коммиты
зафиксированы в `scripts/fetch-deps.sh`.

### Сборка на GitHub

Репозиторий уже содержит workflow `.github/workflows/build.yml`. Он собирает
`stream_yolo` и `face_stream` на Ubuntu runner при каждом push, а также
запускается вручную:

1. открой вкладку **Actions** → **build**;
2. нажми **Run workflow** → **Run workflow**;
3. после завершения скачай артефакты `yolo_stream-bin` и `yolo_face-bin`
   (внутри — готовые бинарники `build/bin/*`).

Кросс-компиляция и OpenCV выполняются на GitHub, локальный toolchain не
нужен. Артефакты Actions хранятся ограниченное время; для долгосрочной
публикации прикрепи бинарники к GitHub Release.

## Что важно знать

- **Датасеты и кадры пользователя в репозиторий не входят.** Модели — только
  готовые `.cvimodel`; изображения, видео и разметка остаются локально.
- **TPU-фикс — патчер, а не GPL-бинарник.** `tpu-fix/patch_tpu.sh` правит
  таймаут в установленном на плате модуле, делает резервную копию и умеет
  откатываться. Исходников драйвера и `.ko` в репозитории нет.
- **Лицензии сторонних моделей и библиотек** — в
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), тексты — в `LICENSES/`.
  Часть вендорных библиотек Cvitek/Sophgo идёт без явной лицензии; модели
  YOLO — производные Ultralytics (AGPL-3.0). Перед коммерческим
  использованием прочитайте notices.
- **Автозапуск не ставится.** Плата после загрузки свободна; поток или лица
  запускаются вручную.

## Лицензия

Оригинальный код репозитория распространяется по MIT — см. [`LICENSE`](LICENSE).
Сторонние библиотеки, runtime и модели остаются под своими лицензиями; их
условия и источники перечислены в [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
и `LICENSES/`. MIT не заменяет лицензии сторонних компонентов.
