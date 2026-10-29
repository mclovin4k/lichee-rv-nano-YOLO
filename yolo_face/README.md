# face_stream — MJPEG + распознавание лиц на NPU (LicheeRV Nano, SG2002/CV181x)

Готовое приложение для LicheeRV Nano с камерой GC4653: MJPEG 960×540 на
порту 7777, детекция лиц SCRFD, распознавание по галерее (cviface) и
пассивная проверка живости по морганию. В комплекте — собранный бинарь,
runtime-библиотеки, загрузчик и скрипты запуска, поэтому кросс-компиляция для
запуска не нужна.

## Состав

| Что | Где |
|---|---|
| Исходники | `src/` |
| Готовый бинарь и runtime | `dist/board/face_stream`, `dist/board/runtime/` |
| Модели TDL | `dist/board/models/face/` |
| Запуск/остановка | `dist/board/start_face.sh`, `dist/board/run_face.sh` |
| Установщик по SSH | `install.sh` |
| Сборка | `CMakeLists.txt`, `cmake/`, `scripts/`, `patches/` |
| Контрольные суммы | `dist/md5sums.txt` |

`libcvi_tdl.so` грузится через `dlopen`: она собрана с shared OpenCV 3.2, а
приложение — со статическим OpenCV 4.13 (CVI-бэкенд камеры). Прямая линковка
смешивает две реализации OpenCV и падает. Поэтому в runtime рядом лежат
`libcvi_tdl.so`, `libcnpy.so`, `libcvi_ive.so` и shared OpenCV 3.2.

Патченная middleware-сборка `libvpu_tdl.so` нужна только TDL (GDC-символы).
VPSS/VI работают с системной `libvpu.so` из `/mnt/system/usr/lib` — она
согласована с драйвером прошивки. Обе библиотеки одновременно видны в
`/proc/PID/maps`; это нормально.

## Установка на плату

```sh
chmod +x install.sh
./install.sh                 # спросит IP платы
BOARD_IP=<IP> ./install.sh
TPU_PATCH=1 ./install.sh <IP>   # ещё и фикс таймаута TPU
```

Адрес платы не фиксирован: в разных сетях (USB/RNDIS, Wi-Fi, Ethernet) он
свой. Установщик:

1. проверяет SSH;
2. останавливает старые `face_stream`/`stream_yolo`;
3. копирует пакет в `/root/yolo_face`, сохраняя `/root/faces` (галерею);
4. сверяет md5;
5. по желанию применяет `../tpu-fix/patch_tpu.sh`;
6. **не** добавляет автозапуск.

Камера настраивается штатным образом по `/mnt/data/sensor_cfg.ini` (I2C-адрес
матрицы и `lane_id` MIPI-пар). Скрипты не трогают pinmux и `/dev/mem`; если
файла нет, установщик предупредит.

### Ручная установка

```sh
tar -C dist/board -cf - . | ssh root@<IP> 'rm -rf /root/yolo_face; mkdir -p /root/yolo_face; tar -xf - -C /root/yolo_face'
ssh root@<IP> 'chmod +x /root/yolo_face/face_stream /root/yolo_face/*.sh'
```

## Запуск

```sh
ssh root@<IP> '/root/yolo_face/start_face.sh start'
ssh root@<IP> '/root/yolo_face/start_face.sh status'
ssh root@<IP> '/root/yolo_face/start_face.sh stop'
```

`start_face.sh start` сам останавливает `stream_yolo`: у них общая камера и
порт 7777. Обратно: `yolo_stream/start_stream.sh` останавливает
`face_stream`. Автозапуска нет — режим выбирается вручную после загрузки.

- Веб-морда: `http://<IP>:7777/` — видео, список людей, поле имени, «Запомнить».
- MJPEG: `http://<IP>:7777/stream`
- Лог: `/root/yolo_face/log/face.log`
- Галерея: `/root/faces/<имя>/feature.bin` (+ `meta.json`, `preview.jpg`)

Супервизор перезапускает приложение, если сторожевой таймер вернул код 2
(зависание TPU). Обычный выход по SIGINT не перезапускается.

### Загрузчик

По умолчанию бинарь запускается напрямую и использует системный
`/lib/ld-musl-riscv64.so.1`. Если в образе нет системного загрузчика или он
несовместим, `run_face.sh` берёт `runtime/ld-musl-riscv64.so.1`:

```sh
SY_BUNDLED_LOADER=1 /root/yolo_face/start_face.sh start
```

При явном запуске через загрузчик `comm` процесса становится
`ld-musl-riscv64`, поэтому `start_face.sh` хранит pid в
`/tmp/yolo_face.app.pid`, а не ищет процесс через `pidof`.

## Как это работает

1. OpenCV с CVI-бэкендом отдаёт NV21-кадр 960×540 из VPSS (без копирования
   через `cv::Mat`).
2. Отдельный канал VPSS отдаёт RGB_888 для TDL: SCRFD находит лица,
   cviface считает эмбеддинг, галерея сравнивает по косинусу.
3. Инференс идёт в отдельном потоке: зависший TPU не останавливает видео.
4. Кадр с рамками уходит в аппаратный VENC (JPEG), затем в MJPEG-сервер.
5. «Запомнить» собирает 12 кадров крупнейшего лица, усредняет
   нормализованный вектор и кладёт его в `/root/faces/<имя>/`.

Цвета рамок: жёлтая `имя?` — совпадение есть, живость ещё не подтверждена;
зелёная `имя` — подтверждена морганием; красная `photo?` — лицо не моргает;
жёлтая `Unknown` — лица нет в галерее.

Liveness-модель TDL (`cls_rgbliveness_256_256`) на этой прошивке роняет
драйвер TPU, поэтому по умолчанию выключена (`SY_LIVE=0`). Защита от
фотографии — пассивная проверка моргания: человеку ничего делать не нужно,
фотография не моргает. Подтверждение живёт 15 с и протухает без новых
морганий.

## Сборка

Нужны:

- RISC-V musl toolchain. `scripts/fetch-deps.sh` по умолчанию скачивает
  [musl-cross 20261001](https://github.com/musl-cross/musl-cross/releases/tag/20261001)
  (`riscv64-unknown-linux-musl`, SHA256 зафиксирован); поддерживаются также
  [musl.cc](https://musl.cc/) `riscv64-linux-musl-cross` (GCC 11.2.1) и
  Sipeed/Sophgo с префиксом `riscv64-unknown-linux-musl-`.
- SDK [milkv-duo/cvitek-tdl-sdk-sg200x](https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x)
  (заголовки middleware v2 и `cvi_tdl`, `libvenc.so`, `libsys.so`),
  зафиксирован коммит в `scripts/fetch-deps.sh`.
- OpenCV: официальный релиз
  [nihui/opencv-mobile v36, opencv-mobile-4.13.0.zip](https://github.com/nihui/opencv-mobile/releases/tag/v36)
  (Apache-2.0) + патч `patches/opencv-mobile-4.13.0-cvi-capture.patch`.
- `libcviruntime`/`libcvikernel` заголовки и библиотеки — `third_party/cviruntime`.

```sh
cd yolo_face
./scripts/fetch-deps.sh       # тулчейн, SDK, исходники OpenCV + патч
./scripts/build-opencv.sh     # сборка/установка OpenCV в .deps/opencv-install
./scripts/build.sh            # build/bin/face_stream
```

Или указать готовые пути:

```sh
TOOLCHAIN_PATH=/path/to/riscv64-linux-musl-cross \
CVISDK_PATH=/path/to/cvitek-tdl-sdk-sg200x \
OpenCV_DIR=/path/to/opencv/lib/cmake/opencv4 \
./scripts/build.sh
```

`TOOLCHAIN_PATH` читается и из окружения, и через `-DTOOLCHAIN_PATH`; для
`try_compile` надёжнее экспортировать его. `scripts/fetch-deps.sh` качает
только нужные файлы SDK (sparse checkout), без примеров и бинарей.

Ту же сборку можно запустить на GitHub: вкладка **Actions** → **build** →
**Run workflow**. Готовый `face_stream` будет в артефакте `yolo_face-bin`.

## Модели и лицензии

Модели `scrfd_det_face_432_768`, `feature_cviface_112_112`,
`cls_rgbliveness_256_256` взяты из примеров TDL SDK. Файла лицензии у
`milkv-duo/cvitek-tdl-sdk-sg200x` нет; распространяются как есть, без
гарантий. `libcvi_tdl.so`, `libcnpy.so`, `libcvi_ive.so`, `libcviruntime.so`,
`libcvikernel.so`, `libcvimath.so` и `libvpu_tdl.so` — вендорные бинарники
Cvitek/Sophgo без явной лицензии в источнике. Подробности и ссылки —
в `../THIRD_PARTY_NOTICES.md`; это не «лицензированный» код, а готовая
поставка для платы.

OpenCV 4.13 (static, для приложения) распространяется по Apache-2.0,
OpenCV 3.2 (shared, для TDL) — по BSD-3-Clause. Патч камеры применяется к
официальному релизу `nihui/opencv-mobile` v36.

## Ручки окружения

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `SY_DET_THR` | 0.40 | порог детекции SCRFD |
| `SY_FACE_THR` | 0.40 | порог сходства с галереей |
| `SY_FACE_PERIOD_MS` | 70 | пауза между инференсами |
| `SY_MAX_BOXES` | 8 | больше рамок — сбой детекции, кадр пропускается |
| `SY_PUB_BOXES` | 5 | сколько лиц показывать |
| `SY_PUB_SCORE` | 0.55 | минимальный score для показа |
| `SY_PUB_MIN` | 48 | минимальная сторона рамки, px |
| `SY_LIVE` | 0 | 1 — включить liveness-модель (не рекомендуется) |
| `SY_LIVE_BLINKS` | 1 | сколько морганий подтверждают живость |
| `SY_LIVE_CHECK_MS` | 12000 | сколько ждать моргания |
| `SY_LIVE_KEEP_MS` | 15000 | сколько живёт подтверждение |
| `SY_ROTATE` | 0 | поворот кадра для детектора: 0/90/180/270 |
| `SY_FACES_DIR` | `/root/faces` | каталог галереи |
| `SY_TDL_LIB` | `runtime/libcvi_tdl.so` | библиотека TDL |
| `SY_VPU_LIB` | `runtime/libvpu_tdl.so` | libvpu для TDL |
| `SY_BUNDLED_LOADER` | 0 | 1 — запуск через встроенный musl loader |
| `WIDTH`, `HEIGHT` | 960, 540 | размер кадра |

## Ограничения

- TDL принимает кадр 960×540; поворот 90/270 даёт 540×960 и
  `ILLEGAL_PARAM`. Для повёрнутого корпуса используйте `SY_ROTATE` только
  если TDL это принимает; по умолчанию 0.
- Liveness-модель отключена (см. выше).
- Зависание TPU (`wait dmabuf failed[-110]`, `run cmdbuf failed:2`) —
  платформенный дефект драйвера. С фиксом из `../tpu-fix/` ожидание
  сокращается с ~60 с до ~1 с; приложение пересоздаёт handle TDL.
- Закрытие камеры иногда зависает в драйвере (`vb_qbuf`, состояние `D`);
  если процесс не убивается — нужен `reboot`.
- MJPEG q75 при 30 fps — около 9 Мбит/с.
