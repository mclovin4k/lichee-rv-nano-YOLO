# Сторонние компоненты, авторы и лицензии

Этот файл описывает сторонний код, бинарники и модели, которые входят в
репозиторий или скачиваются при сборке. Полные тексты лицензий — в
`LICENSES/`. Для компонентов без явной лицензии это указано отдельно: такие
файлы распространяются «как есть», и перед публикацией/коммерческим
использованием условия нужно уточнить у правообладателя.

Дата среза: 2026-10-28.

## Сводка

| Компонент | Автор / источник | Лицензия | Где лежит |
|---|---|---|---|
| opencv-mobile 4.13.0 (v36) | nihui, команда OpenCV | Apache-2.0 | скачивается `scripts/fetch-deps.sh` |
| OpenCV 4.13 (внутри opencv-mobile) | OpenCV team | Apache-2.0 | статически в `stream_yolo`, `face_stream` |
| Патч CVI-бэкенда | производная opencv-mobile | Apache-2.0 | `yolo_stream/patches/`, `yolo_face/patches/` |
| OpenCV 3.2 (shared) | OpenCV team | BSD-3-Clause | `yolo_face/dist/board/runtime/libopencv_*.so.3.2` |
| cvitek-tdl-sdk-sg200x | milkv-duo / Sophgo | нет файла лицензии | заголовки, `libvenc.so`, `libsys.so` при сборке |
| cviruntime / cvikernel | Sophgo | нет файла лицензии | `*/third_party/cviruntime/` |
| libcvi_tdl.so | Cvitek / Sophgo TDL SDK | нет файла лицензии | `yolo_face/dist/board/runtime/` |
| libcvi_ive.so, libcnpy.so, libcvimath.so | Cvitek / Sophgo TDL SDK | нет файла лицензии | `yolo_face/dist/board/runtime/` |
| libvpu_tdl.so (патченная middleware) | Cvitek / Sophgo | нет файла лицензии | `yolo_face/dist/board/runtime/` |
| middleware (ISP/VPSS/VENC/…) | Cvitek / Sophgo | нет файла лицензии | `yolo_stream/dist/board/runtime/`, `yolo_face/.../runtime/` |
| libz 1.2.11 | Jean-loup Gailly, Mark Adler | Zlib | `yolo_face/.../runtime/libz.so.1.2.11` |
| musl loader | Rich Felker и участники musl | MIT | `*/dist/board/runtime/ld-musl-riscv64.so.1` |
| libstdc++, libgcc, libatomic | GCC / FSF | GPL-3.0 + GCC Runtime Library Exception | `*/dist/board/runtime/` |
| Модели YOLOv8n (6 шт.) | производные Ultralytics YOLOv8n | AGPL-3.0 | `yolo_stream/dist/board/models/` |
| Модель дрона YOLO11n | производная Ultralytics YOLO11n | AGPL-3.0 | `yolo_stream/dist/board/models/yolov11n_det_drone_640_640_INT8_cv181x.cvimodel` |
| SCRFD face detector | InsightFace (архитектура), Cvitek TDL SDK (веса) | код MIT; веса — без явной лицензии | `yolo_face/dist/board/models/face/` |
| cviface / liveness | Cvitek TDL SDK | нет файла лицензии | `yolo_face/dist/board/models/face/` |
| ret7020/LicheeRVNano | ret7020 | нет файла лицензии; код переписан | только ссылка в README |
| PySide6 | Qt Company / участники Qt | LGPL-3.0 / GPL-3.0 / коммерческая | зависимость `yolo_studio` (не входит в репозиторий) |
| opencv-python-headless | OpenCV team | Apache-2.0 | зависимость `yolo_studio` |
| numpy | NumPy developers | BSD-3-Clause | зависимость `yolo_studio` |
| paramiko | Jeff Forcier и участники | LGPL-2.1 | зависимость `yolo_studio` |
| PyYAML | Kirill Simonov и участники | MIT | зависимость `yolo_studio` |
| Ultralytics (ultralytics) | Ultralytics | AGPL-3.0 | внешнее окружение `ultra` |
| tpu-mlir | Sophgo | Apache-2.0 | внешнее окружение `tpu310` |
| soph_tpu.ko | Sophgo | GPL-2.0 | **не входит** в репозиторий; патчится на плате |

## OpenCV и CVI-бэкенд камеры

`opencv-mobile-4.13.0.zip` — официальный релиз
[nihui/opencv-mobile v36](https://github.com/nihui/opencv-mobile/releases/tag/v36)
(архив `opencv-mobile-4.13.0.zip`, SHA256
`0aeaea4837d19894765aae23cb82fc4b87cec07449bee4b4404a63bdda15d6ee`).
Проект распространяется по Apache-2.0; внутри — OpenCV 4.13.0 (Apache-2.0).

Патчи `patches/opencv-mobile-4.13.0-cvi-capture.patch` меняют файлы
`modules/highgui/src/capture_cvi.cpp` и `videocapture.cpp` из этого же
архива, поэтому являются производными и распространяются по Apache-2.0.
Патч добавляет сырой NV21-режим (`OPENCV_CVI_RAW`), чтение геометрии сенсора
из `/mnt/data/sensor_cfg.ini`, возврат зависшей VPSS-группы и поддержку
нужных размеров кадра.

`libopencv_core.so.3.2`, `libopencv_imgproc.so.3.2`,
`libopencv_imgcodecs.so.3.2` в runtime `yolo_face` — shared-сборка OpenCV
3.2.0. Сам OpenCV 3.2 распространяется по BSD-3-Clause (текст в
`LICENSES/BSD-3-Clause.txt`). Это предварительно собранные библиотеки из
поставки Cvitek/TDL SDK; исходники OpenCV 3.2 — на
[opencv.org](https://opencv.org/releases/). Они нужны только для
`libcvi_tdl.so`; приложение `face_stream` собрано со статическим OpenCV 4.13.

## Cvitek / Sophgo SDK и вендорные библиотеки

- [milkv-duo/cvitek-tdl-sdk-sg200x](https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x),
  коммит `02dfb9091d390a7b0d1177fe7b651db1b99854bc`. Используются заголовки
  middleware v2 и `cvi_tdl`, а также `libvenc.so`/`libsys.so`; скачиваются
  скриптом `scripts/fetch-deps.sh` (sparse checkout). **Файла лицензии в
  репозитории нет.**
- [sophgo/cviruntime](https://github.com/sophgo/cviruntime) — заголовки
  `cviruntime.h`, `cvitpu_debug.h` и библиотеки `libcviruntime.so`,
  `libcvikernel.so` (`third_party/cviruntime/`). **Файла лицензии в
  репозитории нет.**
- Бинарники Cvitek/Sophgo в `dist/board/runtime/`: `libcvi_tdl.so`,
  `libcvi_ive.so`, `libcnpy.so`, `libcvimath.so`, `libvpu_tdl.so`,
  `libisp.so`, `libisp_algo.so`, `libae.so`, `libawb.so`, `libvenc.so`,
  `libsys.so`, `libsns_gc4653.so`, `libcvi_bin.so`, `libcvi_bin_isp.so`.
  Это вендорный runtime для cv181x/SG2002; **явной лицензии в источнике
  нет**, файлы включены для запуска на соответствующей плате «как есть»,
  без гарантий. `libvpu_tdl.so` — переименованная патченная сборка
  middleware `libvpu.so` из TDL SDK (нужна TDL для GDC-символов); системная
  `libvpu.so` в поставку не входит и берётся с платы.
- `libcvi_tdl.so` — из TDL SDK, md5 `5284ac664ef5330bacbac1fdb6de70ea`.
  Она собрана с shared OpenCV 3.2 и загружается через `dlopen`, поэтому
  рядом лежат `libopencv_*.so.3.2`, `libcnpy.so`, `libcvi_ive.so`, `libz`.
- `libz.so.1.2.11` — zlib 1.2.11, лицензия Zlib
  (`LICENSES/Zlib.txt`); используется `libcvi_tdl.so`/`libcnpy.so`.

## Модели

### YOLOv8n / YOLO11n (детекция)

Шесть моделей `yolov8n_det_*` — INT8-конверсии сетей Ultralytics YOLOv8n
(COCO-80, человек/транспорт, каска, светофор, номер, огонь/дым). Модель
`yolov11n_det_drone_640_640` — производная Ultralytics YOLO11n (1 класс),
дообученная автором репозитория; в поставке одна версия, md5
`d479ed5849723b9137d71fd18cc3d4ba`.

Ultralytics YOLO распространяется по **AGPL-3.0**
(`LICENSES/AGPL-3.0.txt`). Производные веса и сконвертированные
`.cvimodel` мы также считаем AGPL-3.0. Если вы распространяете эти файлы
(в том числе в составе прошивки или закрытого продукта), нужно выполнить
условия AGPL: предоставить получателям соответствующий исходный код
(предпочтительная форма — `.pt`/ONNX и скрипты конвертации) и сохранить
уведомления. Исходные веса YOLOv8n/YOLO11n доступны у Ultralytics; пайплайн
конвертации — `yolo_studio/tools/` (`export_yolo11.py`,
`build_cvimodel.sh`). Датасеты и пользовательские кадры в репозиторий не
входят.

### Лица (SCRFD, cviface, liveness)

`scrfd_det_face_432_768_INT8_cv181x.cvimodel` — детектор лиц на архитектуре
SCRFD. Архитектура и код InsightFace распространяются по MIT
(`LICENSES/MIT.txt`); веса в этой поставке взяты из примеров Cvitek TDL SDK
(в SDK есть `cvimodel/scrfd_768_432_int8_1x.cvimodel`). Условия
распространения весов в источнике не указаны.

`feature_cviface_112_112_INT8_cv181x.cvimodel` (эмбеддинги лиц) и
`cls_rgbliveness_256_256_INT8_cv181x.cvimodel` (liveness) — модели Cvitek
TDL SDK. **Явной лицензии в источнике нет.** Они включены, потому что нужны
для работы `yolo_face`; распространяются «как есть», без гарантий. Перед
коммерческим использованием или публикацией уточните права у Cvitek/Sophgo.

## Системные библиотеки в runtime

- `ld-musl-riscv64.so.1` — musl libc, MIT (`LICENSES/MIT.txt`). Нужен как
  fallback-загрузчик (`SY_BUNDLED_LOADER=1`); по умолчанию используется
  системный `/lib/ld-musl-riscv64.so.1`.
- `libstdc++.so.6`, `libgcc_s.so.1`, `libatomic.so.1` — runtime GCC 11.2.1.
  Лицензия GPL-3.0 с GCC Runtime Library Exception
  (`LICENSES/GPL-3.0.txt`, `LICENSES/GCC-Runtime-Library-Exception.txt`).
  Exception разрешает распространять программы, слинкованные с этими
  библиотеками, на любых условиях.

## yolo_studio (Python)

`yolo_studio/requirements.txt` ставит зависимости из PyPI; в репозитории их
исходников нет:

- PySide6 — LGPL-3.0 / GPL-3.0 / коммерческая лицензия Qt
  (`LICENSES/LGPL-3.0.txt`);
- opencv-python-headless — Apache-2.0;
- numpy — BSD-3-Clause;
- paramiko — LGPL-2.1 (`LICENSES/LGPL-2.1.txt`);
- PyYAML — MIT.

Дополнительно README предлагает внешние окружения:

- Ultralytics (`ultra`) — AGPL-3.0, используется для обучения/экспорта;
- tpu-mlir (`tpu310`) — Apache-2.0, используется для сборки `.cvimodel`.

Эти пакеты не входят в репозиторий и ставятся пользователем отдельно.

## tpu-fix и GPL-модуль

`tpu-fix/patch_tpu.sh` правит несколько байтов в **уже установленном на
плате** модуле `soph_tpu.ko`. Сам модуль — часть прошивки Sophgo,
распространяется по GPL-2.0 (`LICENSES/GPL-2.0.txt`). В репозитории нет ни
`.ko`, ни исходников драйвера: патчер только меняет таймаут в бинарнике на
устройстве пользователя и делает резервную копию. Скрипт не линкуется с
модулем и не является производной драйвера.

## Базовая идея и ссылки

Проект `stream_yolo` изначально вдохновлён
[ret7020/LicheeRVNano](https://github.com/ret7020/LicheeRVNano). В этом
репозитории **нет файла лицензии**, поэтому код не копировался как есть:
текущая реализация переписана, а upstream указан как источник идеи и
полезных примеров работы с CV181x. Если вы будете переиспользовать код
upstream, сначала уточните условия у автора.

## Гарантии

Все сторонние бинарники и модели поставляются **без гарантий**, «как есть».
Их работоспособность проверена только на LicheeRV Nano (SG2002/CV181x) с
камерой GC4653 и конкретной прошивкой. Правообладатели не связаны с этим
репозиторием и не поддерживают его.
