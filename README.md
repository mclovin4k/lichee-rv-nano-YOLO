# lichee — рабочие материалы по LicheeRV Nano (SG2002/CV181x)

- `yolo_stream/` — финальное приложение: MIPI CSI (GC4653) → YOLOv8 на NPU →
  MJPEG 30 fps для VLC/ffplay, с автозапуском и супервизором на плате.
  Подробности — в `yolo_stream/README.md`.
- `capture_test/` — ранние эксперименты с захватом (можно игнорировать).
- `scripts/` — вспомогательные скрипты сборочной машины.

Ключевые внешние зависимости (вне этого репозитория):

- opencv-mobile 4.13 с CVI-бэкендом — исходники в `~/Документы/opencv-0xjba/`,
  изменения лежат патчем в `yolo_stream/patches/`.
- тулчейн riscv64 и SDK — на сборочной машине (`/tmp/opencode/riscv-tc`,
  `/tmp/opencode/cvisdk`), на плате — `/root/mixboard1`, `/root/libs_patch`.

Плата: `root@192.0.2.1` (RNDIS через USB), поток — `http://192.0.2.1:7777/`.
