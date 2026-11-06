# Package: MJPEG stream + YOLO NPU detection (LicheeRV Nano, SG2002)

Hardware: LicheeRV Nano with the GC4653 camera. 960×540 @30 fps MJPEG stream,
YOLOv8n/YOLO11n detection on the NPU (~70 ms per inference).

The release package contains **only this project's files**: the binary, the
start/run scripts, an empty `models/` directory and `md5sums.txt`. It does not
contain vendor libraries or models — the Cvitek/Sophgo runtime is proprietary
and the models are either AGPL-3.0 derivatives of Ultralytics or SDK files
without a license. `install.sh` keeps the board copies, restores missing
libraries from local files or the pinned upstream SDK, and uploads the models
you provide. See `../../THIRD_PARTY_NOTICES.md`.

| Item | Location |
|---|---|
| Board files: binary, scripts, empty models/ | `board/README.md` |
| Release package | `yolo_stream-board.tar.gz` on GitHub Releases |
| SSH installer (downloads the release) | `../install.sh` |
| Sources and build | `../` |

Quick start: `../install.sh <IP>` downloads the latest release, verifies its
SHA256, unpacks it into `/root/yolo_stream`, keeps the existing runtime and
model files, restores missing vendor libraries and uploads `MODELS_DIR` (or
`../vendor/models`) if set. Then `/root/yolo_stream/start_stream.sh auto`.
`PAYLOAD_DIR` selects a locally built package; `RELEASE=v1.0.1` pins a release.
No autostart is installed; the stock image configures the camera itself via
`/mnt/data/sensor_cfg.ini` (sensor I2C address and MIPI `lane_id`).

- Stream: `http://<IP>:7777/`
- Control: `/root/yolo_stream/start_stream.sh` (model menu)
- Log: `/root/yolo_stream/log/stream.log`
- Models: `models/` next to the script, `/root/models`, then
  `/mnt/system/usr/bin/ai/models` (stock image)

There are 7 stock YOLOv8n/YOLO11n detection models on the board image, but the
binary is not tied to them: any YOLOv8/YOLO11 detection export with the six
outputs produced by `yolo_studio` works, int8 or fp32. The number of classes
and the input size are read from the tensors.
