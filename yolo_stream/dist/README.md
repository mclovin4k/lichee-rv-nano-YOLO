# Package: MJPEG stream + YOLOv8 NPU detection (LicheeRV Nano, SG2002)

Hardware: LicheeRV Nano with the GC4653 camera. 960×540 @30 fps MJPEG stream,
YOLOv8n/YOLO11n detection on the NPU (~70 ms per inference). The release
package contains everything that is needed on the board: the binary, runtime
libraries, matching musl loader/libc, models and start scripts.

| Item | Location |
|---|---|
| Board files: binary, runtime, models, start menu | `board/README.md` |
| Release package | `yolo_stream-board.tar.gz` on GitHub Releases |
| SSH installer (downloads the release) | `../install.sh` |
| TPU hang fix (patcher, no `.ko`) | `../../tpu-fix/` |
| Sources and build | `../` |

Quick start: `../install.sh <IP>` downloads the latest release, verifies its
SHA256, unpacks it into `/root/yolo_stream` and optionally applies the TPU
patch. `PAYLOAD_DIR` selects a locally built package; `RELEASE=v1.0.0` pins a
release. No autostart is installed; the stock image configures the camera
itself via `/mnt/data/sensor_cfg.ini` (sensor I2C address and MIPI `lane_id`).

- Stream: `http://<IP>:7777/`
- Control: `/root/yolo_stream/start_stream.sh` (model menu)
- Log: `/root/yolo_stream/log/stream.log`
- Drone detector: `board/models/yolov11n_det_drone_640_640...cvimodel` —
  a single common YOLO11n model (1 class, md5
  `d479ed5849723b9137d71fd18cc3d4ba`).
