# Board files: stream_yolo

Contents of the release package: `stream_yolo` (binary), `run_stream.sh`
(environment + loader), `start_stream.sh` (model menu and control),
`runtime/` (libraries and musl loader), `models/` (7 YOLOv8n/YOLO11n-det
models). The binary and the matching `libc`/loader are produced by the GitHub
workflow and are not stored in git.

## Installation

The easiest way is the release installer from the repository root:

```sh
cd yolo_stream
./install.sh <IP>
```

It downloads `yolo_stream-board.tar.gz` from the latest GitHub Release,
verifies its checksum and copies everything to `/root/yolo_stream`. You can
also unpack the release tarball manually and copy its contents there. Then:

```sh
/root/yolo_stream/start_stream.sh auto
```

The camera is configured by the stock image through `/mnt/data/sensor_cfg.ini`.
No autostart is installed.

## Control

```sh
/root/yolo_stream/start_stream.sh             # menu: choose a model and start it
/root/yolo_stream/start_stream.sh auto        # default model (coco80)
/root/yolo_stream/start_stream.sh <substring> # a specific model, e.g. fire
/root/yolo_stream/start_stream.sh status
/root/yolo_stream/start_stream.sh stop
/root/yolo_stream/start_stream.sh restart
```

- Stream: `http://<IP>:7777/` (MJPEG)
- Log: `/root/yolo_stream/log/stream.log` (the previous one is moved to `.1`
  at startup)
- Models: `models/` next to the script, then `/root/models`
- PID: `/tmp/yolo_stream.app.pid`

`stream_yolo` and `face_stream` from the neighboring `yolo_face` project use
the same camera and port 7777: `start_stream.sh` stops `face_stream` itself,
and `start_face.sh start` stops `stream_yolo` itself.

## Runtime

`runtime/` contains the camera/ISP/VENC/VPSS middleware libraries,
`libvenc`/`libsys`, `libcviruntime`/`libcvikernel`, `libstdc++`/`libgcc`/
`libatomic` and the musl loader/libc. The only libraries still taken from the
board are the firmware-matched system `libvpu.so` and `libz.so`
(`/mnt/system/lib`, `/mnt/system/usr/lib`).

`run_stream.sh` sets `LD_LIBRARY_PATH` and uses the bundled loader
(`runtime/ld-musl-riscv64.so.1`) by default, so the package works on images
with a different or missing system musl loader. To use the system loader:

```sh
SY_BUNDLED_LOADER=0 /root/yolo_stream/start_stream.sh restart
```

## Models

Put a `*.cvimodel` file into `models/` or `/root/models` — it will appear in
the menu. Any YOLOv8/YOLO11-det exports work (3-scale head × 64 DFL + N
classes, int8 or fp32): the binary determines the number of classes and the
input size from the tensors itself. Models with other heads (SCRFD, pose,
classification, etc.) are rejected with a clear log message.

The drone model `yolov11n_det_drone_640_640` is a single common version
(YOLO11n, 1 class). It is a derivative of Ultralytics YOLO11n (AGPL-3.0);
licenses and sources are in `../../../THIRD_PARTY_NOTICES.md`.

Class names are not stored in `.cvimodel`. You can put a
`<model_name>.names` file next to it (one name per line) — then boxes are
labeled with names; otherwise `class:N` remains. yolo_studio creates and
uploads this file together with the model.

## What to expect (normal)

- `[stream] frames=... fps=30.0` — the stream holds 30 fps.
- `yolo8: auto-detect: <N> classes` — classes were read from the model.
- `[stream] ... det=60..90ms ... boxes=N det_age=0.0s` — detection at
  ~12-15 Hz, boxes are fresh.

Known quirks: the application survives a TPU wedge (`run dmabuf timeout`)
by itself (boxes disappear for a few seconds, the stream does not break; with
the TPU patch the wedge is limited to ~1 s); noisy
`isp_clut_ctrl_postprocess` lines in the log are harmless. If the stream
still gets stuck, run `/root/yolo_stream/start_stream.sh restart`.

## Board boot

```
rcS -> stock S* image scripts, including MIPI setup from /mnt/data/sensor_cfg.ini
    -> sshd/usb0 according to the image settings
```

Neither the stream nor `face_stream` is started automatically: after boot,
choose the mode manually (`start_stream.sh` or `yolo_face/start_face.sh start`).
