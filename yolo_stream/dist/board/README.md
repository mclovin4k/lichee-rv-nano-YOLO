# Board files: stream_yolo

Contents of the release package: `stream_yolo` (binary), `run_stream.sh`
(environment setup), `start_stream.sh` (model menu and control), an empty
`models/` directory and `md5sums.txt`.

The package deliberately contains **no vendor libraries and no models**: the
Cvitek/Sophgo runtime is proprietary and models are usually AGPL-3.0
derivatives of Ultralytics, so neither is redistributed. `install.sh` copies
the libraries from the firmware of your board and uploads the models you
choose. See `../../../THIRD_PARTY_NOTICES.md`.

## Installation

The easiest way is the release installer from the repository root:

```sh
cd yolo_stream
./install.sh <IP>
```

It downloads `yolo_stream-board.tar.gz` from the latest GitHub Release,
verifies its checksum, copies the files to `/root/yolo_stream` and then

1. copies the vendor libraries from `/mnt/system/lib`,
   `/mnt/system/usr/lib` and `/mnt/system/usr/lib/3rd` into
   `/root/yolo_stream/runtime` (the firmware already has them);
2. uploads `*.cvimodel` from `MODELS_DIR` or `../vendor/models` if either is
   set and not empty;
3. verifies the package checksums on the board.

You can also unpack the release tarball manually, copy its contents to
`/root/yolo_stream` and copy the libraries as described above. Then:

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

The binary is a musl executable and uses the loader and libc of the firmware.
`run_stream.sh` builds `LD_LIBRARY_PATH` in this order:

```
/root/yolo_stream/runtime            # copies made by install.sh
/mnt/system/usr/lib
/mnt/system/usr/lib/3rd
/mnt/system/lib
/usr/lib
```

If `runtime/` is empty or missing, the firmware libraries are used directly.
That works on a stock image but not on a stripped one — in that case run
`scripts/fetch-vendor.sh --board <IP>` on your workstation and upload
`vendor/lib` to `/root/yolo_stream/runtime`.

A bundled loader is used only when you put one there on purpose:

```sh
SY_BUNDLED_LOADER=1 /root/yolo_stream/start_stream.sh restart
```

## Models

Put a `*.cvimodel` file into `models/` or `/root/models` — it will appear in
the menu. Any YOLOv8/YOLO11-det export works (3-scale head × 64 DFL + N
classes, int8 or fp32): the binary determines the number of classes and the
input size from the tensors itself. Models with other heads (SCRFD, pose,
classification, etc.) are rejected with a clear log message.

There are three ways to get models:

- export your own with Lichee YOLO Studio (training → ONNX → tpu-mlir);
- keep the models that already live in `/mnt/system/usr/bin/ai/models`;
- fetch the SDK examples with `scripts/fetch-vendor.sh --download`.

Models derived from Ultralytics YOLOv8/YOLO11 are AGPL-3.0: if you publish
them, publish the corresponding sources too. Face and SDK example models have
no explicit license. See `../../../models/README.md` and
`../../../THIRD_PARTY_NOTICES.md`.

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
