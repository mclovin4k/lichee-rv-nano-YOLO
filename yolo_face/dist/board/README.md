# Board files: face_stream

Contents of the release package: `face_stream`, `run_face.sh`,
`start_face.sh`, an empty `models/face/` directory and `md5sums.txt`.

The package deliberately contains **no vendor libraries and no models**: the
Cvitek/Sophgo runtime is proprietary and the face models come from the TDL SDK
without a stated license, so neither is redistributed. `install.sh` copies the
libraries from the firmware of your board and uploads the models you choose.
See `../../../THIRD_PARTY_NOTICES.md`.

## Installation

The easiest way is the release installer from the repository root:

```sh
cd yolo_face
./install.sh <IP>
```

It downloads `yolo_face-board.tar.gz` from the latest GitHub Release, verifies
its checksum, copies the files to `/root/yolo_face` and then

1. copies `libcvi_tdl`, its dependencies (`libcnpy`, `libcvi_ive`, shared
   OpenCV 3.2), the camera/ISP/VENC middleware and the rest of the vendor
   runtime from the firmware into `/root/yolo_face/runtime`;
2. uploads `*.cvimodel` from `MODELS_DIR` or `../vendor/models` into
   `models/face/`;
3. verifies the package checksums on the board.

Then:

```sh
/root/yolo_face/start_face.sh start
```

The camera is configured by the stock image through `/mnt/data/sensor_cfg.ini`.
No autostart is installed.

## Control

```sh
/root/yolo_face/start_face.sh start|stop|restart|status|log
```

- UI: `http://<IP>:7777/`
- MJPEG: `http://<IP>:7777/stream`
- Log: `/root/yolo_face/log/face.log`
- Gallery: `/root/faces/<name>/`
- PID: `/tmp/yolo_face.pid` (supervisor), `/tmp/yolo_face.app.pid`

`start_face.sh start` stops `stream_yolo` (shared camera and port). The
reverse is also true: `yolo_stream/start_stream.sh` stops `face_stream`.

## Runtime

The binary is a musl executable and uses the loader and libc of the firmware.
`run_face.sh` builds `LD_LIBRARY_PATH` in this order:

```
/root/yolo_face/runtime              # copies made by install.sh
/mnt/system/usr/lib
/mnt/system/usr/lib/3rd
/mnt/system/lib
/usr/lib
```

`libcvi_tdl.so` is loaded with `dlopen` because it is built against shared
OpenCV 3.2, while the application uses static OpenCV 4.13: mixing them in one
link would crash. The TDL stack therefore needs `libcnpy.so`,
`libcvi_ive.so`, `libopencv_{core,imgproc,imgcodecs}.so.3.2` and `libz` next
to it — `install.sh` copies all of them from the firmware.

`libvpu_tdl.so` is optional. It is a patched middleware build that provides
the GDC symbols TDL needs; if it is absent, the application falls back to the
system `libvpu.so` from `/mnt/system/usr/lib`, which is the normal case on a
stock image. If you have the patched file, drop it into `runtime/` and
`run_face.sh` will point `SY_VPU_LIB` at it.

A bundled loader is used only when you put one there on purpose:

```sh
SY_BUNDLED_LOADER=1 /root/yolo_face/start_face.sh restart
```

## Models

`models/face/` must contain:

| File | Purpose |
|---|---|
| `scrfd_det_face_432_768_INT8_cv181x.cvimodel` | face detector |
| `feature_cviface_112_112_INT8_cv181x.cvimodel` | face embeddings |
| `cls_rgbliveness_256_256_INT8_cv181x.cvimodel` | liveness check |

They are part of the Cvitek TDL SDK examples and are already present on the
board image in `/mnt/system/usr/bin/ai/models`; `install.sh` copies the ones it
finds there when you pass `MODELS_DIR`, and
`scripts/fetch-vendor.sh --download` fetches the SDK detector
(`scrfd_768_432_int8_1x.cvimodel`) if you prefer to start from that.

The SCRFD architecture is MIT (InsightFace); the weights, `cviface` and the
liveness model have no explicit license in the SDK. Check the terms with
Cvitek/Sophgo before redistributing them. See `../../../models/README.md`.
