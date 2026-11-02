# face_stream — MJPEG + NPU face recognition (LicheeRV Nano, SG2002/CV181x)

A ready-to-run application for the LicheeRV Nano with the GC4653 camera:
960×540 MJPEG on port 7777, SCRFD face detection, gallery-based recognition
(cviface) and passive blink-based liveness checking. The release package
includes the binary, runtime libraries, matching musl loader/libc and start
scripts, so no cross-compilation is needed to run it.

## Contents

| Item | Location |
|---|---|
| Sources | `src/` |
| Ready-to-run package | `yolo_face-board.tar.gz` from Releases (binary, scripts; no vendor files, no models) |
| Runtime skeleton | `dist/board/runtime/` — empty in git; `install.sh` fills it from the board firmware |
| TDL models | `dist/board/models/face/` — empty in git; upload your own or use the ones on the board |
| Start/stop scripts | `dist/board/start_face.sh`, `dist/board/run_face.sh` |
| SSH installer | `install.sh` (downloads the latest release by default) |
| Build | `CMakeLists.txt`, `cmake/`, `scripts/`, `patches/`, `../scripts/package-board.sh` |

`libcvi_tdl.so` is loaded via `dlopen`: it is built against shared OpenCV 3.2,
while the application uses static OpenCV 4.13 (with the CVI camera backend).
Linking them directly mixes two OpenCV implementations and crashes. That is
why `libcvi_tdl.so`, `libcnpy.so`, `libcvi_ive.so` and shared OpenCV 3.2 are
placed together in the runtime.

The patched middleware build `libvpu_tdl.so` is optional: TDL uses a few GDC
symbols that the stock `libvpu.so` also provides, so without the patched file
the application falls back to `/mnt/system/usr/lib/libvpu.so`, which matches
the firmware driver. Put `libvpu_tdl.so` into `runtime/` only if you have a
build that is known to work with your firmware; `run_face.sh` then points
`SY_VPU_LIB` at it.

## Installing on the board

No clone is needed — the installer can be fetched on its own:

```sh
curl -fsSLO https://raw.githubusercontent.com/mclovin4k/lichee-rv-nano-YOLO/master/yolo_face/install.sh
sh install.sh                # downloads the latest release, asks for the IP
```

From a checkout of the repository:

```sh
chmod +x install.sh
./install.sh                 # downloads the latest release, asks for the IP
BOARD_IP=<IP> ./install.sh
RELEASE=v1.0.0 ./install.sh <IP>
PAYLOAD_DIR=/path/to/package ./install.sh <IP>
TPU_PATCH=1 ./install.sh <IP>   # also apply the TPU timeout fix
```

The board address is not fixed: it differs between networks (USB/RNDIS,
Wi-Fi, Ethernet). By default the installer:

1. downloads `yolo_face-board.tar.gz` from the latest GitHub Release;
2. verifies its SHA256;
3. checks SSH and stops old `face_stream`/`stream_yolo` processes;
4. copies the package to `/root/yolo_face`, preserving `/root/faces` (the
   gallery);
5. verifies the package md5 list on the board;
6. optionally applies `../tpu-fix/patch_tpu.sh`;
7. does **not** add an autostart entry.

The archive contains only this project's binary and scripts. After the upload
the installer copies the vendor runtime from the firmware
(`/mnt/system/lib`, `/mnt/system/usr/lib`, `/mnt/system/usr/lib/3rd`) into
`/root/yolo_face/runtime`, including `libcvi_tdl`, its shared OpenCV 3.2
dependencies and `libz`. Models are uploaded from `MODELS_DIR` or
`../vendor/models`; without them the models already on the board are used.
Use `PAYLOAD_DIR` to install a locally built package instead of downloading
one.

The camera is configured by the stock image through `/mnt/data/sensor_cfg.ini`
(sensor I2C address and MIPI `lane_id`). The scripts do not touch pinmux or
`/dev/mem`; if the file is missing, the installer warns about it.

### Manual installation

```sh
curl -fLO https://github.com/mclovin4k/lichee-rv-nano-YOLO/releases/latest/download/yolo_face-board.tar.gz
curl -fLO https://github.com/mclovin4k/lichee-rv-nano-YOLO/releases/latest/download/yolo_face-board.tar.gz.sha256
sha256sum -c yolo_face-board.tar.gz.sha256
tar -xzf yolo_face-board.tar.gz -C /tmp/yolo_face-package
tar -C /tmp/yolo_face-package -cf - . | \
    ssh root@<IP> 'rm -rf /root/yolo_face; mkdir -p /root/yolo_face; tar -xf - -C /root/yolo_face'
ssh root@<IP> 'chmod +x /root/yolo_face/face_stream /root/yolo_face/*.sh'
```

Then copy the vendor libraries from the firmware (this is what `install.sh`
automates):

```sh
ssh root@<IP> 'mkdir -p /root/yolo_face/runtime && cd /root/yolo_face/runtime &&
  for n in libcviruntime.so libcvikernel.so libvenc.so libsys.so libisp.so libisp_algo.so \
           libae.so libawb.so libsns_gc4653.so libcvi_bin.so libcvi_bin_isp.so \
           libcvi_tdl.so libcvimath.so libcnpy.so libcvi_ive.so libz.so.1.2.11 \
           libopencv_core.so.3.2 libopencv_imgproc.so.3.2 libopencv_imgcodecs.so.3.2; do
    for d in /mnt/system/lib /mnt/system/usr/lib /mnt/system/usr/lib/3rd; do
      [ -f "$d/$n" ] && cp -L "$d/$n" . && break
    done
  done'
```

## Running

```sh
ssh root@<IP> '/root/yolo_face/start_face.sh start'
ssh root@<IP> '/root/yolo_face/start_face.sh status'
ssh root@<IP> '/root/yolo_face/start_face.sh stop'
```

`start_face.sh start` stops `stream_yolo` by itself: they share the camera and
port 7777. The reverse is also true: `yolo_stream/start_stream.sh` stops
`face_stream`. There is no autostart — the mode is chosen manually after boot.

- Web UI: `http://<IP>:7777/` — video, list of people, name field, "Remember".
- MJPEG: `http://<IP>:7777/stream`
- Log: `/root/yolo_face/log/face.log`
- Gallery: `/root/faces/<name>/feature.bin` (plus `meta.json`,
  `preview.jpg`)

The supervisor restarts the application if the watchdog returns code 2 (TPU
hang). A normal exit via SIGINT is not restarted.

### Loader and libraries

The binary is a musl executable and runs on the loader and libc of the
firmware. `run_face.sh` puts, in this order, the copies in `runtime/`, then the
firmware directories, into `LD_LIBRARY_PATH`:

```
runtime/  /mnt/system/usr/lib  /mnt/system/usr/lib/3rd  /mnt/system/lib  /usr/lib
```

A bundled loader is used only if you place one in `runtime/` and opt in:

```sh
SY_BUNDLED_LOADER=1 /root/yolo_face/start_face.sh start
```

`start_face.sh` stores the pid in `/tmp/yolo_face.app.pid`, so it works with
both the system loader and an explicit one.

## How it works

1. OpenCV with the CVI backend delivers a 960×540 NV21 frame from VPSS
   (without copying through `cv::Mat`).
2. A separate VPSS channel delivers RGB_888 for TDL: SCRFD finds faces,
   cviface computes an embedding, and the gallery compares them by cosine
   similarity.
3. Inference runs in a separate thread: a stuck TPU does not stop the video.
4. The frame with boxes goes to the hardware VENC (JPEG), then to the MJPEG
   server.
5. "Remember" collects 12 frames of the largest face, averages the normalized
   vector, and stores it in `/root/faces/<name>/`.

Box colors: yellow `name?` — a match exists, but liveness has not been
confirmed yet; green `name` — confirmed by blinking; red `photo?` — the face
does not blink; yellow `Unknown` — the face is not in the gallery.

The TDL liveness model (`cls_rgbliveness_256_256`) crashes the TPU driver on
this firmware, so it is disabled by default (`SY_LIVE=0`). Photo protection is
provided by passive blink detection: the person does not need to do anything,
and a photograph does not blink. The confirmation lives for 15 s and expires
without new blinks.

## Building

You need:

- RISC-V musl toolchain. By default, `scripts/fetch-deps.sh` downloads
  [musl-cross 20261001](https://github.com/musl-cross/musl-cross/releases/tag/20261001)
  (`riscv64-unknown-linux-musl`, SHA256 pinned); [musl.cc](https://musl.cc/)
  `riscv64-linux-musl-cross` (GCC 11.2.1) and Sipeed/Sophgo with the
  `riscv64-unknown-linux-musl-` prefix are also supported.
- SDK [milkv-duo/cvitek-tdl-sdk-sg200x](https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x)
  (middleware v2 and `cvi_tdl` headers, `libvenc.so`, `libsys.so`), the commit
  is pinned in `scripts/fetch-deps.sh`.
- OpenCV: the official release
  [nihui/opencv-mobile v36, opencv-mobile-4.13.0.zip](https://github.com/nihui/opencv-mobile/releases/tag/v36)
  (Apache-2.0) plus `patches/opencv-mobile-4.13.0-cvi-capture.patch`.
- `libcviruntime`/`libcvikernel` headers and libraries —
  `third_party/cviruntime`.

```sh
cd yolo_face
./scripts/fetch-deps.sh       # toolchain, SDK, OpenCV sources + patch
./scripts/build-opencv.sh     # build/install OpenCV into .deps/opencv-install
./scripts/build.sh            # build/bin/face_stream
```

Or point the build at pre-installed paths:

```sh
TOOLCHAIN_PATH=/path/to/riscv64-linux-musl-cross \
CVISDK_PATH=/path/to/cvitek-tdl-sdk-sg200x \
OpenCV_DIR=/path/to/opencv/lib/cmake/opencv4 \
./scripts/build.sh
```

`TOOLCHAIN_PATH` is read both from the environment and via
`-DTOOLCHAIN_PATH`; exporting it is more reliable for `try_compile`.
`scripts/fetch-deps.sh` downloads only the required SDK files (sparse
checkout), without examples and binaries.

The same build can be run on GitHub: **Actions** tab → **build** →
**Run workflow**. The run produces `yolo_face-bin` (the bare binary) and
`yolo_face-board` (binary + scripts, no vendor files) artifacts. Pushing a
`v*` tag publishes both release assets automatically.

To create the complete package locally after a build:

```sh
../scripts/package-board.sh yolo_face build/bin/face_stream ../out
```

The archive contains the binary, the scripts and an empty `models/face/`
directory. To embed the vendor files for a private, offline archive, add
`--with-vendor ../vendor` (do not publish that archive).

## Models and licenses

The repository does not ship the `scrfd_det_face_432_768`,
`feature_cviface_112_112` and `cls_rgbliveness_256_256` models: they come from
the TDL SDK examples and have no license file. They are already present in
`/mnt/system/usr/bin/ai/models` on a stock image; `install.sh` uploads them
from `MODELS_DIR` when you point it there. See
[`../models/README.md`](../models/README.md).

The vendor libraries the application loads at run time (`libcvi_tdl.so`,
`libcnpy.so`, `libcvi_ive.so`, `libcviruntime.so`, `libcvikernel.so`,
`libcvimath.so`, shared OpenCV 3.2, `libz`) are copied from the board firmware
during installation, not from this repository. Details and links are in
`../THIRD_PARTY_NOTICES.md`.

OpenCV 4.13 (static, for the application) is distributed under Apache-2.0;
OpenCV 3.2 (shared, for TDL) under BSD-3-Clause. The camera patch is applied
to the official `nihui/opencv-mobile` v36 release.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `SY_DET_THR` | 0.40 | SCRFD detection threshold |
| `SY_FACE_THR` | 0.40 | gallery similarity threshold |
| `SY_FACE_PERIOD_MS` | 70 | pause between inferences |
| `SY_MAX_BOXES` | 8 | more boxes means a detection failure; the frame is skipped |
| `SY_PUB_BOXES` | 5 | how many faces to display |
| `SY_PUB_SCORE` | 0.55 | minimum score to display |
| `SY_PUB_MIN` | 48 | minimum box side, px |
| `SY_LIVE` | 0 | 1 — enable the liveness model (not recommended) |
| `SY_LIVE_BLINKS` | 1 | how many blinks confirm liveness |
| `SY_LIVE_CHECK_MS` | 12000 | how long to wait for a blink |
| `SY_LIVE_KEEP_MS` | 15000 | how long the confirmation lives |
| `SY_ROTATE` | 0 | frame rotation for the detector: 0/90/180/270 |
| `SY_FACES_DIR` | `/root/faces` | gallery directory |
| `SY_TDL_LIB` | `runtime/libcvi_tdl.so` | TDL library |
| `SY_VPU_LIB` | `runtime/libvpu_tdl.so` | libvpu for TDL |
| `SY_BUNDLED_LOADER` | 1 | 0 — use the system musl loader instead of `runtime/` |
| `WIDTH`, `HEIGHT` | 960, 540 | frame size |

## Limitations

- TDL accepts a 960×540 frame; rotating by 90/270 gives 540×960 and
  `ILLEGAL_PARAM`. For a rotated enclosure, use `SY_ROTATE` only if TDL
  accepts it; the default is 0.
- The liveness model is disabled (see above).
- TPU hangs (`wait dmabuf failed[-110]`, `run cmdbuf failed:2`) are a platform
  driver defect. With the fix from `../tpu-fix/`, the wait is reduced from
  ~60 s to ~1 s; the application recreates the TDL handle.
- Closing the camera sometimes hangs in the driver (`vb_qbuf`, state `D`); if
  the process cannot be killed, a `reboot` is required.
- MJPEG q75 at 30 fps is about 9 Mbit/s.
