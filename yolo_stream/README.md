# stream_yolo — camera → NPU detector → MJPEG on LicheeRV Nano (SG2002/CV181x)

A 960×540 @30 fps MJPEG stream with YOLOv8 boxes computed on the NPU, plus an
HTTP server on port 7777 for VLC/ffplay (the resolution is set by
`SY_HP="W H"`). Frames are not converted to BGR and not copied through
`cv::Mat`: VPSS outputs NV21, and the same buffer is used both by the hardware
JPEG encoder and (via a letterboxed copy) by the detector. The project was
originally based on
[ret7020/LicheeRVNano](https://github.com/ret7020/LicheeRVNano); the current
implementation has been rewritten (see below).

## Contents

| Item | Location |
|---|---|
| Sources | `main.cpp`, `yolo8.c`, `vpss_det.c`, `venc_jpeg.c`, `draw_nv21.c`, `MJPEGWriter.*` |
| Ready-to-run package | `yolo_stream-board.tar.gz` from Releases (binary, scripts; no vendor files, no models) |
| Runtime skeleton | `dist/board/runtime/` — empty in git; `install.sh` fills it from the board firmware |
| Models | `dist/board/models/` — empty in git; upload your own or use the ones on the board |
| Start/stop scripts | `dist/board/start_stream.sh`, `dist/board/run_stream.sh` |
| SSH installer | `install.sh` (downloads the latest release by default) |
| Build | `CMakeLists.txt`, `cmake/`, `scripts/`, `patches/`, `../scripts/package-board.sh` |

## Architecture

```
GC4653 (MIPI CSI; I2C address and MIPI lane_id come from /mnt/data/sensor_cfg.ini)
  → VI (2560×1440)                     ─┐ kernel/RTOS, DDR
  → VPSS group 0, channel 0: crop/scale 960×540 (SY_HP), NV21 (fmt=19)
      │
      ├─ [main stream] boxes drawn over Y/UV → CVI_SYS_IonFlushCache →
      │                VENC (PT_JPEG, channel 2) → MJPEGWriter :7777
      │
      └─ [detector thread] frame copy (luma+chroma, stride preserved)
           → preprocessing NV21→RGB planar int8 (letterbox 640×640)
           → CVI_NN_ForwardAsync / CVI_NN_ForwardWait (800 ms deadline)
           → decode 6 tensors (anchor-free YOLOv8, 3 branches) + NMS
           → publish boxes (mutex + generation)
```

Key design decisions:

- **No Mat on the hot path.** `OPENCV_CVI_RAW=1` switches the opencv-mobile
  backend into "raw" mode: `read_frame` caches the plane mappings of the VPSS
  frame (`CVI_SYS_MmapCache`, cached by physical address until `close()`) and
  invalidates the cache on every frame. There is no per-frame `mmap/munmap`
  and no `BGR→NV21` conversion anymore.
- **Inference goes directly through `cviruntime`** (`yolo8.c`), without
  `CVI_TDL_*`. Asynchronous forward with a deadline: if the wait takes too
  long, the model is unloaded (`CVI_NN_CleanupModel`) and registered again —
  this is the self-recovery after a driver wedge.
- **Thread separation.** The detector works on a private frame copy (a fresh
  frame is submitted only when the detector is idle), so a TPU wedge does not
  stop the stream: the picture keeps running at 30 fps and the boxes simply
  become stale (`det_age` in the statistics).
- **Freshness matters more than having boxes.** A result that returns from
  forward too late (its frame is older than `SY_DET_MAX_AGE_MS`, 2 s by
  default) is not published. Drawn boxes older than `SY_BOX_TTL_MS` (4 s by
  default) are not rendered: instead the frame shows `no det Ns`, which makes
  it clear that the detector is waiting for platform recovery rather than the
  stream being stuck. This way the stream never shows a box from a scene that
  is already gone.
- **VENC is hardware** (PT_JPEG) and reads the same physical buffer;
  `CVI_SYS_IonFlushCache` is called before sending so the encoder sees the
  drawn pixels.
- **Restart without reboot.** When opening the camera, if
  `CVI_VPSS_CreateGrp` fails (a stale group from a previous process is left
  behind), the group is destroyed and created again.
- **Classes and model input come from tensors.** The binary reads the number
  of classes, the input size and the output format itself (`auto` mode), so
  any YOLOv8/v11-det export works (int8 and fp32), not just the hard-coded 80
  classes.

## Detection accuracy (comparison with the TDL reference)

The boxes were compared with the board's reference pipeline — the TDL sample
`sample_object_detection` (same coco80 cvimodel, same frame, offline). The
class, score and coordinates match to within a tenth of a pixel:

| frame | TDL | stream_yolo |
|---|---|---|
| person | person 0.8017, x 0.98..611.59, y 270.78..638.78 | 0.802, x 0.1..611.8, y 270.9..638.8 |
| cabinet | refrigerator 0.6981, x 324.09..583.91, y 296.67..638.92 | 0.698, x 324.9..583.8, y 297.4..638.9 |

Three defects found during this comparison (all fixed):

1. **NV21→RGB: luma formula.** The code had `c = y - 128` (a typo for the
   limited-range `c = y - 16`): half of the luma range was mapped to black,
   the detector saw an almost black frame — that was the cause of the
   "missing boxes". VPSS outputs **full-range** NV21 (a raw dump contains Y
   values 15 and 255, outside the 16..235 range), so full-range conversion is
   used:
   `r = y + 1.402(v-128)`, `g = y - 0.344(u-128) - 0.714(v-128)`,
   `b = y + 1.772(u-128)`.
2. **DFL decoding.** Values after `sigmoid` were fed into softmax; the
   reference applies softmax to the raw dequantized logits (`exp(p * qscale)`).
   Because of the compressed distribution, the boxes had the wrong size.
3. **Tensor input offset.** The fill was `(px - 128) * gain`; the reference
   config has mean = 0, i.e. `q = px * qscale/255` (0..127), with no
   subtraction.

For re-checking there is an offline mode without a camera:

- `SY_DUMP_NV21=/tmp/f.nv21` — save the first raw camera frame;
- `SY_TEST_NV21=/tmp/f.nv21` — run the detector on that file (tight layout:
  Y `640×640` + UV `640×320`) and print the results, without VPSS/VENC.

## Models

No model is shipped in git or in the release package: exports derived from
Ultralytics YOLOv8/YOLO11 are AGPL-3.0, and this repository is MIT. See
[`../models/README.md`](../models/README.md) for the full picture.

The binary accepts any YOLOv8/YOLO11 detection export with the six-output raw
head produced by `yolo_studio`, int8 or fp32; it reads the number of classes
and the input size from the tensors (`auto` mode). Put the file into
`dist/board/models/` (or upload it with `MODELS_DIR=... ./install.sh <IP>`);
`start_stream.sh` also searches `/root/models` and
`/mnt/system/usr/bin/ai/models`, so the stock models keep working without
copying anything.

Class names come from a `<model_name>.names` file next to the model (one name
per line). `yolo_studio` writes and uploads it automatically.

## Building

You need a RISC-V musl toolchain, the SDK and OpenCV:

- RISC-V musl toolchain. By default, `scripts/fetch-deps.sh` downloads
  [musl-cross 20261001](https://github.com/musl-cross/musl-cross/releases/tag/20261001)
  (`riscv64-unknown-linux-musl`, SHA256 pinned); [musl.cc](https://musl.cc/)
  `riscv64-linux-musl-cross` (GCC 11.2.1) and Sipeed/Sophgo with the
  `riscv64-unknown-linux-musl-` prefix are also supported;
- [milkv-duo/cvitek-tdl-sdk-sg200x](https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x)
  — middleware headers and `libvenc.so`/`libsys.so`, the commit is pinned in
  `scripts/fetch-deps.sh`;
- the official [nihui/opencv-mobile v36](https://github.com/nihui/opencv-mobile/releases/tag/v36)
  (`opencv-mobile-4.13.0.zip`, Apache-2.0) plus the
  `patches/opencv-mobile-4.13.0-cvi-capture.patch`.

Everything is downloaded and built by the scripts:

```sh
cd yolo_stream
./scripts/fetch-deps.sh       # toolchain, SDK, OpenCV sources + patch
./scripts/build-opencv.sh     # OpenCV into .deps/opencv-install
./scripts/build.sh            # build/bin/stream_yolo
```

Or with pre-installed paths:

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
**Run workflow**. The run produces `yolo_stream-bin` (the bare binary) and
`yolo_stream-board` (binary + scripts, no vendor files) artifacts. Pushing a
`v*` tag publishes both release assets automatically.

To create the complete package locally after a build:

```sh
../scripts/package-board.sh yolo_stream build/bin/stream_yolo ../out
```

The archive contains the binary, the scripts and an empty `models/` directory.
To embed the vendor files for a private, offline archive, add
`--with-vendor ../vendor` (do not publish that archive).

## Installing on the board

No clone is needed — the installer can be fetched on its own:

```sh
curl -fsSLO https://raw.githubusercontent.com/mclovin4k/lichee-rv-nano-YOLO/master/yolo_stream/install.sh
sh install.sh                # downloads the latest release, asks for the IP
```

From a checkout of the repository:

```sh
chmod +x install.sh
./install.sh                 # downloads the latest release, asks for the IP
BOARD_IP=<IP> ./install.sh
RELEASE=v1.0.1 ./install.sh <IP>
PAYLOAD_DIR=/path/to/package ./install.sh <IP>
```

By default the installer downloads `yolo_stream-board.tar.gz` from the latest
GitHub Release, verifies its SHA256, copies it to `/root/yolo_stream`, checks
the package md5 list on the board, and does **not** add an autostart entry.

The archive contains only this project's binary and scripts. Right after the
upload the installer runs a small script **on the board** that copies the
vendor libraries from `/mnt/system/lib`, `/mnt/system/usr/lib` and
`/mnt/system/usr/lib/3rd` into `/root/yolo_stream/runtime`. Models are uploaded
from `MODELS_DIR` or `../vendor/models` when either is set (the search is
recursive); otherwise the models already on the board are used. Use
`PAYLOAD_DIR` to install a locally built package instead.

The installer preserves already uploaded models, and `/root/faces` is used
only by `yolo_face`. The board address is not fixed: it differs between
networks (USB/RNDIS, Wi-Fi, Ethernet).

The camera is configured by the stock image through `/mnt/data/sensor_cfg.ini`
(sensor I2C address and MIPI `lane_id`). The scripts do not touch pinmux or
`/dev/mem`; if the file is missing, the installer warns about it.

### Manual installation

```sh
curl -fLO https://github.com/mclovin4k/lichee-rv-nano-YOLO/releases/latest/download/yolo_stream-board.tar.gz
curl -fLO https://github.com/mclovin4k/lichee-rv-nano-YOLO/releases/latest/download/yolo_stream-board.tar.gz.sha256
sha256sum -c yolo_stream-board.tar.gz.sha256
tar -xzf yolo_stream-board.tar.gz -C /tmp/yolo_stream-package
tar -C /tmp/yolo_stream-package -cf - . | \
    ssh root@<IP> 'rm -rf /root/yolo_stream; mkdir -p /root/yolo_stream; tar -xf - -C /root/yolo_stream'
ssh root@<IP> 'chmod +x /root/yolo_stream/stream_yolo /root/yolo_stream/*.sh'
```

Then copy the vendor libraries from the firmware once (this is what
`install.sh` automates):

```sh
ssh root@<IP> 'mkdir -p /root/yolo_stream/runtime && cd /root/yolo_stream/runtime &&
  for n in libcviruntime.so libcvikernel.so libvenc.so libsys.so libisp.so libisp_algo.so \
           libae.so libawb.so libsns_gc4653.so libcvi_bin.so libcvi_bin_isp.so; do
    for d in /mnt/system/lib /mnt/system/usr/lib /mnt/system/usr/lib/3rd; do
      [ -f "$d/$n" ] && cp -L "$d/$n" . && break
    done
  done'
```

## Running

```sh
ssh root@<IP> '/root/yolo_stream/start_stream.sh'          # model menu
ssh root@<IP> '/root/yolo_stream/start_stream.sh auto'     # no menu
ssh root@<IP> '/root/yolo_stream/start_stream.sh status'
ssh root@<IP> '/root/yolo_stream/start_stream.sh stop'
```

- Stream: `http://<IP>:7777/` (MJPEG, VLC/ffplay)
- Log: `/root/yolo_stream/log/stream.log` (the previous one is moved to `.1`
  at startup)
- Models: `models/` next to the script, then `/root/models`, then
  `/mnt/system/usr/bin/ai/models`
- Last model file: `/root/yolo_stream/log/last_model`

`stream_yolo` and `face_stream` from the neighboring `yolo_face` project use
the same camera and port 7777, so the modes are mutually exclusive:
`start_stream.sh` stops `face_stream`, and `start_face.sh start` stops
`stream_yolo`. No autostart is installed — the mode is chosen manually after
boot.

### Loader and libraries

The binary is a musl executable and runs on the loader and libc of the
firmware. `run_stream.sh` puts, in this order, the copies in `runtime/`, then
the firmware directories, into `LD_LIBRARY_PATH`:

```
runtime/  /mnt/system/usr/lib  /mnt/system/usr/lib/3rd  /mnt/system/lib  /usr/lib
```

A bundled loader is used only if you place one in `runtime/` and opt in:

```sh
SY_BUNDLED_LOADER=1 /root/yolo_stream/start_stream.sh auto
```

The script stores the pid in `/tmp/yolo_stream.app.pid`, so it works with both
the system loader and an explicit one.

## Environment variables (no rebuild required)

| Variable | Default | Effect |
|---|---|---|
| `SY_HP` | "960 540" | stream resolution, W H |
| `SY_KEEP_LOG` | off | `1` — do not rotate the log to `.1` at startup |
| `SY_STREAM_FPS` | 0 (max) | throttle the capture/encode loop |
| `SY_ENC_EVERY` | 1 | encode every Nth frame |
| `SY_ENC` | 1 | `0` — disable VENC (for experiments) |
| `SY_DRAW` | 1 | `0` — do not draw boxes |
| `SY_DET` | 1 | `0` — disable the detector thread |
| `SY_HTTP_PORT` | 7777 | MJPEG server port (e.g. `7778` for a clean capture stream) |
| `SY_DET_THR` | 0.5 | detection score threshold (lower — more weak boxes) |
| `SY_NAMES_FILE` | `<model>.names` | class-name sidecar (one per line); without it boxes are labeled `class:N` |
| `SY_WEDGE_MS` | 0 (off) | detector silence threshold: "stalled" is logged once per interval |
| `SY_WEDGE_REBOOT` | 0 | `1` — reboot if silence lasts longer than 3×`SY_WEDGE_MS` |
| `SY_DET_MAX_AGE_MS` | 2000 | do not publish a result if its frame is older than this (after a wedge) |
| `SY_BOX_TTL_MS` | 4000 | do not draw boxes older than this; `0` — always draw |
| `SY_MAX_BOXES` | 16 | do not publish a result with more than N boxes (corrupt frame); `0` — disable |
| `SY_DUMP_NV21` | off | write the first raw NV21 frame to a file (debug) |
| `SY_TEST_NV21` | off | run the detector offline on an NV21 file and exit (debug) |
| `SY_BUNDLED_LOADER` | 0 | `1` — load the app with `runtime/ld-musl-riscv64.so.1` |

Command-line arguments: `stream_yolo <path to .cvimodel> <classes|auto> <W> [H]`
(`auto` — the binary reads the number of classes from the model tensors;
normally this argument is supplied by `start_stream.sh`).

Class names are not stored in `.cvimodel`. Put a
`<model_name>.names` file next to it (one name per line) — `stream_yolo` will
label the boxes with those names. yolo_studio creates and uploads such a file
together with the model automatically; without it the labels remain
`class:N`.

## Verification

```sh
ssh root@<IP> 'grep "\[stream\]" /root/yolo_stream/log/stream.log | tail -3'
# [stream] frames=1080 fps=30.0 det=83.8ms (prep=38.3 fwd=37.0 dec=10.7) boxes=1 det_age=0.0s
#   fps      — stream rate (independent of the detector)
#   det      — EMA of one inference time (prep/fwd/dec — input preparation,
#              NPU forward, tensor decoding)
#   boxes    — number of boxes in the latest result
#   det_age  — age of the latest boxes (grows during a TPU wedge)

ssh root@<IP> 'grep "^det:" /root/yolo_stream/log/stream.log | tail -3'
# det: n=1 max=0.636 | 72:0.64 [325 297 585 639]
#   n        — number of boxes after NMS
#   max      — best network score in the frame (visible even without boxes)
#   cls:score [x1 y1 x2 y2] — detections, coordinates in the 640×640 frame

# during a TPU wedge the frame shows "no det Ns" (boxes older than the TTL
# are not drawn), and late results are dropped:
ssh root@<IP> 'grep -c "dropped stale result" /root/yolo_stream/log/stream.log'

# a frame from the stream (on the host):
curl -s -m 5 http://<IP>:7777/ | head -c 200000 > /tmp/m.bin
# then extract the first JPEG (FFD8..FFD9)
```

## Limitations

- In the backend's `close()` the second `stop_streaming` sometimes crashes
  (exit 139) — the camera is released, and the next start brings up the VPSS
  group ("reclaiming stale group").
- A process stuck in the TPU ioctl cannot be killed with `kill -9` until the
  driver timeout expires; `start_stream.sh` waits for it to disappear, so no
  "old/new" process pair remains.
- MJPEG q75 at 30 fps is about 9 Mbit/s; use `SY_ENC_EVERY` to save
  bandwidth.
- Third-party model and library licenses are in `../THIRD_PARTY_NOTICES.md`.
