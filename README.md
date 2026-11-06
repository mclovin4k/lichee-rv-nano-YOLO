# LicheeRV Nano (SG2002/CV181x): MJPEG, YOLO on the NPU, face recognition and a model studio

[![build](https://github.com/mclovin4k/lichee-rv-nano-YOLO/actions/workflows/build.yml/badge.svg)](https://github.com/mclovin4k/lichee-rv-nano-YOLO/actions/workflows/build.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A working toolkit for the LicheeRV Nano (SG2002 / CV181x, RISC-V) with the
GC4653 camera: MJPEG streaming, YOLOv8/YOLO11 detection on the NPU, face
recognition, and a desktop studio for importing datasets, labeling, training
and exporting `.cvimodel` models. The repository contains three independent
projects.

The repository contains **its own code only**. Cvitek/Sophgo runtime
libraries and `.cvimodel` models are not redistributed: they are proprietary
(or AGPL, in the case of Ultralytics derivatives), so the installers take the
libraries from the board firmware and upload only the models you provide. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

| Project | What it is | Quick start |
|---|---|---|
| [`yolo_stream/`](yolo_stream/README.md) | MJPEG 960×540 + YOLOv8/YOLO11 on the NPU, port 7777 | `yolo_stream/install.sh <IP>` → `start_stream.sh auto` |
| [`yolo_face/`](yolo_face/README.md) | MJPEG + face detection/recognition (SCRFD + cviface) | `yolo_face/install.sh <IP>` → `start_face.sh start` |
| [`yolo_studio/`](yolo_studio/README.md) | Desktop tool: dataset import/capture, labeling, training, `.cvimodel` export | download the Linux/Windows build from Releases |

## Demo

Two modes on the same camera — object detection and face recognition:

<table>
  <tr>
    <td width="50%" align="center" valign="top">
      <a href="yolo_stream/README.md">
        <img src="docs/images/demo-stream.gif"
             width="100%" alt="YOLO object detection on the LicheeRV Nano NPU">
      </a>
      <br>
      <sub><b><a href="yolo_stream/README.md">yolo_stream</a></b> —
      YOLOv8/YOLO11 detection on the NPU</sub>
    </td>
    <td width="50%" align="center" valign="top">
      <a href="yolo_face/README.md">
        <img src="docs/images/demo-face.gif"
             width="100%" alt="Face detection and recognition on the LicheeRV Nano NPU">
      </a>
      <br>
      <sub><b><a href="yolo_face/README.md">yolo_face</a></b> —
      SCRFD detection, cviface recognition and blink liveness</sub>
    </td>
  </tr>
</table>

**Lichee YOLO Studio** — import a dataset, label frames, train on CPU and export
a `.cvimodel` for the board:

<p align="center">
  <img src="docs/images/yolo-studio.png" width="1100"
       alt="Lichee YOLO Studio label page with two drone bounding boxes">
</p>

`yolo_stream` and `yolo_face` share the camera and port 7777, so starting one
stops the other. No autostart is installed: after boot you choose the mode
manually.

## Ready-to-run releases

Download the latest files from
[GitHub Releases](https://github.com/mclovin4k/lichee-rv-nano-YOLO/releases/latest):

| Asset | Contents |
|---|---|
| `yolo_stream-board.tar.gz` | `stream_yolo`, start/run scripts, empty `models/`, checksums — no vendor libraries |
| `yolo_face-board.tar.gz` | `face_stream`, start/run scripts, empty `models/`, checksums — no vendor libraries |
| `yolo_studio-linux-x86_64.tar.gz` | one-file Lichee YOLO Studio for Linux x86_64 (slim) |
| `yolo_studio-linux-x86_64-full.tar.gz` | one-folder Studio with ultralytics + torch inside |
| `yolo_studio-windows-x86_64.tar.gz` | one-file `LicheeYOLOStudio.exe` (slim) |
| `yolo_studio-windows-x86_64-full.tar.gz` | one-folder Studio with ultralytics + torch inside |

GitHub shows the SHA-256 digest of every asset on the release page.

The board archives are small because they do not carry the vendor runtime: on
installation the libraries are copied from the firmware that is already on
your board, and models are uploaded from your machine if you have them.

```sh
git clone https://github.com/mclovin4k/lichee-rv-nano-YOLO.git
cd lichee-rv-nano-YOLO

cd yolo_stream
./install.sh <IP>              # or: BOARD_IP=<IP> ./install.sh

cd ../yolo_face
./install.sh <IP>
```

Useful installer options:

```sh
MODELS_DIR=~/models/my-detector ./install.sh <IP>  # upload your .cvimodel files
./install.sh --build <IP>                          # cross-compile locally instead
RELEASE=v1.0.1 ./install.sh <IP>                   # pin a release
PAYLOAD_DIR=/path/to/unpacked-package ./install.sh <IP>
```

The installer stops an old stream/face process, copies the package to
`/root/yolo_stream` or `/root/yolo_face`, verifies the checksums, copies the
vendor libraries from `/mnt/system/lib`, `/mnt/system/usr/lib` and
`/mnt/system/usr/lib/3rd` into `$DEST/runtime`, uploads every `.cvimodel` and
`.names` file found under `MODELS_DIR` (or `vendor/models`), and leaves the
camera configuration alone.

### Lichee YOLO Studio

Both Studio variants need no Python and no installer:

- **Linux slim:** unpack `yolo_studio-linux-x86_64.tar.gz`, run
  `./LicheeYOLOStudio`. Built on Ubuntu 22.04, needs glibc 2.35 or newer.
- **Linux full:** unpack `yolo_studio-linux-x86_64-full.tar.gz`, run
  `LicheeYOLOStudio-full/LicheeYOLOStudio-full`; training and ONNX export work
  out of the box.
- **Windows slim / full:** same archives, `LicheeYOLOStudio.exe` or
  `LicheeYOLOStudio-full\LicheeYOLOStudio-full.exe`. If SmartScreen appears,
  choose **More info → Run anyway**.

On Windows, `tpu-mlir` runs inside **WSL 2**: the Studio creates the
environment there, calls `wsl.exe` for the `.cvimodel` step and translates the
paths automatically. Install WSL 2 once with `wsl --install` if it is not
present; after that no Python, `tpu-mlir`, `sudo` or `apt` setup is required —
the Studio downloads a private Python 3.10 with `uv` when the distribution does
not have one. The ONNX file itself is portable.

The Qt X11 library `libxcb-cursor.so.0` is bundled in the Linux builds. The
OpenGL/EGL stack comes from the system (`libegl1`, `libgl1`, `libxkbcommon0`,
`libfontconfig1`, `libdbus-1-3` on Debian/Ubuntu). A normal desktop already has
them; a minimal image does not, and the executable then prints the exact
package names to install. See [`yolo_studio/README.md`](yolo_studio/README.md)
for details and the Wayland fallback.

## Vendor libraries and models

`yolo_stream` and `yolo_face` link against the Cvitek/Sophgo runtime and load
`.cvimodel` files on the board. Neither is published here. If you build
locally or want a self-contained archive for your own use, collect them with:

```sh
scripts/fetch-vendor.sh --board <IP>       # from a running board (best source)
scripts/fetch-vendor.sh --sdk ~/sdk-dir    # from an SDK/firmware you unpacked
scripts/fetch-vendor.sh --models ~/models  # your own exported models
scripts/fetch-vendor.sh --download         # the parts Sophgo publishes on GitHub
```

The script fills `vendor/` (ignored by git):

```
vendor/lib/*.so                 vendor libraries used at run time
vendor/models/*.cvimodel        models
vendor/cviruntime/include/*.h   headers needed to build
vendor/cviruntime/lib/*.so      libcviruntime/libcvikernel needed to build
```

`scripts/build.sh` finds `vendor/cviruntime` automatically, and
`install.sh` uploads `vendor/models` when `MODELS_DIR` is not set. The `--board`
and `--sdk` modes also copy the two link-time libraries, which are only
distributed inside the firmware.

To create a private, self-contained board archive (for example for an offline
installation), add `--with-vendor`:

```sh
scripts/package-board.sh yolo_stream yolo_stream/build/bin/stream_yolo \
    out --with-vendor vendor
```

Do not publish that archive: it contains vendor files.

## Installing on the board

The board address is not fixed (USB/RNDIS, Wi-Fi and Ethernet use different
addresses). The installers ask for the IP and support both a local clone and a
standalone download:

```sh
cd yolo_stream && ./install.sh              # asks for the IP
# or
BOARD_IP=192.0.2.100 ./install.sh
```

Without cloning the repository:

```sh
curl -fsSLO https://raw.githubusercontent.com/mclovin4k/lichee-rv-nano-YOLO/master/yolo_stream/install.sh
sh install.sh
curl -fsSLO https://raw.githubusercontent.com/mclovin4k/lichee-rv-nano-YOLO/master/yolo_face/install.sh
sh install.sh
```

The camera is configured by the stock image through `/mnt/data/sensor_cfg.ini`
(sensor I2C address and MIPI `lane_id`). The scripts do not change pinmux and
do not write to `/dev/mem`.

- Stream: `http://<IP>:7777/` (VLC/ffplay)
- Stream log: `/root/yolo_stream/log/stream.log`
- Face log: `/root/yolo_face/log/face.log`

Nothing is started automatically. Use `start_stream.sh` or
`start_face.sh start` after installation.

## Building

Each board project is built with its own scripts; shared dependencies (RISC-V
musl toolchain, SDK, cviruntime headers, OpenCV) are downloaded into a local
`.deps/`:

```sh
cd yolo_stream
./scripts/fetch-deps.sh
./scripts/build-opencv.sh
./scripts/build.sh
```

The same applies to `yolo_face`. `scripts/fetch-deps.sh` downloads
[musl-cross 20261001](https://github.com/musl-cross/musl-cross/releases/tag/20261001)
(`riscv64-unknown-linux-musl`, SHA256 pinned), the
[milkv-duo/cvitek-tdl-sdk-sg200x](https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x)
SDK (headers plus the prebuilt `libvenc`, `libsys` and TPU libraries) and the
[cviruntime](https://github.com/sophgo/cviruntime) headers; OpenCV is
[opencv-mobile v36](https://github.com/nihui/opencv-mobile/releases/tag/v36)
(4.13.0, Apache-2.0) with the CVI backend patch. Exact versions and commits
are pinned in `scripts/fetch-deps.sh`.

To package a board bundle locally:

```sh
scripts/package-board.sh yolo_stream yolo_stream/build/bin/stream_yolo out
```

The result is `out/yolo_stream-board.tar.gz` and its `.sha256` file — the same
slim archive that is published. The script accepts `yolo_face` and
`yolo_face/build/bin/face_stream` as well.

### Building on GitHub

`.github/workflows/build.yml` builds on every push:

- `stream_yolo` and `face_stream` from upstream dependencies only;
- Lichee YOLO Studio: a slim one-file executable and a full one-folder bundle
  (with ultralytics and torch) for Linux and Windows;
- a release for every `v*` tag with all six packages and checksums.

1. open the **Actions** tab → **build**;
2. click **Run workflow** → **Run workflow**;
3. download the `yolo_stream-board`, `yolo_face-board`,
   `yolo_studio-linux`, or `yolo_studio-windows` artifact.

Cross-compilation, OpenCV, PyInstaller, the offscreen smoke test and the
`--have-ml` check of the full Studio build run on GitHub; no local toolchain is
needed. To publish a release, push a version tag:

```sh
git tag v1.0.1
git push origin v1.0.1
```

## Things to know

- **Custom datasets are supported.** Lichee YOLO Studio can import arbitrary
  images (JPEG/PNG/BMP), optionally resize them, and pick up YOLO labels,
  `data.yaml`, or `classes.txt` if present. See the
  [Studio guide](yolo_studio/README.md#importing-an-existing-dataset).
- **No models are shipped with this repository.** Export your own with the
  Studio, use the models already present on the board
  (`/mnt/system/usr/bin/ai/models`), or fetch the SDK examples with
  `scripts/fetch-vendor.sh --download`. See
  [`models/README.md`](models/README.md).
- **Third-party licenses** are described in
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), with full texts in
  `LICENSES/`. The repository redistributes none of them; what you build or
  export yourself keeps its own license (Ultralytics derivatives are
  AGPL-3.0).
- **No autostart is installed.** The board is free after boot; the stream or
  face mode is started manually.
- **The stock firmware uses `root:root` over SSH.** Change the password and
  keep the board on a trusted network; Lichee YOLO Studio stores the SSH
  credentials in its local config file.

## License

The original code in this repository is distributed under the MIT license —
see [`LICENSE`](LICENSE). Third-party components remain under their own
licenses; their terms and sources are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and `LICENSES/`. The MIT
license does not replace the license of anything you download, build or
export yourself.
