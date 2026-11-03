# Third-party components

This repository contains **only original code** (MIT, see `LICENSE`). It does
not redistribute vendor binaries, vendor headers or converted models: the
Cvitek/Sophgo runtime and the `.cvimodel` files are proprietary or carry
licenses that do not allow us to publish them.

Everything third-party is either downloaded by the user at build time, copied
from the user's own board, or installed from the Python package index. This
file lists those components, their licenses and where they come from. Full
license texts are in `LICENSES/`.

Snapshot date: 2026-11-02.

## What is in the repository

| Category | Contents | License |
|---|---|---|
| Board applications | `yolo_stream/`, `yolo_face/` sources and scripts | MIT |
| Desktop studio | `yolo_studio/` sources | MIT |
| Camera patch | `*/patches/opencv-mobile-4.13.0-cvi-capture.patch` | Apache-2.0 (derivative of opencv-mobile) |
| Docs, CI | `README.md`, `docs/`, `.github/workflows/` | MIT |

No `.so`, `.cvimodel`, `.ko`, `.a` or other vendor binary is tracked in git,
and none is uploaded to the releases. `vendor/`, `*/dist/board/runtime/` and
`*/third_party/cviruntime/` are ignored by git on purpose.

## Vendor libraries and models: how they get to your board

`scripts/fetch-vendor.sh` collects the files the applications need from a
source **you** control, on your machine:

| Source | Command | Notes |
|---|---|---|
| Your board over SSH | `scripts/fetch-vendor.sh --board <IP>` | Copies the libraries that the stock firmware already ships (`/mnt/system/lib`, `/mnt/system/usr/lib`, `/mnt/system/usr/lib/3rd`) and the models present on the board |
| A vendor SDK/image you downloaded | `scripts/fetch-vendor.sh --sdk <dir>` | Same files from an extracted `cvitek-tdl-sdk-sg200x` checkout or a firmware tree |
| Your own models | `scripts/fetch-vendor.sh --models <dir>` | `.cvimodel` files you exported with `yolo_studio` |
| Upstream Sophgo/Milk-V repositories | `scripts/fetch-vendor.sh --download` | The parts Sophgo publishes on GitHub: middleware and TPU libraries, `cviruntime` headers, the SDK example models |

`yolo_stream/install.sh` and `yolo_face/install.sh` repeat the board step
during installation: they copy the vendor libraries from the board firmware
into `$DEST/runtime` and upload every model found under `MODELS_DIR` (or under
`vendor/models`). Nothing vendor-specific is transferred between machines by
this project.

## Components downloaded at build time

| Component | Source | License | Used for |
|---|---|---|---|
| opencv-mobile 4.13.0 (v36) | [nihui/opencv-mobile](https://github.com/nihui/opencv-mobile) | Apache-2.0 | static OpenCV with the CVI camera backend |
| OpenCV 4.13.0 (inside opencv-mobile) | [opencv.org](https://opencv.org/releases/) | Apache-2.0 | linked into `stream_yolo`, `face_stream` |
| cvitek-tdl-sdk-sg200x | [milkv-duo](https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x) @ `02dfb90` | **no license file** | middleware headers, `libvenc.so`, `libsys.so`, prebuilt `libcviruntime.so`/`libcvikernel.so` at build time |
| cviruntime headers | [sophgo/cviruntime](https://github.com/sophgo/cviruntime) @ `ef80449` | **no license file** | `cviruntime.h`, `cvitpu_debug.h` |
| RISC-V musl toolchain | [musl-cross 20261001](https://github.com/musl-cross/musl-cross/releases/tag/20261001) | GPL-3.0 / LGPL-2.1 / MIT + exceptions | cross compiler (build tool only) |
| tpu-mlir | [sophgo/tpu-mlir](https://github.com/sophgo/tpu-mlir) | Apache-2.0 | conversion to `.cvimodel` (installed by the user) |

The SDK and `cviruntime` files are fetched into `.deps/` or `vendor/` and are
used **only on the machine that builds**; they are not committed and not part
of any release archive. If you build with `--with-vendor`, the resulting
archive is for your own use — do not publish it.

## Models

The repository ships **no model weights**. There are three ways to get
`.cvimodel` files:

1. **Export your own** with `yolo_studio` → training → ONNX → tpu-mlir. If the
   starting point is an Ultralytics checkpoint (YOLOv8/YOLO11), the weights
   and the converted model are **AGPL-3.0** (`LICENSES/AGPL-3.0.txt`):
   publishing them means publishing the corresponding source (a `.pt` file is
   not source; provide the training data, configuration and scripts).
2. **Use the models already on the board** —
   `/mnt/system/usr/bin/ai/models`. They come with the vendor firmware; the
   project does not redistribute them.
3. **Use the SDK example models** fetched by `scripts/fetch-vendor.sh
   --download` (`yolov5_cv181x_int8_*.cvimodel`, `scrfd_768_432_int8_1x`).
   The SDK has **no license file**; check with Sophgo before redistributing
   them.

Face models (`scrfd_det_face_*`, `feature_cviface_*`,
`cls_rgbliveness_*`) come from the Cvitek TDL SDK examples. The SCRFD
architecture is MIT (InsightFace), the weights have no stated license.

## yolo_studio dependencies

Installed from PyPI by `requirements.txt` (not vendored):

| Package | License |
|---|---|
| PySide6 / Qt 6 | LGPL-3.0 (or GPL-3.0 / commercial) |
| opencv-python-headless | Apache-2.0 |
| numpy | BSD-3-Clause |
| paramiko | LGPL-2.1 |
| PyYAML | MIT |

The release archives include `LGPL-3.0.txt` and this file. PySide6 is used
under the LGPL: the libraries are shipped as separate, replaceable files in
the one-folder build, and the sources are available from
[qt.io](https://www.qt.io/) and the PySide6 wheel on PyPI. The one-file build
bundles them for convenience; the same sources and the build script
(`tools/build_desktop.py`) are published here, so the libraries can be rebuilt
or replaced.

### The `-full` Studio bundle

The `yolo_studio-*-full` archives additionally contain:

| Component | License | Source |
|---|---|---|
| ultralytics | AGPL-3.0 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| PyTorch, torchvision | BSD-3-Clause | [pytorch.org](https://pytorch.org/) |
| onnx, onnxruntime | Apache-2.0 / MIT | [onnx.ai](https://onnx.ai/) |
| onnxslim | MIT | [inisis/OnnxSlim](https://github.com/inisis/OnnxSlim) |

The Studio calls the ultralytics command line as a separate process
(`--yolo-cli`), so the application and the model tooling are separate
programs that are merely distributed together. The AGPL-3.0 text is in
`LICENSES/AGPL-3.0.txt`; the exact versions are listed in the archive's
`THIRD_PARTY_NOTICES.md`. If you modify ultralytics and redistribute the
bundle, you must publish your modified sources under the AGPL.

## Runtime libraries on the board

The board applications are musl binaries and use the loader, libc and vendor
libraries of the firmware. `run_stream.sh` / `run_face.sh` put
`$DEST/runtime` (the copies made by `install.sh`) in front of
`/mnt/system/usr/lib`, `/mnt/system/usr/lib/3rd` and `/mnt/system/lib`.
Nothing from those directories is published by this project.

## Background and references

`stream_yolo` was inspired by
[ret7020/LicheeRVNano](https://github.com/ret7020/LicheeRVNano), which has
**no license file**; its code was not copied. The current implementation is a
rewrite, and the upstream project is credited as the source of the idea and
of useful CV181x examples.

## Warranty

Third-party components are provided **without warranty**, "as is", by their
respective authors. This project was tested only on a LicheeRV Nano
(SG2002/CV181x) with the GC4653 camera and specific firmware versions. The
rights holders are not affiliated with this repository and do not support it.
