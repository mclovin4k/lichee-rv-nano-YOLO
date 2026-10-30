# Third-party components, authors and licenses

This file describes the third-party code, binaries and models that are
included in the repository or downloaded during the build. Full license texts
are in `LICENSES/`. Components without an explicit license are called out
separately: such files are distributed "as is", and their terms must be
clarified with the rights holder before commercial use.

Snapshot date: 2026-10-28.

## Summary

| Component | Author / source | License | Location |
|---|---|---|---|
| opencv-mobile 4.13.0 (v36) | nihui, OpenCV team | Apache-2.0 | downloaded by `scripts/fetch-deps.sh` |
| OpenCV 4.13 (inside opencv-mobile) | OpenCV team | Apache-2.0 | statically linked into `stream_yolo`, `face_stream` |
| CVI backend patch | derivative of opencv-mobile | Apache-2.0 | `yolo_stream/patches/`, `yolo_face/patches/` |
| OpenCV 3.2 (shared) | OpenCV team | BSD-3-Clause | `yolo_face/dist/board/runtime/libopencv_*.so.3.2` |
| cvitek-tdl-sdk-sg200x | milkv-duo / Sophgo | no license file | headers, `libvenc.so`, `libsys.so` during the build |
| cviruntime / cvikernel | Sophgo | no license file | `*/third_party/cviruntime/` |
| libcvi_tdl.so | Cvitek / Sophgo TDL SDK | no license file | `yolo_face/dist/board/runtime/` |
| libcvi_ive.so, libcnpy.so, libcvimath.so | Cvitek / Sophgo TDL SDK | no license file | `yolo_face/dist/board/runtime/` |
| libvpu_tdl.so (patched middleware) | Cvitek / Sophgo | no license file | `yolo_face/dist/board/runtime/` |
| middleware (ISP/VPSS/VENC/…) | Cvitek / Sophgo | no license file | `yolo_stream/dist/board/runtime/`, `yolo_face/.../runtime/` |
| libz 1.2.11 | Jean-loup Gailly, Mark Adler | Zlib | `yolo_face/.../runtime/libz.so.1.2.11` |
| musl loader | Rich Felker and musl contributors | MIT | `*/dist/board/runtime/ld-musl-riscv64.so.1` |
| libstdc++, libgcc, libatomic | GCC / FSF | GPL-3.0 + GCC Runtime Library Exception | `*/dist/board/runtime/` |
| YOLOv8n models (6) | derivatives of Ultralytics YOLOv8n | AGPL-3.0 | `yolo_stream/dist/board/models/` |
| YOLO11n drone model | derivative of Ultralytics YOLO11n | AGPL-3.0 | `yolo_stream/dist/board/models/yolov11n_det_drone_640_640_INT8_cv181x.cvimodel` |
| SCRFD face detector | InsightFace (architecture), Cvitek TDL SDK (weights) | code MIT; weights without an explicit license | `yolo_face/dist/board/models/face/` |
| cviface / liveness | Cvitek TDL SDK | no license file | `yolo_face/dist/board/models/face/` |
| ret7020/LicheeRVNano | ret7020 | no license file; code rewritten | README reference only |
| PySide6 | Qt Company / Qt contributors | LGPL-3.0 / GPL-3.0 / commercial | `yolo_studio` dependency (not included) |
| opencv-python-headless | OpenCV team | Apache-2.0 | `yolo_studio` dependency |
| numpy | NumPy developers | BSD-3-Clause | `yolo_studio` dependency |
| paramiko | Jeff Forcier and contributors | LGPL-2.1 | `yolo_studio` dependency |
| PyYAML | Kirill Simonov and contributors | MIT | `yolo_studio` dependency |
| Ultralytics (ultralytics) | Ultralytics | AGPL-3.0 | external `ultra` environment |
| tpu-mlir | Sophgo | Apache-2.0 | external `tpu310` environment |
| soph_tpu.ko | Sophgo | GPL-2.0 | **not included** in the repository; patched on the board |

## OpenCV and the CVI camera backend

`opencv-mobile-4.13.0.zip` is the official
[nihui/opencv-mobile v36](https://github.com/nihui/opencv-mobile/releases/tag/v36)
release (archive `opencv-mobile-4.13.0.zip`, SHA256
`0aeaea4837d19894765aae23cb82fc4b87cec07449bee4b4404a63bdda15d6ee`).
The project is distributed under Apache-2.0; it contains OpenCV 4.13.0
(Apache-2.0).

The `patches/opencv-mobile-4.13.0-cvi-capture.patch` patches modify
`modules/highgui/src/capture_cvi.cpp` and `videocapture.cpp` from the same
archive, so they are derivatives and are distributed under Apache-2.0. The
patch adds a raw NV21 mode (`OPENCV_CVI_RAW`), reads the sensor geometry from
`/mnt/data/sensor_cfg.ini`, recovers a stuck VPSS group, and supports the
required frame sizes.

`libopencv_core.so.3.2`, `libopencv_imgproc.so.3.2` and
`libopencv_imgcodecs.so.3.2` in the `yolo_face` runtime are a shared build of
OpenCV 3.2.0. OpenCV 3.2 itself is distributed under BSD-3-Clause (text in
`LICENSES/BSD-3-Clause.txt`). These are prebuilt libraries from the
Cvitek/TDL SDK package; OpenCV 3.2 sources are available at
[opencv.org](https://opencv.org/releases/). They are needed only by
`libcvi_tdl.so`; the `face_stream` application is built with static OpenCV
4.13.

## Cvitek / Sophgo SDK and vendor libraries

- [milkv-duo/cvitek-tdl-sdk-sg200x](https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x),
  commit `02dfb9091d390a7b0d1177fe7b651db1b99854bc`. The middleware v2 and
  `cvi_tdl` headers are used, along with `libvenc.so`/`libsys.so`; they are
  downloaded by `scripts/fetch-deps.sh` (sparse checkout). **There is no
  license file in the repository.**
- [sophgo/cviruntime](https://github.com/sophgo/cviruntime) — the
  `cviruntime.h` and `cvitpu_debug.h` headers and the `libcviruntime.so` and
  `libcvikernel.so` libraries (`third_party/cviruntime/`). **There is no
  license file in the repository.**
- Cvitek/Sophgo binaries in `dist/board/runtime/`: `libcvi_tdl.so`,
  `libcvi_ive.so`, `libcnpy.so`, `libcvimath.so`, `libvpu_tdl.so`,
  `libisp.so`, `libisp_algo.so`, `libae.so`, `libawb.so`, `libvenc.so`,
  `libsys.so`, `libsns_gc4653.so`, `libcvi_bin.so`, `libcvi_bin_isp.so`.
  This is the vendor runtime for cv181x/SG2002; **there is no explicit
  license in the source**, and the files are included to run on the
  corresponding board "as is", without warranty. `libvpu_tdl.so` is a renamed
  patched build of the TDL SDK middleware `libvpu.so` (TDL needs it for GDC
  symbols); the system `libvpu.so` is not included and is taken from the
  board.
- `libcvi_tdl.so` comes from the TDL SDK, md5
  `5284ac664ef5330bacbac1fdb6de70ea`. It is built against shared OpenCV 3.2
  and loaded via `dlopen`, so `libopencv_*.so.3.2`, `libcnpy.so`,
  `libcvi_ive.so` and `libz` are placed next to it.
- `libz.so.1.2.11` is zlib 1.2.11 under the Zlib license
  (`LICENSES/Zlib.txt`); it is used by `libcvi_tdl.so`/`libcnpy.so`.

## Models

### YOLOv8n / YOLO11n (detection)

The six `yolov8n_det_*` models are INT8 conversions of Ultralytics YOLOv8n
networks (COCO-80, person/vehicle, helmet, traffic light, license plate,
fire/smoke). The `yolov11n_det_drone_640_640` model is a derivative of
Ultralytics YOLO11n (1 class), fine-tuned by the repository author; the
package contains one version, md5 `d479ed5849723b9137d71fd18cc3d4ba`.

Ultralytics YOLO is distributed under **AGPL-3.0**
(`LICENSES/AGPL-3.0.txt`). We also consider the derivative weights and the
converted `.cvimodel` files to be AGPL-3.0. If you distribute these files
(including as part of firmware or a closed-source product), you must comply
with the AGPL: provide recipients with the corresponding source code (the
preferred form is `.pt`/ONNX plus the conversion scripts) and keep the
notices. The original YOLOv8n/YOLO11n weights are available from Ultralytics;
the conversion pipeline is in `yolo_studio/tools/` (`export_yolo11.py`,
`build_cvimodel.sh`). Datasets and user frames are not part of the
repository.

### Faces (SCRFD, cviface, liveness)

`scrfd_det_face_432_768_INT8_cv181x.cvimodel` is a face detector based on the
SCRFD architecture. The architecture and InsightFace code are distributed
under MIT (`LICENSES/MIT.txt`); the weights in this package come from the
Cvitek TDL SDK examples (the SDK contains
`cvimodel/scrfd_768_432_int8_1x.cvimodel`). The distribution terms for the
weights are not stated in the source.

`feature_cviface_112_112_INT8_cv181x.cvimodel` (face embeddings) and
`cls_rgbliveness_256_256_INT8_cv181x.cvimodel` (liveness) are Cvitek TDL SDK
models. **There is no explicit license in the source.** They are included
because `yolo_face` needs them; they are distributed "as is", without
warranty. Clarify the rights with Cvitek/Sophgo before commercial use or
redistribution.

## System libraries in the runtime

- `ld-musl-riscv64.so.1` — musl libc, MIT (`LICENSES/MIT.txt`). It is bundled
  with the board release packages and used by default so the loader, `libc`
  and the binary always come from the same toolchain. Set
  `SY_BUNDLED_LOADER=0` to use the system `/lib/ld-musl-riscv64.so.1`.
- `libstdc++.so.6`, `libgcc_s.so.1`, `libatomic.so.1` — GCC runtime libraries
  taken from the same toolchain as the board binary. Licensed under GPL-3.0
  with the GCC Runtime Library Exception (`LICENSES/GPL-3.0.txt`,
  `LICENSES/GCC-Runtime-Library-Exception.txt`). The exception allows
  distributing programs linked against these libraries under any terms.

## Build tools

`scripts/fetch-deps.sh` downloads the RISC-V musl toolchain from GitHub:
[musl-cross 20261001](https://github.com/musl-cross/musl-cross/releases/tag/20261001),
`riscv64-unknown-linux-musl.tar.xz` (SHA256
`054acc4b5baa886363ce5acaa6a015fd90d2f715b2e4324813fd4d49d717362f`).
It contains GCC, binutils and musl, distributed under GPL-3.0/LGPL-2.1/MIT
with the corresponding exceptions. The toolchain is not part of the
repository and is used only at build time. The `musl.cc`
`riscv64-linux-musl-cross` (GCC 11.2.1) toolchain is also supported via the
`TOOLCHAIN_URL` variable.

## yolo_studio (Python)

`yolo_studio/requirements.txt` installs dependencies from PyPI; their sources
are not included in the repository:

- PySide6 — LGPL-3.0 / GPL-3.0 / commercial Qt license
  (`LICENSES/LGPL-3.0.txt`);
- opencv-python-headless — Apache-2.0;
- numpy — BSD-3-Clause;
- paramiko — LGPL-2.1 (`LICENSES/LGPL-2.1.txt`);
- PyYAML — MIT.

The README also recommends external environments:

- Ultralytics (`ultra`) — AGPL-3.0, used for training/export;
- tpu-mlir (`tpu310`) — Apache-2.0, used to build `.cvimodel` files.

These packages are not part of the repository and are installed by the user
separately.

## tpu-fix and the GPL module

`tpu-fix/patch_tpu.sh` changes a few bytes in the `soph_tpu.ko` module
**already installed on the board**. The module itself is part of the Sophgo
firmware and is distributed under GPL-2.0 (`LICENSES/GPL-2.0.txt`). The
repository contains neither the `.ko` nor the driver sources: the patcher
only changes the timeout in the binary on the user's device and creates a
backup. The script does not link against the module and is not a derivative
of the driver.

## Background and references

The `stream_yolo` project was originally inspired by
[ret7020/LicheeRVNano](https://github.com/ret7020/LicheeRVNano). That
repository **has no license file**, so its code was not copied as is: the
current implementation is rewritten, and the upstream project is credited as
the source of the idea and of useful CV181x examples. If you reuse upstream
code, clarify the terms with the author first.

## Warranty

All third-party binaries and models are provided **without warranty**, "as
is". Their functionality was tested only on the LicheeRV Nano (SG2002/CV181x)
with the GC4653 camera and a specific firmware. The rights holders are not
affiliated with this repository and do not support it.
