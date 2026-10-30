# LicheeRV Nano: camera, NPU and face recognition

A working toolkit for the LicheeRV Nano (SG2002 / CV181x) with the GC4653
camera: MJPEG streaming, YOLO detection on the NPU, face recognition, and a
desktop tool for capturing datasets, labeling and building models. The
repository contains four independent projects.

Ready-to-run release packages include the board binaries, models and matching
runtime libraries, plus desktop builds of Lichee YOLO Studio for Linux and
Windows. Cross-compilation is not needed to run anything on the board.

| Project | What it is | Quick start |
|---|---|---|
| [`yolo_stream/`](yolo_stream/README.md) | MJPEG 960×540 + YOLOv8/YOLO11 on the NPU, 7 models, port 7777 | `yolo_stream/install.sh <IP>` → `start_stream.sh auto` |
| [`yolo_face/`](yolo_face/README.md) | MJPEG + face detection/recognition (SCRFD + cviface) | `yolo_face/install.sh <IP>` → `start_face.sh start` |
| [`yolo_studio/`](yolo_studio/README.md) | Desktop tool: dataset import/capture, labeling, training, `.cvimodel` export | download the Linux/Windows build from Releases |
| [`tpu-fix/`](tpu-fix/README.md) | TPU timeout patcher (`soph_tpu.ko`) without redistributing the `.ko` | `tpu-fix/patch_tpu.sh` on the board |

`yolo_stream` and `yolo_face` share the camera and port 7777, so starting one
stops the other. No autostart is installed: after boot you choose the mode
manually.

## Ready-to-run releases

Download the latest files from
[GitHub Releases](https://github.com/mclovin4k/lichee-rv-nano-YOLO/releases/latest):

| Asset | Contents |
|---|---|
| `yolo_stream-board.tar.gz` | `stream_yolo`, models, start scripts, vendor libraries, musl loader and libc |
| `yolo_face-board.tar.gz` | `face_stream`, face models, start scripts, vendor libraries, musl loader and libc |
| `yolo_studio-linux-x86_64.tar.gz` | one-file Lichee YOLO Studio executable for Linux x86_64 |
| `LicheeYOLOStudio.exe` | one-file Lichee YOLO Studio executable for Windows x86_64 |
| `*.sha256`, `SHA256SUMS.txt` | checksums for every asset |

The board installers download the latest release themselves, verify its
SHA256, unpack it over SSH and check the package checksums on the board:

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
RELEASE=v1.0.0 ./install.sh <IP>              # pin a release
PAYLOAD_DIR=/path/to/unpacked-package ./install.sh <IP>
TPU_PATCH=1 ./install.sh <IP>                 # also apply the TPU timeout fix
```

The board package is self-contained: `run_stream.sh` / `run_face.sh` set
`LD_LIBRARY_PATH` and use the bundled musl loader by default, so the matching
libc, `libgcc`, `libatomic` and vendor libraries are always found.

### Lichee YOLO Studio

- **Linux:** unpack
  `yolo_studio-linux-x86_64.tar.gz`, then run
  `./LicheeYOLOStudio`. The archive is built on Ubuntu 22.04 and needs glibc
  2.35 or newer (Ubuntu 22.04+, Debian 12+, Fedora 36+, and similar).
- **Windows:** run `LicheeYOLOStudio.exe`. There is no installer; if
  SmartScreen appears, choose **More info → Run anyway**.

The Qt X11 library `libxcb-cursor.so.0` is bundled in the Linux release. The
OpenGL/EGL stack comes from the system (`libegl1`, `libgl1`, `libxkbcommon0`,
`libfontconfig1`, `libdbus-1-3` on Debian/Ubuntu). A normal desktop already has
them; a minimal image does not, and the executable then prints the exact
package names to install. See [`yolo_studio/README.md`](yolo_studio/README.md)
for details and the Wayland fallback.

## Installing on the board

The board address is not fixed (USB/RNDIS, Wi-Fi and Ethernet use different
addresses). The installers ask for the IP, stop an old stream/face process,
copy the package to `/root/yolo_stream` or `/root/yolo_face`, verify checksums,
and leave the camera configuration alone:

```sh
cd yolo_stream && ./install.sh              # asks for the IP
# or
BOARD_IP=192.0.2.100 ./install.sh
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
musl toolchain, SDK, OpenCV) are downloaded into a local `.deps/`:

```sh
cd yolo_stream
./scripts/fetch-deps.sh
./scripts/build-opencv.sh
./scripts/build.sh
```

The same applies to `yolo_face`. By default, `scripts/fetch-deps.sh`
downloads
[musl-cross 20261001](https://github.com/musl-cross/musl-cross/releases/tag/20261001)
(`riscv64-unknown-linux-musl`, SHA256 pinned); `musl.cc`
`riscv64-linux-musl-cross` (GCC 11.2.1) and Sipeed/Sophgo toolchains are also
supported. SDK —
[milkv-duo/cvitek-tdl-sdk-sg200x](https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x),
OpenCV — the official [opencv-mobile v36](https://github.com/nihui/opencv-mobile/releases/tag/v36)
(4.13.0, Apache-2.0) with the CVI backend patch. Exact versions and commits
are pinned in `scripts/fetch-deps.sh`.

To build a complete, self-contained board package locally (binary + models +
vendor libraries + matching musl loader/libc):

```sh
scripts/package-board.sh yolo_stream yolo_stream/build/bin/stream_yolo \
    yolo_stream/.deps/riscv64-linux-musl-cross out
```

The result is `out/yolo_stream-board.tar.gz` and its `.sha256` file. The same
script accepts `yolo_face` and `yolo_face/build/bin/face_stream`.

### Building on GitHub

The repository includes `.github/workflows/build.yml`. On every push it builds
`stream_yolo`, `face_stream`, and the Linux/Windows Studio executables, then
uploads them as artifacts. A tag such as `v1.0.0` additionally publishes a
GitHub Release with all four ready-to-run packages and checksums.

1. open the **Actions** tab → **build**;
2. click **Run workflow** → **Run workflow**;
3. download the `yolo_stream-board`, `yolo_face-board`,
   `yolo_studio-linux`, or `yolo_studio-windows` artifact.

Cross-compilation, OpenCV, PyInstaller and smoke tests run on GitHub; no local
toolchain is needed. To publish a release, push a version tag:

```sh
git tag v1.0.0
git push origin v1.0.0
```

## Things to know

- **Custom datasets are supported.** Lichee YOLO Studio can import arbitrary
  images (JPEG/PNG/BMP), optionally resize them, and pick up YOLO labels,
  `data.yaml`, or `classes.txt` if present. See the
  [Studio guide](yolo_studio/README.md#importing-an-existing-dataset).
- **User datasets and frames are not part of the repository.** Only
  ready-made `.cvimodel` files are included; images, videos and labels stay
  local.
- **The TPU fix is a patcher, not a GPL binary.** `tpu-fix/patch_tpu.sh`
  changes the timeout in the module already installed on the board, creates a
  backup and can roll back. Neither driver sources nor a `.ko` are included in
  the repository.
- **Third-party model and library licenses** are described in
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), with full texts in
  `LICENSES/`. Some vendor libraries from Cvitek/Sophgo ship without an
  explicit license; the YOLO models are derivatives of Ultralytics
  (AGPL-3.0). Read the notices before commercial use.
- **No autostart is installed.** The board is free after boot; the stream or
  face mode is started manually.

## License

The original code in this repository is distributed under the MIT license —
see [`LICENSE`](LICENSE). Third-party libraries, runtimes and models remain
under their own licenses; their terms and sources are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and `LICENSES/`. The MIT
license does not replace the licenses of third-party components.
