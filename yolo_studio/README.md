# Lichee YOLO Studio

A desktop tool for **Linux and Windows** that covers the complete lifecycle of
a custom YOLO model for the **LicheeRV Nano (cv181x)**:

1. **Dataset import or capture** — import existing images, or capture frames
   from the board camera with guided angle steps.
2. **Labeling** — draw a box on a keyframe; the tool propagates it to the
   following frames with optical flow (semi-automatic).
3. **Training** — locally on CPU (slow) or by importing a ready-made `.pt`.
4. **Build** — export ONNX with six "raw" outputs → INT8 calibration with
   tpu-mlir → `.cvimodel` for cv181x.
5. **Integration** — upload the model to the board over SSH into
   `/root/models/` and append it to the `start_stream.sh` menu.

## Ready-to-run builds

The easiest option is a release build from
[GitHub Releases](https://github.com/mclovin4k/lichee-rv-nano-YOLO/releases/latest);
Python and Qt do not need to be installed. Each platform has two variants:

| Asset | Size | Training / ONNX export |
|---|---|---|
| `yolo_studio-<os>-x86_64.tar.gz` | ~150 MB | needs a separate Python with `ultralytics` |
| `yolo_studio-<os>-x86_64-full.tar.gz` | ~600–900 MB | works out of the box — torch and ultralytics are inside |

The full variant is a folder, not a single file: unpack it and run
`LicheeYOLOStudio-full/LicheeYOLOStudio-full.exe` (Windows) or
`LicheeYOLOStudio-full/LicheeYOLOStudio-full` (Linux). It starts without
extracting anything, so startup is as fast as the slim build. The single-file
executable stays small because it does not carry PyTorch.

**Linux x86_64**

```sh
curl -fLO https://github.com/mclovin4k/lichee-rv-nano-YOLO/releases/latest/download/yolo_studio-linux-x86_64.tar.gz
curl -fLO https://github.com/mclovin4k/lichee-rv-nano-YOLO/releases/latest/download/yolo_studio-linux-x86_64.tar.gz.sha256
sha256sum -c yolo_studio-linux-x86_64.tar.gz.sha256
tar -xzf yolo_studio-linux-x86_64.tar.gz
chmod +x LicheeYOLOStudio
./LicheeYOLOStudio
```

The Linux build is made on Ubuntu 22.04 and needs glibc 2.35 or newer
(Ubuntu 22.04+, Debian 12+, Fedora 36+, and similar). Qt's X11 plugins and the
`libxcb-cursor.so.0` dependency are bundled, but the OpenGL/EGL stack is taken
from the system. A normal desktop install already has it; on a minimal image
install:

```sh
# Debian/Ubuntu
sudo apt install libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3
# Fedora
sudo dnf install mesa-libEGL mesa-libGL libxkbcommon fontconfig dbus-libs
# Arch
sudo pacman -S mesa libxkbcommon fontconfig dbus
```

The executable checks these libraries at startup and prints the exact package
names when one is missing. If the xcb platform plugin still cannot load, the
X11 client libraries are missing as well:

```sh
# Debian/Ubuntu
sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
    libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libxcb-util1
# Fedora
sudo dnf install xcb-util-cursor libxkbcommon-x11
# Arch
sudo pacman -S xcb-util-cursor libxkbcommon-x11
```

On Wayland, `QT_QPA_PLATFORM=wayland ./LicheeYOLOStudio` often works when the
X11 backend does not.

**Windows x86_64**

```powershell
curl.exe -fLO https://github.com/mclovin4k/lichee-rv-nano-YOLO/releases/latest/download/yolo_studio-windows-x86_64.tar.gz
curl.exe -fLO https://github.com/mclovin4k/lichee-rv-nano-YOLO/releases/latest/download/yolo_studio-windows-x86_64.tar.gz.sha256
# verify with: Get-FileHash yolo_studio-windows-x86_64.tar.gz -Algorithm SHA256
tar -xzf yolo_studio-windows-x86_64.tar.gz
.\LicheeYOLOStudio.exe
```

There is no installer; the slim archive is a single `.exe` plus the license
files. The `-full` archive is a folder — unpack it and run
`LicheeYOLOStudio-full\LicheeYOLOStudio-full.exe`. If SmartScreen appears,
choose **More info → Run anyway**.

## Requirements

**Host (Linux or Windows):**

- for a release build: no Python installation (the `-full` variant also
  includes ultralytics and torch; the slim one needs nothing for capture,
  labeling and upload);
- for running from source: Python 3.10+ (tested on 3.13) and the packages from
  `requirements.txt` (PySide6, OpenCV, numpy, paramiko, PyYAML);
- for training: the `-full` build, or a separate environment with
  `ultralytics` + `torch` (CPU);
- for conversion: a `tpu-mlir` environment (Python 3.10) — the
  `tools/build_cvimodel.sh` script and the `TPU_PY` variable.

**Board:**

- firmware with working SSH (the default user is `root`), a camera and the
  installed `/root/yolo_stream` package (or the legacy
  `/root/start_stream.sh`);
- `stream_yolo` from this repository — with `SY_HTTP_PORT` and `SO_REUSEADDR`
  support (required for capture mode on a separate port);
- `start_stream.sh` from `../yolo_stream/dist/board/` — with
  `/root/models/models.order` support.

## Running from source

Linux:

```sh
cd yolo_studio
uv venv --python 3.13 .venv      # or: python3 -m venv .venv
. .venv/bin/activate
uv pip install -r requirements.txt
python app.py
```

`./run.sh` does the same: it selects `.venv/bin/python` when present and checks
the Linux Qt/X11 dependency before starting.

Windows (PowerShell or `cmd.exe`):

```bat
cd yolo_studio
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

The paths to the `ultra` (ultralytics) and `tpu310` (tpu-mlir) environments,
the board IP, SSH user and password are configured in **Settings…**. Both
Python fields can be left empty: the application searches `PATH` and the usual
virtualenv directories (`~/.venvs/<name>`, `~/venvs/<name>`, `.venv`) and
checks that the interpreter can import `ultralytics` / `tpu_mlir`. The **Detect**
button runs the same search and fills in the path it finds.

```sh
# ultralytics (training, ONNX export) — Linux/macOS
python3 -m venv ~/.venvs/ultra
~/.venvs/ultra/bin/python -m pip install ultralytics

:: ultralytics — Windows
py -3 -m venv %USERPROFILE%\.venvs\ultra
%USERPROFILE%\.venvs\ultra\Scripts\python -m pip install ultralytics
```

`tpu-mlir` is Linux-only, so cvimodel conversion runs on Linux (or in WSL on
Windows); the ONNX step works everywhere.

## Quick start

1. **Capture → New…** — dataset name and comma-separated class names (for
   example, `drone` or `drone, person`). The class order defines their ids:
   the first is 0, the second is 1. If the data already exists, use
   **Import images from folder…** instead (see below).
2. **Clean stream** — the tool stops the YOLO stream on the board over SSH
   and starts a stream without detection or boxes on port `7778`.
3. **Connect camera** — live video appears in the window.
4. Choose a step ("Front", "Left", …) and click **Start step**. While the
   countdown runs, frames are saved at ~4–5 per second; blurry and nearly
   identical frames are discarded.
5. After the scenario, click **Restore YOLO**: the normal stream with
   detection comes back on `:7777`.
6. **Labeling** — select the box class in the list, draw a box on the first
   frame of the episode, and click **Track forward →**. The boxes are
   propagated to the following frames; mistakes can be fixed with the mouse.
   Delete removes the selected box; changing the class recolors the selected
   box.
7. **Model** — check the class list in the **Classes** field (it can be
   edited both here and on the Capture tab), then **Prepare dataset**
   (train/val 80/20 + up to 200 frames for calibration). Next, either
   **Import ready .pt…** or **Train on CPU**, then **Export ONNX** and
   **Build cvimodel**.
8. **Board** — **Check connection**, select the `.cvimodel`, and click
   **Upload to board**. The model ends up in `/root/models/` and at the end of
   the `start_stream.sh` menu; the **Start selected** button immediately
   restarts the stream with it.

During the build, a `<name>.names` file is created next to the `.cvimodel`
(one class per line). It is uploaded together with the model, and
`stream_yolo` labels boxes with class names (`person 0.83`) instead of
`class:1`.

## Importing an existing dataset

You do not have to capture frames from the board. On the **Capture** tab:

1. create a dataset and set its class names, or open an existing one;
2. choose **Import size** — `keep` preserves the original resolution, while
   `640x480` (any `WIDTHxHEIGHT`) letterboxes images into that size with gray
   padding;
3. click **Import images from folder…** and select a folder.

The importer scans the folder recursively and accepts JPEG, PNG and BMP
images. It copies them into `images/` as `frame_XXXXX.jpg`. A YOLO `.txt`
label is imported when it is next to the image or in the sibling `labels/`
directory (the usual `images/` + `labels/` layout). Images without labels are
imported too and can be labeled in the **Label** tab. Progress and any skipped
files are shown on the page.

Class ids are remapped by name when the source folder contains a
`data.yaml`/`data.yml` or `classes.txt`/`obj.names`/`names.txt`: names are
merged into the dataset class list, so a source class 1 that is `car` remains
`car` even if it becomes id 3 in the destination. Without a class list, YOLO
ids are used as-is and the class list is extended with `class_N` names when
needed. Label coordinates are transformed together with the image when an
import size is selected.

## How it works

### Capture mode

The normal board stream draws boxes and loads the TPU. Datasets need clean
frames, so the tool starts a second mode:

```sh
SY_DET=0 SY_DRAW=0 SY_HTTP_PORT=7778 /root/yolo_stream/start_stream.sh auto
```

`SY_DET=0` disables inference, `SY_DRAW=0` disables drawing, and
`SY_HTTP_PORT` moves MJPEG to 7778 so it does not conflict with the normal
stream. Capture goes through OpenCV (`cv2.VideoCapture` MJPEG), the frames are
960×540.

### Dataset

```
<dataset>/
  dataset.json        classes, creation date
  images/             source frames (frame_00001.jpg …)
  labels/             YOLO txt files with the same names
  calib/              frames for INT8 calibration
  model/              best.pt / best.onnx / *.cvimodel / build/
  images/train|val/   split for ultralytics
  labels/train|val/
  data.yaml
```

A dataset can contain one or several classes; their order defines the id
(0, 1, …). A frame can contain several boxes, including boxes of different
classes. The class list is defined when the dataset is created and can be
edited on the Capture and Model tabs; when it changes, `dataset.json` and
`data.yaml` are updated (if the split already exists). When labeling, the
class is selected in the list on the right; tracking preserves the class of
each box.

### Tracking

`goodFeaturesToTrack` inside the box + `calcOpticalFlowPyrLK` between frames;
the translation is the median over the retained points, and the scale comes
from the spread of the points. If there are too few points or the jump is too
large, the box stays in place, and after three such frames in a row tracking
stops with a warning.

### Model build

- `tools/export_yolo11.py` — exports `.pt` to ONNX with six outputs
  (`box8/16/32`, `cls8/16/32`) — the format that `yolo8.c` decodes on the
  board;
- `tools/build_cvimodel.sh` — `model_transform.py` → `run_calibration.py`
  (INT8, KL) → `model_deploy.py --chip cv181x`;
- default name: `yolov11n_det_<class>_<imgsz>_<imgsz>_INT8_cv181x.cvimodel`.

The `imgsz` used for export and conversion must match (the `imgsz` field on
the Model page). Calibration uses frames from `calib/` (up to 200).

### Board

- upload: SFTP to `/root/models/`;
- the model name is appended to `/root/models/models.order`;
  `start_stream.sh` prints such models at the end of the menu (so the new one
  is last);
- start: `/root/yolo_stream/start_stream.sh <name>` (the script stops the
  previous stream and starts a new one; the legacy
  `/root/start_stream.sh` path is also supported);
- "Restore auto stream" — `/root/yolo_stream/start_stream.sh auto` (the
  default model, usually coco80).

## Building standalone executables

PyInstaller cannot cross-compile: build the Linux executable on Linux and the
Windows `.exe` on Windows. Install the runtime dependencies and PyInstaller,
then choose what to build:

```sh
cd yolo_studio
python -m pip install -r requirements.txt pyinstaller

python tools/build_desktop.py --mode onefile          # dist/LicheeYOLOStudio[.exe]
python tools/build_desktop.py --mode onedir           # dist/LicheeYOLOStudio/
python tools/build_desktop.py --mode both             # both slim variants
python tools/build_desktop.py --mode onedir --with-ml # dist/LicheeYOLOStudio-full/
```

`--with-ml` embeds `ultralytics` and `torch`, which must be importable in the
build environment:

```sh
# Linux: CPU-only torch, otherwise pip pulls several gigabytes of CUDA
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install ultralytics onnx onnxruntime onnxslim
# Windows
python -m pip install ultralytics onnx onnxruntime onnxslim
```

The full bundle is a folder because a 1 GB one-file executable would unpack
itself into a temporary directory on every start. The frozen application runs
training and the ONNX export through its own hidden modes
(`--yolo-cli`, `--export-onnx`), so no external Python is involved.
Conversion to cvimodel still needs Linux/WSL: `tpu-mlir` does not exist for
Windows.

On Linux, install the xcb development/runtime packages listed above first; the
script copies those libraries into the bundle. The build script also adds
`yaml` and `paramiko` as hidden imports, excludes unused Qt bindings and puts
the LGPL notice next to the binaries.

GitHub Actions builds both variants on every push and attaches them to a
release for `v*` tags. The workflow runs an offscreen `--smoke-test` on both
platforms, and for the full build also `--have-ml`, which fails unless torch
and ultralytics can really be imported from the frozen bundle.

## Troubleshooting

**“Python not found … Set the path in Settings”** — the slim build runs
training and ONNX export in a separate environment with `ultralytics`; PyTorch
is not bundled there (use the `-full` variant if you do not want to install
anything). Clear
**Settings → Python Ultralytics** and press **Detect**, or create the
environment as shown in *Running from source* and point the field at
`…/bin/python` (`…\Scripts\python.exe` on Windows). The error dialog also
offers **Locate interpreter…** and **Search again**.

**“cvimodel conversion needs a POSIX shell”** — `tpu-mlir` only runs on Linux.
Convert on a Linux host or in WSL; the exported ONNX is portable.

**The window does not open on Linux** — the Qt/X11 libraries listed in
*Ready-to-run builds* are missing; the launcher prints the exact package names.

## Limitations

- Labeling is semi-automatic: tracking works well with smooth motion and
  rotation; when the viewpoint changes abruptly, the box has to be fixed
  manually.
- Local CPU training is slow — for real datasets, train on a GPU (Colab) and
  import `best.pt`.
- The base `yolo11n.pt` model is downloaded from GitHub; if there is no
  network, specify a local `.pt` or `yolo11n.yaml` (training from scratch
  without downloading).
- Do not rename classes after training without retraining the model; when
  shortening the list, check that the labels contain no old ids.
- Capture mode requires `stream_yolo` with `SY_HTTP_PORT`; on old firmware the
  stream stays on 7777 (the tool will report that the port does not respond).
- During capture the camera is busy with the board stream: YOLO is stopped for
  that time and comes back with the "Restore YOLO" button (or a reboot).

## Structure

```
app.py                  entry point, Linux Qt preflight, --smoke-test
run.sh                  Linux launcher with an xcb dependency check
studio/
  config.py             settings (~/.config/yolo-studio/config.json)
  board.py              SSH/SFTP, stream modes, models.order
  camera.py             MJPEG/USB frame source (QThread)
  dataset.py            frames, import, YOLO labels, split, calibration
  tracking.py           optical flow
  pyenv.py              finds the ultralytics/tpu-mlir interpreters
  pipeline.py           training, ONNX, tpu-mlir
  ui/                   window and pages (capture/label/model/board)
tools/
  export_yolo11.py      .pt → 6-output ONNX
  build_cvimodel.sh     ONNX → cv181x cvimodel
  build_desktop.py      PyInstaller Linux/Windows one-file build
```
