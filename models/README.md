# Models

No `.cvimodel` file is stored in this repository or uploaded to its releases.
Two reasons:

- models exported from Ultralytics YOLOv8/YOLO11 are **AGPL-3.0** derivatives,
  and this repository is MIT;
- the models that come with the Cvitek/Sophgo TDL SDK (face detection,
  embeddings, liveness) have **no explicit license**.

The applications read models from the board at run time, so you can use
whatever you are allowed to use locally.

## Where to get models

### 1. Export your own (recommended)

Use Lichee YOLO Studio: capture or import a dataset → label → train → export
ONNX → build `.cvimodel` with tpu-mlir. The Studio writes
`<name>.cvimodel` and `<name>.names`; upload both:

```sh
cd yolo_stream
MODELS_DIR=~/models/my-detector ./install.sh <IP>
```

The `.names` file has one class name per line and is what makes the boxes show
names instead of `class:N`. The model name itself encodes the input size and
class count: `yolov11n_det_<name>_640_640_INT8_cv181x.cvimodel` is a 640×640
INT8 model for cv181x. The board binary also reads the input size and the
number of classes from the tensors, so a differently named file still works.

If you publish a model derived from Ultralytics weights, the AGPL applies:
publish the training data, configuration, `.pt`/ONNX and conversion scripts
together with the `.cvimodel`.

### 2. Models already on the board

Stock images ship detection and face models under
`/mnt/system/usr/bin/ai/models`. `start_stream.sh` searches
`/mnt/system/usr/bin/ai/models` automatically, and `install.sh` copies the
face models from there when you point `MODELS_DIR` at it:

```sh
MODELS_DIR=/mnt/system/usr/bin/ai/models ./install.sh <IP>   # on the board
```

### 3. SDK examples

`scripts/fetch-vendor.sh --download` fetches the models published in
[milkv-duo/cvitek-tdl-sdk-sg200x](https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x):

- `yolov5_cv181x_int8_sym.cvimodel` / `..._asym.cvimodel` — YOLOv5, not
  compatible with `stream_yolo` (different head), useful as a reference;
- `scrfd_768_432_int8_1x.cvimodel` — SCRFD face detector from the SDK.

The SDK has no license file; treat these files as "all rights reserved" unless
Sophgo says otherwise.

## Face models

`face_stream` expects these three names in `models/face/`:

```
scrfd_det_face_432_768_INT8_cv181x.cvimodel
feature_cviface_112_112_INT8_cv181x.cvimodel
cls_rgbliveness_256_256_INT8_cv181x.cvimodel
```

They come from the Cvitek TDL SDK examples. The filenames are the contract:
the application resolves them by name and can be pointed elsewhere with
`SY_FD_MODEL`, `SY_FR_MODEL`, `SY_LIVE_MODEL`.

## Compatibility

`stream_yolo` accepts YOLOv8/YOLO11 detection exports with the six-output raw
head produced by `yolo_studio/tools/export_yolo11.py`
(`[box8, box16, box32, cls8, cls16, cls32]`, no DFL/NMS). Anything else is
rejected with a log message. int8 and fp32 both work.

## Legal summary

| Model source | License |
|---|---|
| Your export from Ultralytics YOLOv8/YOLO11 | AGPL-3.0 |
| Your export trained from scratch (`yolo11n.yaml`, your data) | your choice |
| SDK example models | no license file |
| Face models (SCRFD weights, cviface, liveness) | no license file |

Full details: [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
