#!/bin/sh
# Build a cv181x (INT8) cvimodel from a 6-output ONNX model.
# usage: build_cvimodel.sh <model.onnx> <calib_dir> [<test.jpg>]
# env:   TPU_PY=<python from the tpu-mlir venv> (default: ~/.venvs/tpu310/bin/python)
#        NAME=<base output name>          (default: model)
#        OUT=<.cvimodel name>             (default: model.cvimodel)
#        IMGSZ=<input side>               (default: 640)
#        CALIB_NUM=<calibration frames>   (default: 200)
set -e

ONNX=${1:?usage: build_cvimodel.sh model.onnx calib_dir [test.jpg]}
CALIB=${2:?usage: build_cvimodel.sh model.onnx calib_dir [test.jpg]}
TEST=${3:-}
V=${TPU_PY:-$HOME/.venvs/tpu310/bin/python}
NAME=${NAME:-model}
OUT=${OUT:-model.cvimodel}
IMGSZ=${IMGSZ:-640}
CALIB_NUM=${CALIB_NUM:-200}

if [ -z "$TEST" ]; then
    TEST=$(find "$CALIB" -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) | head -1)
fi
[ -n "$TEST" ] || { echo "no test frame in $CALIB"; exit 1; }

# tpu-mlir invokes its tools through PATH — put the venv first
PATH=$(dirname "$V"):$PATH
export PATH

model_transform.py --model_name "$NAME" --model_def "$ONNX" \
    --input_shapes "[[1,3,${IMGSZ},${IMGSZ}]]" \
    --mean 0,0,0 --scale 0.0039216,0.0039216,0.0039216 \
    --pixel_format rgb --keep_aspect_ratio \
    --test_input "$TEST" \
    --test_result "${NAME}_top_outputs.npz" \
    --mlir "$NAME.mlir"

run_calibration.py "$NAME.mlir" --dataset "$CALIB" --input_num "$CALIB_NUM" \
    -o "${NAME}_cali_table"

model_deploy.py --mlir "$NAME.mlir" --chip cv181x --quantize INT8 \
    --calibration_table "${NAME}_cali_table" \
    --quant_input --quant_output \
    --test_input "${NAME}_in_f32.npz" --test_reference "${NAME}_top_outputs.npz" \
    --tolerance 0.85,0.45 \
    --model "$OUT"

echo "done: $OUT"
