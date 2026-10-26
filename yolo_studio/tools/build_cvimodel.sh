#!/bin/sh
# Сборка cvimodel cv181x (INT8) из 6-выходного ONNX.
# usage: build_cvimodel.sh <model.onnx> <calib_dir> [<test.jpg>]
# env:   TPU_PY=<python venv-а tpu-mlir> (по умолчанию ~/.venvs/tpu310/bin/python)
#        NAME=<базовое имя выходов>        (по умолчанию model)
#        OUT=<имя .cvimodel>               (по умолчанию model.cvimodel)
#        IMGSZ=<сторона входа>             (по умолчанию 640)
#        CALIB_NUM=<кадров калибровки>     (по умолчанию 200)
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
[ -n "$TEST" ] || { echo "нет тестового кадра в $CALIB"; exit 1; }

# tpu-mlir зовёт свои утилиты через PATH — venv впереди
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

echo "готово: $OUT"
