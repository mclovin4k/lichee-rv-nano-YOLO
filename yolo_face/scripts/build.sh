#!/bin/sh
# Build face_stream for LicheeRV Nano (SG2002/CV181x, riscv64 musl).
#
#   ./scripts/fetch-deps.sh
#   ./scripts/build-opencv.sh
#   ./scripts/build.sh
#
# Or point the script at an existing toolchain/SDK/OpenCV:
#   TOOLCHAIN_PATH=/path/to/riscv-toolchain \
#   CVISDK_PATH=/path/to/cvitek-tdl-sdk-sg200x \
#   OpenCV_DIR=/path/to/opencv/lib/cmake/opencv4 \
#   ./scripts/build.sh
#
# The resulting binary is yolo_face/build/bin/face_stream.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DEPS=${DEPS_DIR:-"$ROOT/.deps"}

TOOLCHAIN_PATH=${TOOLCHAIN_PATH:-"$DEPS/riscv64-linux-musl-cross"}
CVISDK_PATH=${CVISDK_PATH:-"$DEPS/cvitek-tdl-sdk-sg200x"}
# cviruntime headers and libraries: the downloaded deps, the shared vendor/
# directory filled by scripts/fetch-vendor.sh, or a local checkout.
if [ -z "${CVIRUNTIME_PATH:-}" ]; then
    for d in "$DEPS/cviruntime" "$ROOT/../vendor/cviruntime" \
             "$ROOT/third_party/cviruntime"; do
        [ -f "$d/include/cviruntime.h" ] && { CVIRUNTIME_PATH=$d; break; }
    done
fi
CVIRUNTIME_PATH=${CVIRUNTIME_PATH:-"$DEPS/cviruntime"}
OpenCV_DIR=${OpenCV_DIR:-"$DEPS/opencv-install/lib/cmake/opencv4"}
BUILD_DIR=${BUILD_DIR:-"$ROOT/build"}

say() { printf '\n>>> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[ -f "$CVISDK_PATH/sample/3rd/middleware/v2/include/cvi_sys.h" ] || \
    die "SDK not found: $CVISDK_PATH (run scripts/fetch-deps.sh or set CVISDK_PATH)"
[ -f "$OpenCV_DIR/OpenCVConfig.cmake" ] || \
    die "OpenCVConfig.cmake not found in $OpenCV_DIR (run scripts/build-opencv.sh or set OpenCV_DIR)"

say "configuring face_stream"
# TOOLCHAIN_PATH is exported for CMake's try_compile step: -D variables are
# not visible in that nested configure.
TOOLCHAIN_PATH="$TOOLCHAIN_PATH" cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$ROOT/cmake/riscv64-musl.cmake" \
    -DCVISDK_PATH="$CVISDK_PATH" \
    -DCVIRUNTIME_PATH="$CVIRUNTIME_PATH" \
    -DOpenCV_DIR="$OpenCV_DIR"

say "building"
cmake --build "$BUILD_DIR" --parallel "$(nproc 2>/dev/null || echo 4)"

BIN="$BUILD_DIR/bin/face_stream"
[ -f "$BIN" ] || die "binary not produced: $BIN"

say "result"
ls -l "$BIN"
file "$BIN"

cat <<EOF

Done:
  $BIN

Install on the board (asks for its IP):
  $ROOT/install.sh
EOF
