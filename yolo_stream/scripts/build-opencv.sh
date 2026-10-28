#!/bin/sh
# Build opencv-mobile 4.13.0 with the CVI capture backend for LicheeRV Nano.
#
#   ./scripts/fetch-deps.sh
#   ./scripts/build-opencv.sh
#
# Override paths with environment variables:
#   TOOLCHAIN_PATH  RISC-V musl toolchain root
#   OPENCV_SRC      opencv-mobile-4.13.0 source directory
#   OPENCV_BUILD    build directory
#   OPENCV_INSTALL  install prefix (contains lib/cmake/opencv4)
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DEPS=${DEPS_DIR:-"$ROOT/.deps"}

TOOLCHAIN_PATH=${TOOLCHAIN_PATH:-"$DEPS/riscv64-linux-musl-cross"}
OPENCV_SRC=${OPENCV_SRC:-"$DEPS/opencv-src/opencv-mobile-4.13.0"}
OPENCV_BUILD=${OPENCV_BUILD:-"$DEPS/opencv-build"}
OPENCV_INSTALL=${OPENCV_INSTALL:-"$DEPS/opencv-install"}

say() { printf '\n>>> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[ -f "$OPENCV_SRC/CMakeLists.txt" ] || \
    die "OpenCV source not found: $OPENCV_SRC (run scripts/fetch-deps.sh)"
[ -x "$TOOLCHAIN_PATH/bin/riscv64-linux-musl-gcc" ] || \
    die "toolchain not found: $TOOLCHAIN_PATH (run scripts/fetch-deps.sh)"

say "configuring OpenCV ($OPENCV_SRC)"
set --
while IFS= read -r opt; do
    case "$opt" in
        ''|'#'*) continue ;;
    esac
    set -- "$@" "$opt"
done < "$OPENCV_SRC/options.txt"

TOOLCHAIN_PATH="$TOOLCHAIN_PATH" cmake -S "$OPENCV_SRC" -B "$OPENCV_BUILD" \
    -DCMAKE_TOOLCHAIN_FILE="$ROOT/cmake/riscv64-musl.cmake" \
    "$@" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=rv64imafdc -mabi=lp64d -fno-rtti -fno-exceptions -I$OPENCV_SRC/modules/highgui/src" \
    -DCMAKE_CXX_FLAGS="-march=rv64imafdc -mabi=lp64d -fno-rtti -fno-exceptions -I$OPENCV_SRC/modules/highgui/src" \
    -DCMAKE_INSTALL_PREFIX="$OPENCV_INSTALL" \
    -DBUILD_SHARED_LIBS=OFF \
    -DWITH_OPENMP=OFF \
    -DBUILD_LIST= \
    -DBUILD_opencv_core=ON \
    -DBUILD_opencv_imgproc=ON \
    -DBUILD_opencv_photo=ON \
    -DBUILD_opencv_video=ON \
    -DBUILD_opencv_features2d=ON \
    -DBUILD_opencv_highgui=ON \
    -DBUILD_opencv_imgcodecs=OFF \
    -DBUILD_opencv_videoio=OFF \
    -DBUILD_opencv_calib3d=OFF \
    -DBUILD_opencv_flann=OFF \
    -DBUILD_opencv_ml=OFF \
    -DBUILD_opencv_objdetect=OFF \
    -DBUILD_opencv_stitching=OFF \
    -DBUILD_opencv_dnn=OFF \
    -DBUILD_opencv_gapi=OFF \
    -DBUILD_opencv_world=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_python3=OFF \
    -DBUILD_opencv_java=OFF \
    -DBUILD_opencv_js=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_DOCS=OFF \
    -DWITH_CVI=ON \
    -DCPU_BASELINE= \
    -DCPU_DISPATCH=

say "building OpenCV"
cmake --build "$OPENCV_BUILD" --parallel "$(nproc 2>/dev/null || echo 4)"

say "installing OpenCV to $OPENCV_INSTALL"
cmake --install "$OPENCV_BUILD" >/dev/null

cat <<EOF

OpenCV is ready:
  OpenCV_DIR=$OPENCV_INSTALL/lib/cmake/opencv4
EOF
