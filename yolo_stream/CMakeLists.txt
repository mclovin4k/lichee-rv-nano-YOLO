cmake_minimum_required(VERSION 3.15)

project(stream_yolo)
set(CMAKE_CXX_STANDARD 11)

SET(CMAKE_C_COMPILER "$ENV{COMPILER}/riscv64-unknown-linux-musl-gcc")
SET(CMAKE_CXX_COMPILER "$ENV{COMPILER}/riscv64-unknown-linux-musl-g++")
SET(CMAKE_C_LINK_EXECUTABLE "$ENV{COMPILER}/riscv64-unknown-linux-musl-ld")
set(CMAKE_CXX_FLAGS "-march=rv64imafd -O3 -DSENSOR_GCORE_GC4653 -D_MIDDLEWARE_V2_ -DC906 -DUSE_TPU_IVE -fsigned-char -Wno-format-truncation -fdiagnostics-color=always -s")
set(CMAKE_C_FLAGS "-march=rv64imafdc -mabi=lp64d -O3 -fsigned-char -fno-strict-aliasing")
#-DNDEBUG

include_directories(
    third_party/cviruntime/include
    $ENV{SDK_PATH}/sample/3rd/middleware/v2/include
)

set(OpenCV_DIR "/opt/opencv-mobile/opencv-mobile-4.13.0/build-rv/install/lib/cmake/opencv4")

find_package(OpenCV REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})

add_executable(stream_yolo main.cpp MJPEGWriter.cpp yolo.cpp yolo8.c venc_jpeg.c draw_nv21.c vpss_det.c)

# Capture (OpenCV highgui -> CVI backend) reaches most of the middleware via
# dlopen at runtime; we link only what we call directly:
#   - libcviruntime      NPU inference (board copy, matches kernel TPU driver)
#   - libvenc/libsys     hardware JPEG (VENC) + ION buffer for its input
target_link_libraries(stream_yolo
    -march=rv64imafdc -mabi=lp64d -static-libstdc++ -static-libgcc
    -L${CMAKE_CURRENT_SOURCE_DIR}/third_party/cviruntime/lib
    -lcviruntime -lcvikernel
    -L$ENV{SDK_PATH}/sample/3rd/middleware/v2/lib
    -lvenc -lsys
    -lpthread -latomic -lm
    ${OpenCV_LIBS}
)
