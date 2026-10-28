cmake_minimum_required(VERSION 3.15)

project(face_stream C CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# No build-machine paths in the binary: runtime libraries are found through
# LD_LIBRARY_PATH, which run_face.sh sets explicitly.
set(CMAKE_SKIP_RPATH TRUE)

# ---------------------------------------------------------------------------
# Dependencies
#
# CVISDK_PATH      cvitek-tdl-sdk-sg200x checkout (middleware + cvi_tdl headers)
# CVIRUNTIME_PATH  cviruntime headers/libs from this repository
# OpenCV_DIR       directory containing OpenCVConfig.cmake
#
# Environment variables with the same names are honoured, so the build works
# both in CI (scripts/fetch-deps.sh) and with a local SDK/OpenCV install.
# ---------------------------------------------------------------------------

if(NOT CVISDK_PATH AND DEFINED ENV{CVISDK_PATH})
    set(CVISDK_PATH "$ENV{CVISDK_PATH}")
endif()
if(NOT CVISDK_PATH)
    set(CVISDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cvisdk")
endif()
set(CVISDK_PATH "${CVISDK_PATH}" CACHE PATH "cvitek-tdl-sdk-sg200x root")

if(NOT CVIRUNTIME_PATH AND DEFINED ENV{CVIRUNTIME_PATH})
    set(CVIRUNTIME_PATH "$ENV{CVIRUNTIME_PATH}")
endif()
if(NOT CVIRUNTIME_PATH)
    set(CVIRUNTIME_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cviruntime")
endif()
set(CVIRUNTIME_PATH "${CVIRUNTIME_PATH}" CACHE PATH "cviruntime root")

if(NOT OpenCV_DIR AND DEFINED ENV{OpenCV_DIR})
    set(OpenCV_DIR "$ENV{OpenCV_DIR}")
endif()
if(NOT OpenCV_DIR AND EXISTS
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv/lib/cmake/opencv4/OpenCVConfig.cmake")
    set(OpenCV_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv/lib/cmake/opencv4")
endif()
if(OpenCV_DIR)
    set(OpenCV_DIR "${OpenCV_DIR}" CACHE PATH "OpenCV install (OpenCVConfig.cmake)")
endif()

set(CVI_SDK_INCLUDE "${CVISDK_PATH}/sample/3rd/middleware/v2/include")
set(CVI_SDK_LIB     "${CVISDK_PATH}/sample/3rd/middleware/v2/lib")

if(NOT EXISTS "${CVI_SDK_INCLUDE}/cvi_sys.h")
    message(FATAL_ERROR
        "CVISDK_PATH='${CVISDK_PATH}' does not look like cvitek-tdl-sdk-sg200x.\n"
        "Expected ${CVI_SDK_INCLUDE}/cvi_sys.h.\n"
        "Run scripts/fetch-deps.sh or pass -DCVISDK_PATH=/path/to/sdk.")
endif()
if(NOT EXISTS "${CVISDK_PATH}/include/cvi_tdl/cvi_tdl.h")
    message(FATAL_ERROR
        "CVISDK_PATH='${CVISDK_PATH}' has no include/cvi_tdl/cvi_tdl.h.\n"
        "Run scripts/fetch-deps.sh or pass -DCVISDK_PATH=/path/to/sdk.")
endif()
if(NOT EXISTS "${CVI_SDK_LIB}/libvenc.so" OR NOT EXISTS "${CVI_SDK_LIB}/libsys.so")
    message(FATAL_ERROR
        "CVISDK_PATH='${CVISDK_PATH}' has no libvenc.so/libsys.so in\n"
        "${CVI_SDK_LIB}.")
endif()
if(NOT EXISTS "${CVIRUNTIME_PATH}/include/cviruntime.h")
    message(FATAL_ERROR
        "CVIRUNTIME_PATH='${CVIRUNTIME_PATH}' has no include/cviruntime.h.")
endif()

find_package(OpenCV REQUIRED)

message(STATUS "CVISDK_PATH      = ${CVISDK_PATH}")
message(STATUS "CVIRUNTIME_PATH  = ${CVIRUNTIME_PATH}")
message(STATUS "OpenCV_DIR       = ${OpenCV_DIR}")
message(STATUS "OpenCV_VERSION   = ${OpenCV_VERSION}")

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CVIRUNTIME_PATH}/include"
    "${CVI_SDK_INCLUDE}"
    "${CVI_SDK_INCLUDE}/linux"
    "${CVISDK_PATH}/include"
    "${CVISDK_PATH}/include/cvi_tdl"
)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
file(MAKE_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")

add_executable(face_stream
    src/main_face.cpp
    src/MJPEGWriter.cpp
    src/webui.cpp
    src/face_pipeline.c
    src/vpss_face.c
    src/venc_jpeg.c
    src/draw_nv21.c
)

target_compile_definitions(face_stream PRIVATE
    SENSOR_GCORE_GC4653
    _MIDDLEWARE_V2_
    C906
    USE_TPU_IVE
)

target_compile_options(face_stream PRIVATE
    -O3
    -fsigned-char
    -fno-strict-aliasing
    -Wno-format-truncation
)

# libcvi_tdl.so is loaded with dlopen (it is built against shared OpenCV 3.2),
# so only the directly called middleware and cviruntime libraries are linked.
target_link_directories(face_stream PRIVATE
    "${CVIRUNTIME_PATH}/lib"
    "${CVI_SDK_LIB}"
)

target_link_options(face_stream PRIVATE
    -static-libstdc++
    -static-libgcc
    -s
)

target_link_libraries(face_stream PRIVATE
    cviruntime
    cvikernel
    venc
    sys
    pthread
    atomic
    m
    ${CMAKE_DL_LIBS}
    rt
    ${OpenCV_LIBS}
)

install(TARGETS face_stream RUNTIME DESTINATION bin)
