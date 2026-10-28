# Cross-compilation for LicheeRV Nano (SG2002/CV181x, musl libc).
#
# The toolchain directory is taken from (in this order):
#   1. -DTOOLCHAIN_PATH=... / -DCMAKE_TOOLCHAIN_FILE=...
#   2. the TOOLCHAIN_PATH environment variable
#   3. <repo>/yolo_stream/third_party/toolchain (optional local copy)
#
# Both common prefixes are accepted, so the musl.cc toolchain
# (riscv64-linux-musl-*) and the Sipeed/Sophgo one
# (riscv64-unknown-linux-musl-*) work as-is.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(DEFINED TOOLCHAIN_PATH AND NOT "${TOOLCHAIN_PATH}" STREQUAL "")
    set(_toolchain "${TOOLCHAIN_PATH}")
elseif(DEFINED ENV{TOOLCHAIN_PATH} AND NOT "$ENV{TOOLCHAIN_PATH}" STREQUAL "")
    set(_toolchain "$ENV{TOOLCHAIN_PATH}")
else()
    set(_toolchain "${CMAKE_CURRENT_LIST_DIR}/../third_party/toolchain")
endif()

if(NOT IS_DIRECTORY "${_toolchain}")
    message(FATAL_ERROR
        "RISC-V musl toolchain not found at '${_toolchain}'. "
        "Set TOOLCHAIN_PATH or pass -DTOOLCHAIN_PATH=/path/to/toolchain.")
endif()

set(_triple "")
foreach(_candidate riscv64-unknown-linux-musl riscv64-linux-musl)
    if(EXISTS "${_toolchain}/bin/${_candidate}-gcc")
        set(_triple "${_candidate}")
        break()
    endif()
endforeach()

if(_triple STREQUAL "")
    message(FATAL_ERROR
        "No riscv64 musl compiler in '${_toolchain}/bin'. Expected "
        "riscv64-unknown-linux-musl-gcc or riscv64-linux-musl-gcc.")
endif()

set(CMAKE_C_COMPILER   "${_toolchain}/bin/${_triple}-gcc")
set(CMAKE_CXX_COMPILER "${_toolchain}/bin/${_triple}-g++")
set(CMAKE_AR           "${_toolchain}/bin/${_triple}-ar"     CACHE FILEPATH "" FORCE)
set(CMAKE_RANLIB       "${_toolchain}/bin/${_triple}-ranlib" CACHE FILEPATH "" FORCE)
set(CMAKE_STRIP        "${_toolchain}/bin/${_triple}-strip"  CACHE FILEPATH "" FORCE)

set(CMAKE_C_FLAGS_INIT   "-march=rv64imafdc -mabi=lp64d")
set(CMAKE_CXX_FLAGS_INIT "-march=rv64imafdc -mabi=lp64d")

set(CMAKE_FIND_ROOT_PATH "${_toolchain}/${_triple}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
