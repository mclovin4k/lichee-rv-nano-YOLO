#!/bin/sh
# Download pinned build dependencies into yolo_face/.deps/ (ignored by git).
#
#   ./scripts/fetch-deps.sh
#
# Sources:
#   - RISC-V musl toolchain: https://github.com/musl-cross/musl-cross/releases
#   - cvitek-tdl-sdk-sg200x: https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x
#   - opencv-mobile 4.13.0:  https://github.com/nihui/opencv-mobile (Apache-2.0)
#
# The SDK checkout is sparse: the middleware headers, cvi_tdl headers, libvenc.so and
# libsys.so are downloaded.  The OpenCV source is the official 4.13.0 release
# archive, patched with yolo_face/patches/opencv-mobile-4.13.0-cvi-capture.patch.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DEPS=${DEPS_DIR:-"$ROOT/.deps"}
mkdir -p "$DEPS"

TOOLCHAIN_URL=${TOOLCHAIN_URL:-https://github.com/musl-cross/musl-cross/releases/download/20261001/riscv64-unknown-linux-musl.tar.xz}
TOOLCHAIN_SHA256=${TOOLCHAIN_SHA256:-054acc4b5baa886363ce5acaa6a015fd90d2f715b2e4324813fd4d49d717362f}
TC_DIR="$DEPS/riscv64-linux-musl-cross"

SDK_REPO=${SDK_REPO:-https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x.git}
SDK_COMMIT=${SDK_COMMIT:-02dfb9091d390a7b0d1177fe7b651db1b99854bc}
SDK_DIR="$DEPS/cvitek-tdl-sdk-sg200x"

OCV_VERSION=4.13.0
OCV_URL=${OCV_URL:-https://github.com/nihui/opencv-mobile/releases/download/v36/opencv-mobile-4.13.0.zip}
OCV_SHA256=${OCV_SHA256:-0aeaea4837d19894765aae23cb82fc4b87cec07449bee4b4404a63bdda15d6ee}
OCV_ZIP="$DEPS/opencv-mobile-$OCV_VERSION.zip"
OCV_SRC_ROOT="$DEPS/opencv-src"
OCV_DIR="$OCV_SRC_ROOT/opencv-mobile-$OCV_VERSION"
OCV_PATCH="$ROOT/patches/opencv-mobile-4.13.0-cvi-capture.patch"

say() { printf '\n>>> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

need() {
    command -v "$1" >/dev/null 2>&1 || die "missing '$1'"
}
need git
need patch
need tar

download() { # download <url> <file>
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --connect-timeout 20 "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$2" "$1"
    else
        die "need curl or wget"
    fi
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        die "need sha256sum or shasum"
    fi
}

fetch_commit() { # fetch_commit <repo> <commit> <dir>
    rm -rf "$3"
    mkdir -p "$3"
    git -C "$3" init -q
    git -C "$3" remote add origin "$1"
    git -C "$3" fetch -q --depth 1 origin "$2"
    git -C "$3" checkout -q FETCH_HEAD
}

toolchain_prefix() { # print the compiler prefix if the toolchain is complete
    if [ -x "$TC_DIR/bin/riscv64-linux-musl-gcc" ]; then
        echo riscv64-linux-musl
    elif [ -x "$TC_DIR/bin/riscv64-unknown-linux-musl-gcc" ]; then
        echo riscv64-unknown-linux-musl
    else
        return 1
    fi
}

if ! toolchain_prefix >/dev/null; then
    say "toolchain: $TOOLCHAIN_URL"
    download "$TOOLCHAIN_URL" "$DEPS/toolchain.download"
    if [ -n "$TOOLCHAIN_SHA256" ]; then
        got=$(sha256_of "$DEPS/toolchain.download")
        [ "$got" = "$TOOLCHAIN_SHA256" ] || \
            die "toolchain sha256 mismatch: got $got, expected $TOOLCHAIN_SHA256"
    fi
    tar -xf "$DEPS/toolchain.download" -C "$DEPS"
    rm -f "$DEPS/toolchain.download"

    # musl.cc unpacks as riscv64-linux-musl-cross, while the
    # musl-cross GitHub release unpacks as riscv64-unknown-linux-musl.
    # Keep one stable directory name for build.sh and the CI cache.
    for d in "$DEPS/riscv64-linux-musl-cross" "$DEPS/riscv64-unknown-linux-musl"; do
        [ -d "$d" ] || continue
        if [ "$d" != "$TC_DIR" ]; then
            rm -rf "$TC_DIR"
            mv "$d" "$TC_DIR"
        fi
        break
    done
fi
toolchain_prefix >/dev/null || \
    die "toolchain not unpacked as expected: $TC_DIR"

if [ ! -f "$SDK_DIR/.ready" ]; then
    say "cvitek SDK: $SDK_REPO @ $SDK_COMMIT (headers + libvenc/libsys)"
    rm -rf "$SDK_DIR"
    mkdir -p "$SDK_DIR"
    git -C "$SDK_DIR" init -q
    git -C "$SDK_DIR" remote add origin "$SDK_REPO"
    git -C "$SDK_DIR" config core.sparseCheckout true
    mkdir -p "$SDK_DIR/.git/info"
    cat > "$SDK_DIR/.git/info/sparse-checkout" <<'EOF'
sample/3rd/middleware/v2/include/
include/
sample/3rd/middleware/v2/lib/libvenc.so
sample/3rd/middleware/v2/lib/libsys.so
EOF
    git -C "$SDK_DIR" fetch -q --depth 1 origin "$SDK_COMMIT"
    git -C "$SDK_DIR" checkout -q FETCH_HEAD
    touch "$SDK_DIR/.ready"
fi
[ -f "$SDK_DIR/sample/3rd/middleware/v2/include/cvi_sys.h" ] || \
    die "SDK headers missing (sparse checkout failed)"
[ -f "$SDK_DIR/sample/3rd/middleware/v2/lib/libvenc.so" ] || \
    die "SDK libvenc.so missing (sparse checkout failed)"

if [ ! -f "$OCV_DIR/.ready" ]; then
    say "OpenCV source: $OCV_URL"
    [ -f "$OCV_ZIP" ] || download "$OCV_URL" "$OCV_ZIP"
    got=$(sha256_of "$OCV_ZIP")
    [ "$got" = "$OCV_SHA256" ] || \
        die "OpenCV archive sha256 mismatch: got $got, expected $OCV_SHA256"
    rm -rf "$OCV_SRC_ROOT"
    mkdir -p "$OCV_SRC_ROOT"
    need unzip
    unzip -q "$OCV_ZIP" -d "$OCV_SRC_ROOT"
    [ -f "$OCV_DIR/CMakeLists.txt" ] || \
        die "OpenCV source not found under $OCV_DIR"
    [ -f "$OCV_PATCH" ] || die "patch not found: $OCV_PATCH"
    say "applying $OCV_PATCH"
    patch -d "$OCV_SRC_ROOT" -p1 < "$OCV_PATCH"
    touch "$OCV_DIR/.ready"
fi

cat <<EOF

Dependencies are ready:
  TOOLCHAIN_PATH=$TC_DIR
  CVISDK_PATH=$SDK_DIR
  OPENCV_SRC=$OCV_DIR
  OPENCV_INSTALL=$DEPS/opencv-install

Next:
  ./scripts/build-opencv.sh
  ./scripts/build.sh
EOF
