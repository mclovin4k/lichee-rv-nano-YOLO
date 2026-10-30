# Board files: face_stream

Contents of the release package: `face_stream`, `run_face.sh`,
`start_face.sh`, `runtime/`, `models/face/`. The binary and the matching
`libc`/loader are produced by the GitHub workflow and are not stored in git.

## Installation

The easiest way is the release installer from the repository root:

```sh
cd yolo_face
./install.sh <IP>
```

It downloads `yolo_face-board.tar.gz` from the latest GitHub Release, verifies
its checksum and copies everything to `/root/yolo_face`. You can also unpack
the release tarball manually and copy its contents there. Then:

```sh
/root/yolo_face/start_face.sh start
```

The camera is configured by the stock image through `/mnt/data/sensor_cfg.ini`.
No autostart is installed.

## Control

```sh
/root/yolo_face/start_face.sh start|stop|restart|status|log
```

- UI: `http://<IP>:7777/`
- MJPEG: `http://<IP>:7777/stream`
- Log: `/root/yolo_face/log/face.log`
- Gallery: `/root/faces/<name>/`
- PID: `/tmp/yolo_face.pid` (supervisor), `/tmp/yolo_face.app.pid`

`start_face.sh start` stops `stream_yolo` (shared camera and port). The
reverse is also true: `yolo_stream/start_stream.sh` stops `face_stream`.

## Runtime

`runtime/` contains `libcvi_tdl.so`, its dependencies (`libcnpy`,
`libcvi_ive`, shared OpenCV 3.2, `libz`), the camera/ISP/VENC middleware
libraries, `libvpu_tdl.so` for TDL, `libstdc++`/`libgcc`/`libatomic` and the
musl loader. The system `libvpu.so` is taken from the board
(`/mnt/system/usr/lib`).

`run_face.sh` sets `LD_LIBRARY_PATH` and uses the bundled loader
(`runtime/ld-musl-riscv64.so.1`) by default, so the package works on images
with a different or missing system musl loader. To use the system loader:

```sh
SY_BUNDLED_LOADER=0 /root/yolo_face/start_face.sh restart
```
