# Release checklist

What a release contains, how it is produced and how to verify it.

## Assets

| Asset | Size | Contents |
|---|---|---|
| `yolo_stream-board.tar.gz` | ~1 MB | `stream_yolo`, `start_stream.sh`, `run_stream.sh`, `README.md`, empty `models/`, `md5sums.txt` |
| `yolo_face-board.tar.gz` | ~1 MB | `face_stream`, `start_face.sh`, `run_face.sh`, `README.md`, empty `models/face/`, `md5sums.txt` |
| `yolo_studio-linux-x86_64.tar.gz` | ~100 MB | one-file Studio + `LGPL-3.0.txt` + `THIRD_PARTY_NOTICES.md` |
| `yolo_studio-linux-x86_64-full.tar.gz` | ~700 MB | one-folder Studio with ultralytics + torch, notices inside |
| `yolo_studio-windows-x86_64.tar.gz` | ~100 MB | `LicheeYOLOStudio.exe` + notices |
| `yolo_studio-windows-x86_64-full.tar.gz` | ~900 MB | one-folder Studio with ultralytics + torch, notices inside |
| `*.sha256` | — | one per archive |
| `SHA256SUMS.txt` | — | all of the above |

No vendor library and no model is part of any asset. `install.sh` copies the
libraries from the board firmware and uploads models from `MODELS_DIR`; see
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## How the assets are built

`.github/workflows/build.yml`, job by job:

1. `riscv` (matrix: `yolo_stream`, `yolo_face`) — fetches the musl toolchain,
   the cvitek SDK and OpenCV from upstream only, builds the binary and runs
   `scripts/package-board.sh` (slim package, no vendor files).
2. `studio-linux` — builds the slim one-file executable and the full
   one-folder bundle (`--mode onedir --with-ml`) on Ubuntu 22.04. Torch comes
   from `download.pytorch.org/whl/cpu`, otherwise pip would pull gigabytes of
   CUDA. Smoke tests: `--smoke-test` for both, `--have-ml` for the full one.
3. `studio-windows` — the same on `windows-latest`; PyPI torch is already
   CPU-only there. The `.exe` is smoke-tested offscreen and the full bundle is
   checked with `--have-ml`.
4. `release` — only for `v*` tags; downloads every artifact, copies the
   archives and `.sha256` files, writes `SHA256SUMS.txt` and creates the
   release.

## Publishing

```sh
git tag v1.0.1
git push origin v1.0.1
```

Wait for the `build` workflow to finish:

```sh
gh run list --workflow build --limit 1
gh run watch
```

The release appears at
<https://github.com/mclovin4k/lichee-rv-nano-YOLO/releases>. Edit the notes if
needed (the workflow generates them from commits).

## Verifying a release

```sh
gh release download v1.0.1 --pattern '*.sha256' --pattern 'SHA256SUMS.txt'
sha256sum -c SHA256SUMS.txt --ignore-missing
```

Board archive sanity check (no vendor files, no models):

```sh
tar -tzf yolo_stream-board.tar.gz
# stream_yolo, start_stream.sh, run_stream.sh, README.md, md5sums.txt, models/
```

Studio smoke test without a display:

```sh
tar -xzf yolo_studio-linux-x86_64.tar.gz
QT_QPA_PLATFORM=offscreen ./LicheeYOLOStudio --smoke-test       # exit 0
tar -xzf yolo_studio-linux-x86_64-full.tar.gz
QT_QPA_PLATFORM=offscreen LicheeYOLOStudio-full/LicheeYOLOStudio-full --smoke-test
LicheeYOLOStudio-full/LicheeYOLOStudio-full --have-ml           # prints ok
```

## Removing a release

```sh
gh release delete v1.0.0 --yes --cleanup-tag
```

GitHub may keep unreferenced blobs reachable by SHA for a while; that is
normal and cannot be forced through the API.
