#!/usr/bin/env python3
"""Build standalone Lichee YOLO Studio executables with PyInstaller.

The build must run on the target operating system (PyInstaller cannot cross
compile): Linux on Linux, Windows on Windows. Output goes to ``dist/``:

    --mode onefile           dist/LicheeYOLOStudio[.exe]        (slim)
    --mode onedir            dist/LicheeYOLOStudio/             (slim)
    --mode both              both of the above
    --mode onedir --with-ml  dist/LicheeYOLOStudio-full/        (torch inside)

The slim build needs a separate Python environment for training and ONNX
export (see README). The ``--with-ml`` build bundles ultralytics + torch, so
training works with no Python installed at all; the archive is large
(~1 GB installed) but starts instantly because it is a directory.

Linux: the script bundles the xcb libraries Qt needs (including
libxcb-cursor.so.0). Install the matching development/runtime packages before
building; on Debian/Ubuntu:

    sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 \
        libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
        libxcb-shape0 libxcb-xinerama0 libxcb-xkb1
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
NAME = "LicheeYOLOStudio"

# Libraries dlopen()ed by the Qt xcb platform plugin. They are not always
# installed on minimal desktops, so put them inside the bundle.
LINUX_XCB_LIBS = (
    "libxcb-cursor.so.0",
    "libxcb-icccm.so.4",
    "libxcb-image.so.0",
    "libxcb-keysyms.so.1",
    "libxcb-randr.so.0",
    "libxcb-render-util.so.0",
    "libxcb-shape.so.0",
    "libxcb-xinerama.so.0",
    "libxcb-xkb.so.1",
    "libxkbcommon-x11.so.0",
    "libxkbcommon.so.0",
    "libxcb-util.so.1",
)

_LIB_DIRS = (
    Path("/usr/lib/x86_64-linux-gnu"),
    Path("/lib/x86_64-linux-gnu"),
    Path("/usr/lib/aarch64-linux-gnu"),
    Path("/lib/aarch64-linux-gnu"),
    Path("/usr/lib64"),
    Path("/usr/lib"),
    Path("/lib"),
)

NOTICES = ("LGPL-3.0.txt",)


def find_library(soname: str) -> Path | None:
    for directory in _LIB_DIRS:
        candidate = directory / soname
        if candidate.exists():
            return candidate.resolve()
    try:
        result = subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True, check=False)
    except OSError:
        return None
    for line in result.stdout.splitlines():
        if soname not in line:
            continue
        candidate = line.rsplit("=>", 1)[-1].strip()
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def pyinstaller_cmd(name: str, mode: str, work: Path) -> list[str]:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--windowed",
        "--name", name,
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(work),
        "--specpath", str(ROOT / "build"),
        "--paths", str(ROOT),
        "--hidden-import", "yaml",
        "--hidden-import", "paramiko",
        "--exclude-module", "tkinter",
        "--exclude-module", "PyQt5",
        "--exclude-module", "PyQt6",
    ]
    cmd.append("--onefile" if mode == "onefile" else "--onedir")
    return cmd


def ml_args() -> list[str]:
    """PyInstaller flags for a build that embeds ultralytics and torch."""
    return [
        "--collect-all", "ultralytics",
        "--collect-submodules", "torch",
        "--collect-submodules", "torchvision",
        "--hidden-import", "torch",
        "--hidden-import", "torchvision",
        "--hidden-import", "onnx",
        "--hidden-import", "onnxruntime",
    ]


def copy_notices(target: Path) -> None:
    """LGPL/Qt notices next to the binary (PySide6 is LGPL-3.0)."""
    target.mkdir(parents=True, exist_ok=True)
    for name in NOTICES:
        src = REPO / "LICENSES" / name
        if src.is_file():
            shutil.copy2(src, target / name)
    notices = REPO / "THIRD_PARTY_NOTICES.md"
    if notices.is_file():
        shutil.copy2(notices, target / notices.name)


def build(mode: str, with_ml: bool) -> None:
    name = NAME + ("-full" if with_ml else "")
    tag = "full" if with_ml else "slim"
    print(f"\n=== {name} ({mode}, {tag}) ===")

    cmd = pyinstaller_cmd(name, mode, ROOT / "build" / f"pyinstaller-{name}-{mode}")
    if with_ml:
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            raise SystemExit(
                "--with-ml needs ultralytics in the build environment:\n"
                "  python -m pip install ultralytics")
        cmd += ml_args()
    if sys.platform.startswith("linux"):
        missing = []
        for soname in LINUX_XCB_LIBS:
            path = find_library(soname)
            if path is None:
                missing.append(soname)
                continue
            cmd.extend(["--add-binary", f"{path}:."])
        if missing:
            print("warning: these xcb libraries were not found and will not be "
                  "bundled:\n  " + "\n  ".join(missing), file=sys.stderr)
            print("Install them before building the release (see the module "
                  "docstring for the Debian/Ubuntu package list).", file=sys.stderr)
    cmd.append(str(ROOT / "app.py"))

    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)

    dist = ROOT / "dist"
    if mode == "onefile":
        exe = dist / (name + (".exe" if sys.platform == "win32" else ""))
        if not exe.exists():
            raise SystemExit(f"error: {exe} was not produced")
        print(f"Done: {exe} ({exe.stat().st_size / (1024 * 1024):.1f} MiB)")
    else:
        folder = dist / name
        if not folder.is_dir():
            raise SystemExit(f"error: {folder} was not produced")
        copy_notices(folder)
        size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        print(f"Done: {folder}/ ({size / (1024 * 1024):.1f} MiB)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("onefile", "onedir", "both"),
                        default="onefile",
                        help="onefile: a single executable; onedir: a folder "
                             "that starts faster (default: onefile)")
    parser.add_argument("--with-ml", action="store_true",
                        help="bundle ultralytics and torch (large, training "
                             "works without a separate Python)")
    args = parser.parse_args(argv)

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is not installed. Run:\n"
              "  python -m pip install -r requirements.txt pyinstaller",
              file=sys.stderr)
        return 1

    modes = ("onefile", "onedir") if args.mode == "both" else (args.mode,)
    for mode in modes:
        build(mode, args.with_ml)
    return 0


if __name__ == "__main__":
    sys.exit(main())
