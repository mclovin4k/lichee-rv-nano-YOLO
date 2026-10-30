#!/usr/bin/env python3
"""Build a standalone Lichee YOLO Studio executable with PyInstaller.

The build must run on the target operating system (PyInstaller cannot cross
compile): Linux on Linux, Windows on Windows. The result is written to
``dist/LicheeYOLOStudio`` (or ``dist/LicheeYOLOStudio.exe``).

    python -m pip install -r requirements.txt pyinstaller
    python tools/build_desktop.py

Linux: the script bundles the xcb libraries Qt needs (including
libxcb-cursor.so.0). Install the matching development/runtime packages before
building; on Debian/Ubuntu:

    sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 \
        libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
        libxcb-shape0 libxcb-xinerama0 libxcb-xkb1
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "LicheeYOLOStudio"

# Libraries dlopen()ed by the Qt xcb platform plugin. They are not always
# installed on minimal desktops, so put them inside the one-file bundle.
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


def main() -> int:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is not installed. Run:\n"
              "  python -m pip install -r requirements.txt pyinstaller",
              file=sys.stderr)
        return 1

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onefile", "--windowed",
        "--name", NAME,
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build" / "pyinstaller"),
        "--specpath", str(ROOT / "build"),
        "--paths", str(ROOT),
        "--hidden-import", "yaml",
        "--hidden-import", "paramiko",
        "--exclude-module", "tkinter",
        "--exclude-module", "matplotlib",
        "--exclude-module", "PyQt5",
        "--exclude-module", "PyQt6",
        str(ROOT / "app.py"),
    ]

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

    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)

    exe = ROOT / "dist" / (NAME + (".exe" if sys.platform == "win32" else ""))
    if not exe.exists():
        print(f"error: {exe} was not produced", file=sys.stderr)
        return 1
    print(f"\nDone: {exe} ({exe.stat().st_size / (1024 * 1024):.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
