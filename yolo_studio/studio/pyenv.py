"""Locating the external Python interpreters the pipeline needs.

Lichee YOLO Studio is a GUI around three command line tools:

* training and ONNX export run in a Python environment with ``ultralytics``
  (PyTorch is far too large to bundle, and it must be a real, separate
  interpreter);
* cvimodel conversion runs in a Linux environment with ``tpu-mlir``.

Nothing is hard coded: the interpreter is taken from the settings, from an
environment variable, or detected on ``PATH`` and in the usual virtualenv
locations. A configured path that no longer exists is treated as "not set",
so an old settings file cannot break the application.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent

KINDS: dict[str, dict[str, str]] = {
    "ultra": {
        "config_key": "ultra_python",
        "env": "YOLO_STUDIO_ULTRAPYTHON",
        "module": "ultralytics",
        "label": "Python Ultralytics",
        "purpose": "training and ONNX export",
        "install": "ultralytics",
    },
    "tpu": {
        "config_key": "tpu_python",
        "env": "YOLO_STUDIO_TPUPYTHON",
        "module": "tpu_mlir",
        "label": "Python tpu-mlir",
        "purpose": "cvimodel conversion",
        "install": "tpu-mlir",
    },
}

# Where a virtualenv for these tools is usually created. Both layouts are
# checked, so a Linux-style path also works on Windows and vice versa.
VENV_DIRS = {
    "ultra": ("ultra", "yolo", "ultralytics"),
    "tpu": ("tpu310", "tpu", "tpumlir", "tpu-mlir"),
}
VENV_PARENTS = (
    Path.home() / ".venvs",
    Path.home() / "venvs",
    Path.home() / ".virtualenvs",
    Path.home() / "miniconda3" / "envs",
    Path.home() / "anaconda3" / "envs",
    APP_DIR / ".venv",
    APP_DIR.parent / ".venv",
)

_PROBE_CACHE: dict[tuple[str, str], bool] = {}
_EXE_CACHE: dict[tuple[str, str], Path | None] = {}

WIN = sys.platform.startswith("win")


def is_win() -> bool:
    return WIN


def venv_python(venv: Path) -> Path | None:
    """python executable inside a virtualenv directory, if it exists."""
    for rel in ("Scripts/python.exe", "bin/python", "python.exe"):
        candidate = venv / rel
        if candidate.is_file():
            return candidate
    return None


def sibling(python: str | Path, name: str) -> Path | None:
    """A console script that lives next to the interpreter (yolo, yolo.exe…)."""
    directory = Path(python).parent
    suffixes = (".exe", ".cmd", ".bat", "") if WIN else ("",)
    for suffix in suffixes:
        candidate = directory / (name + suffix)
        if candidate.is_file():
            return candidate
    return None


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess | None:
    kwargs: dict = {}
    if WIN:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, check=False, **kwargs)
    except (OSError, subprocess.SubprocessError):
        return None


def has_module(cmd: list[str], module: str, timeout: float = 60.0) -> bool:
    """True if ``cmd`` can import ``module``. Results are cached."""
    key = (" ".join(cmd), module)
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    result = _run(cmd + ["-c", f"import {module}"], timeout)
    ok = result is not None and result.returncode == 0
    _PROBE_CACHE[key] = ok
    return ok


def interpreter_of(cmd: list[str], timeout: float = 60.0) -> Path | None:
    """The real executable behind a launcher such as ``py -3``."""
    key = (" ".join(cmd), "executable")
    if key in _EXE_CACHE:
        return _EXE_CACHE[key]
    result = _run(cmd + ["-c", "import sys; print(sys.executable)"], timeout)
    path = None
    if result is not None and result.returncode == 0:
        line = result.stdout.strip().splitlines()[-1:] or [""]
        if line[0] and Path(line[0]).is_file():
            path = Path(line[0])
    _EXE_CACHE[key] = path
    return path


def candidates(kind: str, configured: str = "") -> list[list[str]]:
    """Interpreters worth probing, most specific first."""
    info = KINDS[kind]
    found: list[list[str]] = []

    def add(cmd: list[str]) -> None:
        if cmd and cmd not in found:
            found.append(cmd)

    configured = (configured or "").strip()
    if configured:
        add([configured])
    env = os.environ.get(info["env"], "").strip()
    if env:
        add([env])
    if not getattr(sys, "frozen", False) and sys.executable:
        add([sys.executable])
    for parent in VENV_PARENTS:
        for name in VENV_DIRS[kind]:
            python = venv_python(parent / name)
            if python is not None:
                add([str(python)])
    for name in (("py", "python", "python3") if WIN else ("python3", "python")):
        exe = shutil.which(name)
        if exe:
            # The Windows launcher needs the version selector; real
            # interpreters are called directly.
            add([exe, "-3"] if Path(exe).name.lower() in ("py", "py.exe")
                else [exe])
    return found


def resolve(kind: str, configured: str = "") -> Path | None:
    """Best interpreter for ``kind``: an existing configured path, else a
    detected one that can actually import the required module."""
    configured = (configured or "").strip()
    if configured and Path(configured).is_file():
        return Path(configured)
    return detect(kind, configured)


def detect(kind: str, configured: str = "") -> Path | None:
    """Probe candidates and return the first interpreter that works."""
    module = KINDS[kind]["module"]
    for cmd in candidates(kind, configured):
        if not has_module(cmd, module):
            continue
        path = interpreter_of(cmd)
        return path if path is not None else Path(cmd[0])
    return None


def missing_help(kind: str) -> str:
    """Actionable text for the 'not found' dialog."""
    info = KINDS[kind]
    if kind == "tpu":
        return (
            "Lichee YOLO Studio converts ONNX to cvimodel with tpu-mlir, which "
            "only runs on Linux.\n\n"
            "On Linux (or inside WSL on Windows):\n"
            "  python3 -m venv ~/.venvs/tpu310\n"
            "  ~/.venvs/tpu310/bin/python -m pip install tpu-mlir\n\n"
            "Then set Settings → \"Python tpu-mlir\" to\n"
            "  ~/.venvs/tpu310/bin/python\n\n"
            "Windows users: build the cvimodel in WSL or on the board itself; "
            "the ONNX file can be produced here."
        )
    if WIN:
        return (
            "Training and ONNX export need a separate Python environment with "
            "the \"ultralytics\" package (PyTorch is not bundled).\n\n"
            "Open PowerShell:\n"
            "  py -3 -m venv %USERPROFILE%\\.venvs\\ultra\n"
            "  %USERPROFILE%\\.venvs\\ultra\\Scripts\\python -m pip install ultralytics\n\n"
            "Then set Settings → \"Python Ultralytics\" to\n"
            "  %USERPROFILE%\\.venvs\\ultra\\Scripts\\python.exe\n\n"
            "Click \"Detect\" in Settings to search automatically."
        )
    return (
        "Training and ONNX export need a separate Python environment with the "
        "\"ultralytics\" package (PyTorch is not bundled).\n\n"
        "  python3 -m venv ~/.venvs/ultra\n"
        "  ~/.venvs/ultra/bin/python -m pip install ultralytics\n\n"
        "Then set Settings → \"Python Ultralytics\" to\n"
        "  ~/.venvs/ultra/bin/python\n\n"
        "Click \"Detect\" in Settings to search automatically."
    )
