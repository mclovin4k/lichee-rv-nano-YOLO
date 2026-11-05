"""Locating the Python interpreters the pipeline needs.

Lichee YOLO Studio is a GUI around three command line tools:

* training and ONNX export run in a Python environment with ``ultralytics``:
  either the interpreter of the ``-full`` build (which embeds the whole ML
  stack) or an external environment chosen by the user;
* cvimodel conversion runs in a Linux environment with ``tpu-mlir``. On
  Windows that environment lives inside WSL 2 and is reached through
  ``wsl.exe``; the configured path is then a Linux path such as
  ``/home/<user>/.venvs/tpu310/bin/python``.

Nothing is hard coded: the interpreter is taken from the settings, from an
environment variable, or detected on ``PATH`` and in the usual virtualenv
locations. A configured path that no longer exists is treated as "not set"
(a Linux tpu-mlir path on Windows is checked inside WSL), so an old settings
file cannot break the application.
"""

from __future__ import annotations

import base64
import importlib.util
import os
import shlex
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
    Path.home() / "miniforge3" / "envs",
    APP_DIR / ".venv",
    APP_DIR.parent / ".venv",
)

# Directory names that are scanned for a virtualenv when the usual places are
# empty (a venv is often created next to the project it belongs to).
SKIP_DIRS = {
    ".git", ".hg", "node_modules", "site-packages", "AppData", "Library",
    "Windows", "Program Files", "Program Files (x86)", "pkgs", "cache",
    "__pycache__", ".cache", "Trash", "OneDrive", "Dropbox",
}

_PROBE_CACHE: dict[tuple[str, str], bool] = {}
_EXE_CACHE: dict[tuple[str, str], Path | None] = {}
_DETECT_CACHE: dict[str, Path | str | None] = {}
_ML_CACHE: dict[str, bool] = {}
_WSL_CACHE: dict[str, str | None] = {}

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
                              encoding="utf-8", errors="replace",
                              timeout=timeout, check=False, **kwargs)
    except (OSError, subprocess.SubprocessError):
        return None


def is_wsl_path(value: str | Path) -> bool:
    """True when a configured interpreter is a Linux path on Windows."""
    return WIN and str(value).strip().startswith("/")


def wsl_exe() -> str | None:
    """Path to wsl.exe, or None outside Windows / without WSL."""
    if not WIN:
        return None
    if "exe" not in _WSL_CACHE:
        _WSL_CACHE["exe"] = shutil.which("wsl.exe") or shutil.which("wsl")
    return _WSL_CACHE["exe"]


def _wsl_script_cmd(exe: str, script: str) -> list[str]:
    """Run a shell script in WSL without fighting Windows command-line quoting.

    ``wsl.exe`` joins its arguments into a Linux command line, so quotes inside
    a ``sh -c`` script can be mangled. Passing a base64 payload keeps the
    command line to plain ASCII words.
    """
    payload = base64.b64encode(script.encode("utf-8")).decode("ascii")
    return [exe, "-e", "sh", "-c", f"echo {payload} | base64 -d | sh"]


def _wsl_home() -> str | None:
    exe = wsl_exe()
    if exe is None:
        return None
    if "home" in _WSL_CACHE:
        return _WSL_CACHE["home"]
    result = _run([exe, "-e", "printenv", "HOME"], 30.0)
    home = None
    if result is not None and result.returncode == 0:
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if lines and lines[-1].startswith("/"):
            home = lines[-1]
    _WSL_CACHE["home"] = home
    return home


def _wsl_module_ok(python: str, module: str) -> bool:
    exe = wsl_exe()
    if exe is None:
        return False
    script = (f"exec {shlex.quote(python)} -c "
              f"{shlex.quote('import ' + module)}")
    result = _run(_wsl_script_cmd(exe, script), 120.0)
    return result is not None and result.returncode == 0


def _wsl_probe_tpu(configured: str = "", force: bool = False) -> str | None:
    """Find a tpu-mlir interpreter inside the default WSL distribution."""
    if force:
        _WSL_CACHE.pop("exe", None)
        _WSL_CACHE.pop("home", None)
    exe = wsl_exe()
    if exe is None:
        return None
    key = "tpu:" + configured.strip()
    if not force and key in _WSL_CACHE:
        return _WSL_CACHE[key]

    prefix = ""
    cfg = configured.strip()
    if cfg.startswith("~/"):
        prefix = '"$HOME"/' + shlex.quote(cfg[2:]) + " "
    elif cfg.startswith("/"):
        prefix = shlex.quote(cfg) + " "
    script = (
        "for p in " + prefix +
        '"$HOME/.venvs/tpu310/bin/python" '
        '"$HOME/venvs/tpu310/bin/python" '
        '"$HOME/.virtualenvs/tpu310/bin/python" '
        '"$(command -v python3)" "$(command -v python)"; do\n'
        '  [ -n "$p" ] || continue\n'
        '  [ -x "$p" ] || continue\n'
        '  if "$p" -c "import tpu_mlir" >/dev/null 2>&1; then\n'
        '    printf "%s\\n" "$p"\n'
        '    exit 0\n'
        '  fi\n'
        'done\n'
        'exit 1\n'
    )
    result = _run(_wsl_script_cmd(exe, script), timeout=240.0)
    found = None
    if result is not None and result.returncode == 0:
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if lines:
            found = lines[-1]
    _WSL_CACHE[key] = found
    _DETECT_CACHE["tpu"] = found
    return found


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


def deep_candidates() -> list[list[str]]:
    """Virtualenvs anywhere under the home directory (bounded search).

    Covers the common case of an environment created next to a project, which
    the fixed list above cannot know about.
    """
    found: list[list[str]] = []
    home = Path.home()
    for root, dirs, files in os.walk(home, topdown=True):
        depth = len(Path(root).parts) - len(home.parts)
        if depth >= 4:
            dirs[:] = []
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in ("python.exe", "python3", "python"):
            if name in files:
                candidate = Path(root) / name
                if candidate.parent.name in ("Scripts", "bin", "python_venv"):
                    found.append([str(candidate)])
                break
        if (Path(root) / "pyvenv.cfg").is_file():
            python = venv_python(Path(root))
            if python is not None:
                found.append([str(python)])
        if len(found) > 40:
            break
    return found


def detect(kind: str, configured: str = "", force: bool = False) -> Path | str | None:
    """Probe candidates and return the first interpreter that works."""
    if WIN and kind == "tpu":
        return _wsl_probe_tpu(configured, force)
    if not force and kind in _DETECT_CACHE:
        return _DETECT_CACHE[kind]
    module = KINDS[kind]["module"]
    seen: list[list[str]] = []
    for group in (candidates(kind, configured), deep_candidates()):
        for cmd in group:
            if cmd in seen:
                continue
            seen.append(cmd)
            if not has_module(cmd, module):
                continue
            path = interpreter_of(cmd)
            result = path if path is not None else Path(cmd[0])
            _DETECT_CACHE[kind] = result
            return result
    _DETECT_CACHE[kind] = None
    return None


def bundled_ml() -> bool:
    """True when this build already contains ultralytics and torch.

    The "full" one-folder build ships them, so no external environment is
    needed; the slim one-file build does not.
    """
    if "value" not in _ML_CACHE:
        try:
            found = importlib.util.find_spec("ultralytics") is not None
        except (ImportError, ValueError):
            found = False
        _ML_CACHE["value"] = found
    return _ML_CACHE["value"]


def bundled(kind: str) -> bool:
    """Only the ultralytics side can be embedded; tpu-mlir is Linux-only and
    is therefore always an external (possibly WSL) environment."""
    return kind == "ultra" and bundled_ml()


def resolve(kind: str, configured: str = "") -> Path | str | None:
    """Best interpreter for ``kind``: the bundle itself, an existing configured
    path, or a detected interpreter that can import the required module."""
    if WIN and kind == "tpu":
        return _wsl_probe_tpu(configured)
    if bundled(kind):
        return Path(sys.executable)
    configured = (configured or "").strip()
    if configured and Path(configured).is_file():
        return Path(configured)
    return detect(kind, configured)


def yolo_argv(python: str | Path) -> list[str] | None:
    """Command prefix for the ultralytics CLI."""
    if bundled_ml():
        return [sys.executable, "--yolo-cli"]
    yolo = sibling(python, "yolo")
    return [str(yolo)] if yolo is not None else None


def export_argv(python: str | Path, script: Path) -> list[str] | None:
    """Command prefix for the six-output ONNX export."""
    if bundled_ml():
        return [sys.executable, "--export-onnx"]
    if not Path(python).is_file():
        return None
    return [str(python), str(script)]


# --------------------------------------------------------------- setup


def base_pythons() -> list[list[str]]:
    """Interpreters that can create a virtualenv (any Python 3)."""
    found: list[list[str]] = []
    for cmd in candidates("ultra"):
        if cmd in found:
            continue
        if has_module(cmd, "venv", timeout=30.0):
            found.append(cmd)
    return found


def venv_target(kind: str) -> Path | str:
    """Where an environment created by the application is placed."""
    if kind == "tpu" and WIN:
        home = _wsl_home()
        return f"{home}/.venvs/tpu310" if home else "~/.venvs/tpu310"
    return Path.home() / ".venvs" / VENV_DIRS[kind][0]


def _stream(cmd: list[str], on_line=None, cancel=None) -> None:
    """Run a command, forwarding its output; raises on a non-zero exit."""
    kwargs: dict = {}
    if WIN:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1,
                            **kwargs)
    try:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            if on_line:
                on_line(line.rstrip())
            if cancel is not None and cancel.is_set():
                proc.terminate()
                raise RuntimeError("cancelled")
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if rc != 0:
        raise RuntimeError("command failed (exit %d): %s"
                           % (rc, " ".join(str(c) for c in cmd)))


def _create_wsl_env(target: str | Path | None, on_line, cancel) -> str:
    """Create the tpu-mlir virtualenv inside the default WSL distribution."""
    exe = wsl_exe()
    if exe is None:
        raise RuntimeError(
            "WSL 2 was not found. Install it from an administrator "
            "PowerShell:\n  wsl --install\n"
            "Reboot if asked, open the Ubuntu terminal once, then try again.")
    home = _wsl_home()
    if not home:
        raise RuntimeError(
            "WSL is installed, but its home directory could not be read. "
            "Open the WSL terminal once (wsl.exe), finish the first-run "
            "setup, then try again.")
    target_s = str(target) if target else f"{home}/.venvs/tpu310"
    if target_s.startswith("~/"):
        target_s = home + target_s[1:]
    if "\\" in target_s or (len(target_s) > 1 and target_s[1] == ":"):
        raise RuntimeError(
            "The tpu-mlir environment must live inside WSL, not on a Windows "
            f"path.\nUse a Linux path such as {home}/.venvs/tpu310.")
    target_s = target_s.rstrip("/") or f"{home}/.venvs/tpu310"
    python = target_s + "/bin/python"
    check = _run(_wsl_script_cmd(
        exe,
        'PY=python3.10; command -v "$PY" >/dev/null 2>&1 || PY=python3; '
        'command -v "$PY" >/dev/null 2>&1 || exit 2; '
        '"$PY" -m venv --help >/dev/null 2>&1 || exit 3'), 60.0)
    rc = check.returncode if check is not None else -1
    if rc == 2:
        raise RuntimeError(
            "Python 3 was not found inside WSL. Open the Ubuntu terminal and "
            "install it:\n  sudo apt update && sudo apt install python3 "
            "python3-venv python3-pip")
    if rc == 3:
        raise RuntimeError(
            "Python virtualenv support is missing inside WSL. Open the Ubuntu "
            "terminal and install it:\n"
            "  sudo apt update && sudo apt install python3-venv python3-pip\n"
            "On a distro with Python 3.10 use python3.10-venv instead, then "
            "try again.")
    if rc != 0:
        raise RuntimeError(
            "could not check the Python installation inside WSL; open the "
            "Ubuntu terminal once, finish the first-run setup, then try again.")
    if on_line:
        on_line(f"$ wsl: python3 -m venv {target_s}")
    script = (
        "set -e; PY=python3; "
        "if command -v python3.10 >/dev/null 2>&1; then PY=python3.10; fi; "
        f'"$PY" -m venv {shlex.quote(target_s)}; '
        f'{shlex.quote(python)} -m pip install --upgrade pip; '
        f'{shlex.quote(python)} -m pip install tpu-mlir'
    )
    _stream(_wsl_script_cmd(exe, script), on_line, cancel)
    for key in [k for k in _WSL_CACHE if k.startswith("tpu:")]:
        _WSL_CACHE.pop(key, None)
    _DETECT_CACHE.pop("tpu", None)
    if not _wsl_module_ok(python, "tpu_mlir"):
        raise RuntimeError(
            "the WSL environment was created, but \"import tpu_mlir\" still "
            "fails; check the pip output above")
    return python


def create_env(kind: str, target: str | Path | None = None, on_line=None,
               cancel=None) -> Path | str:
    """Create a virtualenv for ``kind`` and install the required package.

    Used by the "Create environment…" button: downloading the packages takes a
    while, so everything is streamed to the log.
    """
    info = KINDS[kind]
    if kind == "tpu" and WIN:
        return _create_wsl_env(target, on_line, cancel)
    bases = base_pythons()
    if not bases:
        raise RuntimeError(
            "no Python 3 interpreter was found. Install it from "
            "https://www.python.org/downloads/ (tick \"Add python.exe to "
            "PATH\"), then try again.")
    base = bases[0]
    target = Path(target) if target else Path(venv_target(kind))
    if on_line:
        on_line(f"$ {' '.join(base)} -m venv {target}")
    _stream(base + ["-m", "venv", str(target)], on_line, cancel)
    python = venv_python(target)
    if python is None:
        raise RuntimeError(f"the virtualenv in {target} has no interpreter")
    _stream([str(python), "-m", "pip", "install", "--upgrade", "pip"],
            on_line, cancel)
    _stream([str(python), "-m", "pip", "install", info["install"]],
            on_line, cancel)
    _PROBE_CACHE.clear()
    _DETECT_CACHE.pop(kind, None)
    if not has_module([str(python)], info["module"], timeout=180.0):
        raise RuntimeError(
            f"the environment was created, but \"import {info['module']}\" "
            "still fails; check the pip output above")
    return python



def missing_help(kind: str) -> str:
    """Actionable text for the 'not found' dialog."""
    info = KINDS[kind]
    if kind == "tpu":
        if WIN:
            return (
                "cvimodel conversion runs with tpu-mlir inside WSL 2.\n\n"
                "Click \"Create environment…\" in this dialog to create the "
                "environment in WSL automatically. If WSL 2 is not installed "
                "yet, install it first from an administrator PowerShell:\n"
                "  wsl --install\n"
                "Reboot if asked and finish the Ubuntu first-run setup.\n\n"
                "Or create it manually in the Ubuntu terminal:\n"
                "  sudo apt update && sudo apt install python3.10-venv\n"
                "  python3.10 -m venv ~/.venvs/tpu310\n"
                "  ~/.venvs/tpu310/bin/python -m pip install tpu-mlir\n\n"
                "Then set Settings → \"Python tpu-mlir\" to\n"
                "  /home/<user>/.venvs/tpu310/bin/python\n\n"
                "The ONNX file produced here can also be converted on a Linux "
                "host or on the board itself."
            )
        return (
            "Lichee YOLO Studio converts ONNX to cvimodel with tpu-mlir, which "
            "only runs on Linux.\n\n"
            "Create an environment (Python 3.10):\n"
            "  python3.10 -m venv ~/.venvs/tpu310\n"
            "  ~/.venvs/tpu310/bin/python -m pip install tpu-mlir\n\n"
            "Then set Settings → \"Python tpu-mlir\" to\n"
            "  ~/.venvs/tpu310/bin/python\n\n"
            "Click \"Detect\" in Settings to search automatically."
        )
    if WIN:
        return (
            "Training and ONNX export need a separate Python environment with "
            "the \"ultralytics\" package (this build does not contain it).\n\n"
            "Open PowerShell:\n"
            "  py -3 -m venv %USERPROFILE%\\.venvs\\ultra\n"
            "  %USERPROFILE%\\.venvs\\ultra\\Scripts\\python -m pip install ultralytics\n\n"
            "Then set Settings → \"Python Ultralytics\" to\n"
            "  %USERPROFILE%\\.venvs\\ultra\\Scripts\\python.exe\n\n"
            "Click \"Detect\" in Settings to search automatically."
        )
    return (
        "Training and ONNX export need a separate Python environment with the "
        "\"ultralytics\" package (this build does not contain it).\n\n"
        "  python3 -m venv ~/.venvs/ultra\n"
        "  ~/.venvs/ultra/bin/python -m pip install ultralytics\n\n"
        "Then set Settings → \"Python Ultralytics\" to\n"
        "  ~/.venvs/ultra/bin/python\n\n"
        "Click \"Detect\" in Settings to search automatically."
    )
