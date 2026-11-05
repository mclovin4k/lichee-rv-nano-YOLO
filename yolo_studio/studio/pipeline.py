"""Model build: training, ONNX export, conversion to cvimodel."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import threading
from pathlib import Path

from . import pyenv

TOOLS = Path(__file__).resolve().parent.parent / "tools"


class PipelineError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


def run_stream(cmd: list[str], cwd: str | Path | None = None,
               env: dict | None = None, on_line=None,
               cancel: threading.Event | None = None,
               timeout: float | None = None) -> int:
    """Runs a command, passing output lines to on_line."""
    kwargs: dict = {}
    if pyenv.is_win():
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, **kwargs)
    rc = -1
    try:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            if on_line:
                on_line(line.rstrip())
            if cancel is not None and cancel.is_set():
                _terminate(proc)
                raise Cancelled("cancelled")
        rc = proc.wait(timeout=timeout)
    finally:
        if proc.poll() is None:
            _terminate(proc)
    if rc != 0:
        raise PipelineError(f"command exited with code {rc}")
    return rc


def _terminate(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------- training ----------

def train_local(data_yaml: str | Path, work_dir: str | Path, ultra_python: str,
                epochs: int = 60, imgsz: int = 640, batch: int = 8,
                base_model: str = "yolo11n.pt", on_line=None,
                cancel: threading.Event | None = None) -> Path:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    # Either the bundled CLI (full build) or yolo/yolo.exe next to the
    # interpreter.
    argv = pyenv.yolo_argv(ultra_python)
    if argv is None:
        raise PipelineError(
            f"the 'yolo' command was not found next to {ultra_python}; "
            "install the tools into that environment:\n"
            f"  {ultra_python} -m pip install ultralytics")
    cmd = argv + [
        "detect", "train",
        f"data={Path(data_yaml)}",
        f"model={base_model}",
        f"epochs={int(epochs)}",
        f"imgsz={int(imgsz)}",
        f"batch={int(batch)}",
        "device=cpu",
        "workers=2",
        f"project={work}",
        "name=train",
        "exist_ok=True",
    ]
    run_stream(cmd, cwd=work, on_line=on_line, cancel=cancel)
    best = work / "train" / "weights" / "best.pt"
    if not best.is_file():
        raise PipelineError(f"training finished but {best} is missing")
    return best


# ---------- ONNX export ----------

def export_onnx(pt: str | Path, ultra_python: str, imgsz: int = 640,
                on_line=None, cancel: threading.Event | None = None) -> Path:
    pt = Path(pt)
    if not pt.is_file():
        raise PipelineError(f"weights not found: {pt}")
    script = TOOLS / "export_yolo11.py"
    argv = pyenv.export_argv(ultra_python, script)
    if argv is None:
        raise PipelineError(
            f"Cannot run the ONNX export with {ultra_python}; install "
            "ultralytics into that environment.")
    run_stream(argv + [str(pt), str(int(imgsz))],
               cwd=pt.parent, on_line=on_line, cancel=cancel)
    onnx = pt.with_suffix(".onnx")
    if not onnx.is_file():
        found = sorted(pt.parent.glob("*.onnx"))
        if not found:
            raise PipelineError("ONNX was not created after export")
        onnx = found[-1]
    return onnx


# ---------- cvimodel ----------

def count_calib(calib_dir: str | Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    d = Path(calib_dir)
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() in exts)


def _wsl_path(exe: str, path: str | Path) -> str:
    """Translate a Windows path with ``wslpath``; Linux paths pass through."""
    text = str(path)
    if text.startswith("/"):
        return text
    kwargs: dict = {}
    if pyenv.is_win():
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            [exe, "-e", "wslpath", "-a", text],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False, **kwargs)
    except OSError as exc:
        raise PipelineError(f"cannot run wsl.exe: {exc}") from exc
    if result.returncode != 0 or not result.stdout.strip():
        detail = (result.stderr or result.stdout).strip()
        raise PipelineError(
            "WSL could not translate a Windows path to Linux:\n"
            f"  {text}\n{detail}".rstrip())
    return result.stdout.strip().splitlines()[-1]


def _wsl_cvimodel_argv(script: Path, onnx: Path, calib: Path,
                       sample_image: Path | None, work: Path, tpu_python: str,
                       out_name: str, imgsz: int, calib_num: int) -> list[str]:
    """The conversion command, run by /bin/sh inside the default WSL distro."""
    exe = pyenv.wsl_exe()
    if exe is None:
        raise PipelineError(
            "WSL 2 was not found. Install it from an administrator PowerShell:\n"
            "  wsl --install\n"
            "Reboot if asked, open the Ubuntu terminal once, then click Build "
            "cvimodel again and choose \"Create WSL environment…\".\n"
            "The ONNX file produced here can also be converted on a Linux "
            "host or on the board itself.")
    tpu = str(tpu_python).strip()
    if not tpu.startswith("/"):
        raise PipelineError(
            "On Windows the tpu-mlir interpreter must be a Linux path inside "
            "WSL, for example:\n"
            "  /home/<user>/.venvs/tpu310/bin/python\n"
            f"The configured value is:\n  {tpu}")

    translated = {
        "script": _wsl_path(exe, script),
        "onnx": _wsl_path(exe, onnx),
        "calib": _wsl_path(exe, calib),
        "work": _wsl_path(exe, work),
    }
    if sample_image is not None:
        translated["sample"] = _wsl_path(exe, sample_image)

    assignments = " ".join([
        "TPU_PY=" + shlex.quote(tpu),
        "NAME=" + shlex.quote("model"),
        "OUT=" + shlex.quote(out_name),
        "CALIB_NUM=" + shlex.quote(str(calib_num)),
        "IMGSZ=" + shlex.quote(str(imgsz)),
    ])
    inner = (
        "cd " + shlex.quote(translated["work"]) + " && "
        "export " + assignments + " && "
        "exec /bin/sh " + shlex.quote(translated["script"])
        + " " + shlex.quote(translated["onnx"])
        + " " + shlex.quote(translated["calib"])
    )
    if "sample" in translated:
        inner += " " + shlex.quote(translated["sample"])
    return [exe, "-e", "sh", "-c", inner]


def build_cvimodel(onnx: str | Path, calib_dir: str | Path,
                   out_name: str, tpu_python: str, work_dir: str | Path,
                   sample_image: str | Path | None = None,
                   imgsz: int = 640,
                   on_line=None,
                   cancel: threading.Event | None = None) -> Path:
    onnx = Path(onnx)
    calib = Path(calib_dir)
    if not onnx.is_file():
        raise PipelineError(f"ONNX not found: {onnx}")
    n_calib = count_calib(calib)
    if n_calib == 0:
        raise PipelineError("no calibration frames — rebuild the dataset")
    script = TOOLS / "build_cvimodel.sh"
    if not script.is_file():
        raise PipelineError(f"script not found: {script}")

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if sample_image is None:
        sample_image = next((p for p in sorted(calib.iterdir())
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}),
                            None)
    n_calib = min(200, n_calib)

    if pyenv.is_win():
        cmd = _wsl_cvimodel_argv(script, onnx, calib,
                                 Path(sample_image) if sample_image else None,
                                 work, tpu_python, out_name, int(imgsz), n_calib)
        env = None
    else:
        shell = shutil.which("sh")
        if shell is None:
            raise PipelineError(
                "cvimodel conversion needs a POSIX shell and tpu-mlir. "
                "Install a POSIX shell or run the conversion on a Linux host.")
        env = os.environ.copy()
        env["TPU_PY"] = tpu_python
        env["NAME"] = "model"
        env["OUT"] = out_name
        env["CALIB_NUM"] = str(n_calib)
        env["IMGSZ"] = str(int(imgsz))
        env["PATH"] = str(Path(tpu_python).parent) + os.pathsep + env.get("PATH", "")
        cmd = [shell, str(script), str(onnx), str(calib)]
        if sample_image is not None:
            cmd.append(str(sample_image))

    run_stream(cmd, cwd=work, env=env, on_line=on_line, cancel=cancel)

    built = work / out_name
    if not built.is_file():
        candidates = sorted(work.glob("*.cvimodel"))
        if not candidates:
            raise PipelineError("cvimodel was not created")
        built = candidates[-1]
    final = onnx.parent / out_name
    shutil.copy2(built, final)
    return final
