"""Model build: training, ONNX export, conversion to cvimodel."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path

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
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
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
    yolo_bin = Path(ultra_python).with_name("yolo")
    if not yolo_bin.is_file():
        raise PipelineError(f"no yolo CLI next to {ultra_python}")
    cmd = [
        str(yolo_bin), "detect", "train",
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
    if not script.is_file():
        raise PipelineError(f"script not found: {script}")
    run_stream([ultra_python, str(script), str(pt), str(int(imgsz))],
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
    env = os.environ.copy()
    env["TPU_PY"] = tpu_python
    env["NAME"] = "model"
    env["OUT"] = out_name
    env["CALIB_NUM"] = str(min(200, n_calib))
    env["IMGSZ"] = str(int(imgsz))
    env["PATH"] = str(Path(tpu_python).parent) + os.pathsep + env.get("PATH", "")

    cmd = ["sh", str(script), str(onnx), str(calib)]
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
