#!/usr/bin/env python3
"""Export .pt (YOLOv8/v11) to ONNX with six "raw" head outputs.

Command line wrapper around ``studio/export_yolo11.py``; run it with the
Python that has ``ultralytics`` installed:

    ~/.venvs/ultra/bin/python tools/export_yolo11.py best.pt 640

In a frozen build the application sources are not importable, so the build
ships a copy under ``tools/studio/export_yolo11.py`` and this wrapper loads it
by path.

usage: export_yolo11.py <best.pt> [imgsz=640]
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = (
    HERE / "studio" / "export_yolo11.py",          # frozen bundle
    HERE.parent / "studio" / "export_yolo11.py",   # repository
)


def _load():
    for path in CANDIDATES:
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("studio_export_yolo11", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main
    raise SystemExit("export_yolo11.py: cannot find studio/export_yolo11.py")


if __name__ == "__main__":
    sys.exit(_load()())
