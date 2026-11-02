#!/usr/bin/env python3
"""Export .pt (YOLOv8/v11) to ONNX with six "raw" head outputs.

Thin command line wrapper around ``studio/export_yolo11.py``; run it with the
Python that has ``ultralytics`` installed:

    ~/.venvs/ultra/bin/python tools/export_yolo11.py best.pt 640

usage: export_yolo11.py <best.pt> [imgsz=640]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.export_yolo11 import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
