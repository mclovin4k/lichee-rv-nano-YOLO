"""Export .pt (YOLOv8/v11) to ONNX with six "raw" head outputs.

The head's forward pass is replaced with one that returns the box/cls branches
for three scales:

    [box8, box16, box32, cls8, cls16, cls32]

This is exactly the format decoded by yolo8.c on the board (no concat/DFL/NMS).

Used both by the GUI (``--export-onnx`` in a frozen build) and by
``tools/export_yolo11.py`` on the command line.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


def export(pt: str | Path, imgsz: int = 640) -> Path:
    """Rewrite the head and export ``pt`` to ONNX next to it."""
    from ultralytics import YOLO  # imported late: needs torch

    model_path = str(pt)
    model = YOLO(model_path)
    print("names:", model.names)

    def forward2(self, x):
        x_reg = [self.cv2[i](x[i]) for i in range(self.nl)]
        x_cls = [self.cv3[i](x[i]) for i in range(self.nl)]
        return x_reg + x_cls

    head = model.model.model[-1]
    head.forward = types.MethodType(forward2, head)
    out = model.export(format="onnx", opset=11, imgsz=int(imgsz), dynamic=False)
    print("exported:", out)
    return Path(out)


def describe(onnx_path: str | Path) -> None:
    """Print the input/output shapes of the exported graph."""
    try:
        import onnx
    except ImportError:
        return
    m = onnx.load(str(onnx_path))
    print("inputs:")
    for i in m.graph.input:
        print("  ", i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
    print("outputs:")
    for o in m.graph.output:
        print("  ", o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    imgsz = int(argv[1]) if len(argv) > 1 else 640
    out = export(argv[0], imgsz)
    describe(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
