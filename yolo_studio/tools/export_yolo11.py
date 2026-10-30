#!/usr/bin/env python3
"""Export .pt (YOLOv8/v11) to ONNX with six "raw" head outputs.

The head's forward pass is replaced with one that returns the box/cls branches
for three scales:
  [box8, box16, box32, cls8, cls16, cls32]
— this is exactly the format decoded by yolo8.c on the board (no concat/DFL/NMS).

usage: export_yolo11.py <best.pt> [imgsz=640]
"""
import sys
import types

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(2)

from ultralytics import YOLO

model_path = sys.argv[1]
imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 640


def forward2(self, x):
    x_reg = [self.cv2[i](x[i]) for i in range(self.nl)]
    x_cls = [self.cv3[i](x[i]) for i in range(self.nl)]
    return x_reg + x_cls


model = YOLO(model_path)
print("names:", model.names)

model.model.model[-1].forward = types.MethodType(forward2, model.model.model[-1])
out = model.export(format="onnx", opset=11, imgsz=imgsz, dynamic=False)
print("exported:", out)

import onnx

m = onnx.load(out)
print("inputs:")
for i in m.graph.input:
    print("  ", i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
print("outputs:")
for o in m.graph.output:
    print("  ", o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
