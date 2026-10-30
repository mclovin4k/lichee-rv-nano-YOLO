"""Shared widgets: frame view with boxes, background tasks, log."""

from __future__ import annotations

import threading
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QPlainTextEdit, QWidget

from ..dataset import Box


CLASS_COLORS = ("#31d17c", "#4ea1ff", "#ff9f43", "#c084fc",
                "#f472b6", "#22d3ee", "#facc15", "#a3e635")


def class_color(cls: int) -> QColor:
    return QColor(CLASS_COLORS[cls % len(CLASS_COLORS)])


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


class FrameView(QWidget):
    """Frame with boxes. In editable mode, boxes are drawn with the mouse."""

    box_added = Signal(object)          # Box
    box_selected = Signal(int)
    box_delete_requested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self.setMouseTracking(True)
        self._image: QImage | None = None
        self._boxes: list[Box] = []
        self._editable = False
        self._selected = -1
        self._drag_start: tuple[float, float] | None = None
        self._drag_cur: tuple[float, float] | None = None
        self._scale = 1.0
        self._offset = (0.0, 0.0)
        self._img_size = (0, 0)
        self._class_names = ["object"]
        self._new_cls = 0

    # ---------- data ----------

    def set_frame(self, frame: np.ndarray | QImage | None) -> None:
        if frame is None:
            self._image = None
            self._img_size = (0, 0)
        elif isinstance(frame, np.ndarray):
            self._image = bgr_to_qimage(frame)
            self._img_size = (frame.shape[1], frame.shape[0])
        else:
            self._image = frame
            self._img_size = (frame.width(), frame.height())
        self.update()

    def set_boxes(self, boxes: list[Box],
                  class_names: list[str] | str | None = None) -> None:
        self._boxes = [Box(b.x, b.y, b.w, b.h, b.cls, b.conf) for b in boxes]
        if class_names is not None:
            if isinstance(class_names, str):
                class_names = [class_names]
            self._class_names = list(class_names) or ["object"]
        self._selected = -1
        self.update()

    def set_new_class(self, cls: int) -> None:
        """Class assigned to the next drawn box."""
        self._new_cls = max(0, int(cls))

    def set_box_class(self, index: int, cls: int) -> None:
        if 0 <= index < len(self._boxes):
            self._boxes[index].cls = max(0, int(cls))
            self.update()

    def boxes(self) -> list[Box]:
        return list(self._boxes)

    def set_editable(self, editable: bool) -> None:
        self._editable = editable
        self.setCursor(Qt.CursorShape.CrossCursor if editable
                       else Qt.CursorShape.ArrowCursor)

    def clear(self) -> None:
        self._image = None
        self._boxes = []
        self._selected = -1
        self._img_size = (0, 0)
        self.update()

    # ---------- geometry ----------

    def _fit(self) -> tuple[float, float, float, float, float]:
        if self._image is None or not self._img_size[0]:
            return 0.0, 0.0, 0.0, 0.0, 1.0
        iw, ih = self._img_size
        scale = min(self.width() / iw, self.height() / ih)
        w, h = iw * scale, ih * scale
        x, y = (self.width() - w) / 2.0, (self.height() - h) / 2.0
        return x, y, w, h, scale

    def _to_image(self, px: float, py: float) -> tuple[float, float] | None:
        x, y, _w, _h, scale = self._fit()
        if scale <= 0:
            return None
        ix, iy = (px - x) / scale, (py - y) / scale
        iw, ih = self._img_size
        if 0 <= ix < iw and 0 <= iy < ih:
            return ix, iy
        return None

    def _box_at(self, ix: float, iy: float) -> int:
        for i in range(len(self._boxes) - 1, -1, -1):
            b = self._boxes[i]
            if b.x <= ix <= b.x2 and b.y <= iy <= b.y2:
                return i
        return -1

    # ---------- events ----------

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#0d0f12"))
        if self._image is None:
            painter.setPen(QColor("#6b7280"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "no frame — connect the camera")
            return
        x, y, w, h, scale = self._fit()
        painter.drawImage(QRectF(x, y, w, h), self._image)

        for i, b in enumerate(self._boxes):
            rect = QRectF(x + b.x * scale, y + b.y * scale,
                          max(1.0, b.w * scale), max(1.0, b.h * scale))
            color = QColor("#ffd23f") if i == self._selected else class_color(b.cls)
            painter.setPen(QPen(color, 2))
            painter.drawRect(rect)
            if 0 <= b.cls < len(self._class_names):
                label = self._class_names[b.cls]
            else:
                label = f"class {b.cls}"
            if b.conf is not None:
                label += f" {b.conf:.2f}"
            painter.setFont(QFont("Sans", 9, QFont.Weight.Bold))
            ty = rect.top() - 4
            if ty < 12:
                ty = rect.bottom() + 14
            painter.drawText(int(rect.left()) + 2, int(ty), label)

        if self._drag_start and self._drag_cur:
            x1, y1 = self._drag_start
            x2, y2 = self._drag_cur
            rx, ry = min(x1, x2), min(y1, y2)
            painter.setPen(QPen(QColor("#4ea1ff"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(x + rx * scale, y + ry * scale,
                                    abs(x2 - x1) * scale, abs(y2 - y1) * scale))

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pt = self._to_image(event.position().x(), event.position().y())
        if pt is None:
            return
        idx = self._box_at(*pt)
        if idx >= 0:
            self._selected = idx
            self.box_selected.emit(idx)
            self.update()
            return
        if self._selected >= 0:
            self._selected = -1
            self.box_selected.emit(-1)
        if self._editable:
            self._drag_start = pt
            self._drag_cur = pt
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is not None:
            pt = self._to_image(event.position().x(), event.position().y())
            if pt is not None:
                self._drag_cur = pt
                self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_start is None or event.button() != Qt.MouseButton.LeftButton:
            return
        start, self._drag_start = self._drag_start, None
        pt = self._to_image(event.position().x(), event.position().y())
        cur, self._drag_cur = self._drag_cur, None
        if pt is not None:
            cur = pt
        if cur is None:
            self.update()
            return
        x1, y1 = start
        x2, y2 = cur
        if abs(x2 - x1) >= 4 and abs(y2 - y1) >= 4:
            box = Box(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1),
                      self._new_cls)
            self._boxes.append(box)
            self._selected = len(self._boxes) - 1
            self.box_added.emit(box)
        self.update()

    def keyPressEvent(self, event) -> None:
        if (event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
                and self._selected >= 0):
            idx = self._selected
            self.box_delete_requested.emit(idx)
            if 0 <= idx < len(self._boxes):
                self._boxes.pop(idx)
            self._selected = -1
            self.update()
            return
        super().keyPressEvent(event)


class Worker(QThread):
    """Runs a function in the background; the function accepts cancel= and on_line=."""

    line = Signal(str)
    failed = Signal(str)
    done = Signal(object)

    def __init__(self, fn: Callable, *args, **kwargs) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            result = self._fn(*self._args, cancel=self._cancel,
                              on_line=self.line.emit, **self._kwargs)
            if not self._cancel.is_set():
                self.done.emit(result)
        except Exception as exc:  # noqa: BLE001 — show the error to the user
            if not self._cancel.is_set():
                self.failed.emit(str(exc) or exc.__class__.__name__)


class LogPane(QPlainTextEdit):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(4000)
        self.setPlaceholderText("build log…")

    def append_line(self, text: str) -> None:
        self.appendPlainText(text)
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())
