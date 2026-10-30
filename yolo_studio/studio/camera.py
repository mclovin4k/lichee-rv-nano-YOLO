"""Frame source: MJPEG from the board or a local USB camera."""

from __future__ import annotations

import os
import time

import cv2
from PySide6.QtCore import QThread, Signal


class CameraThread(QThread):
    frame = Signal(object)      # np.ndarray BGR
    status = Signal(str)

    def __init__(self, source: str | int, parent=None) -> None:
        super().__init__(parent)
        self.source = source
        self._running = True
        self._cap: cv2.VideoCapture | None = None

    def _open(self) -> cv2.VideoCapture | None:
        if isinstance(self.source, str) and self.source.startswith("http"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;5000000")
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(int(self.source), cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self) -> None:
        misses = 0
        while self._running:
            if self._cap is None:
                cap = self._open()
                if cap is None:
                    self.status.emit("camera failed to open, retrying…")
                    time.sleep(1.0)
                    continue
                self._cap = cap
                self.status.emit("camera connected")
            ok, frame = self._cap.read()
            if not ok or frame is None:
                misses += 1
                if misses >= 5:
                    self.status.emit("stream lost, reconnecting…")
                    self._release()
                    misses = 0
                    time.sleep(0.6)
                else:
                    time.sleep(0.05)
                continue
            misses = 0
            self.frame.emit(frame)
        self._release()

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
            self._cap = None

    def stop(self) -> None:
        self._running = False
        if not self.wait(1500):
            self._release()
            self.wait(1500)
