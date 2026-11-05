"""Capture page: camera connection, angle scenario, frame saving."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ..board import Board, BoardError
from ..camera import CameraThread
from ..dataset import Dataset, parse_size
from .theme import accent, primary
from .widgets import FrameView, Worker

STEPS: list[tuple[str, str, float]] = [
    ("Front", "Keep the object centered in the frame and show its front side", 3.0),
    ("Left", "Turn the object's left side toward the camera", 3.0),
    ("Right", "Turn the object's right side toward the camera", 3.0),
    ("Top", "Tilt the object and show its top side", 2.5),
    ("Bottom", "Show the bottom side of the object", 2.5),
    ("Close", "Move the object closer — about half a frame", 2.0),
    ("Far", "Move the object away — medium shot", 2.0),
    ("In context", "Show the object against its surroundings", 3.0),
]


class CapturePage(QWidget):
    def __init__(self, main) -> None:
        super().__init__()
        self.main = main
        self.cfg = main.config
        self._camera: CameraThread | None = None
        self._worker: Worker | None = None
        self._last_frame: np.ndarray | None = None
        self._last_show = 0.0
        self._clean_started = False
        self._active_step: int | None = None
        self._step_start = 0.0
        self._saved_session = 0
        self._skipped_session = 0

        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._capture_tick)

        self._build_ui()
        self._update_dataset_ui()
        self._update_buttons()

    # ---------- UI ----------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)
        top = QHBoxLayout()
        top.setSpacing(12)

        # --- dataset
        ds_box = QGroupBox("Dataset")
        ds_lay = QVBoxLayout(ds_box)
        ds_lay.setSpacing(8)
        self.ds_path = QLabel("not selected")
        self.ds_path.setWordWrap(True)
        self.ds_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        ds_lay.addWidget(self.ds_path)
        row = QHBoxLayout()
        self.btn_new = QPushButton("New…")
        self.btn_new.clicked.connect(self._new_dataset)
        self.btn_open = QPushButton("Open…")
        self.btn_open.clicked.connect(self._open_dataset)
        row.addWidget(self.btn_new)
        row.addWidget(self.btn_open)
        row.addStretch(1)
        ds_lay.addLayout(row)
        cls_row = QHBoxLayout()
        cls_label = QLabel("Classes:")
        cls_label.setFixedWidth(88)
        cls_row.addWidget(cls_label)
        self.cls_edit = QLineEdit()
        self.cls_edit.setPlaceholderText("e.g. drone or drone, person")
        self.cls_edit.setToolTip(
            "Comma-separated class names. Order = id: the first is 0, the second is 1.")
        cls_row.addWidget(self.cls_edit, 1)
        self.btn_class = QPushButton("OK")
        self.btn_class.setFixedWidth(56)
        self.btn_class.clicked.connect(self._save_class)
        cls_row.addWidget(self.btn_class)
        ds_lay.addLayout(cls_row)
        self.ds_stats = QLabel("")
        self.ds_stats.setObjectName("Muted")
        ds_lay.addWidget(self.ds_stats)

        size_row = QHBoxLayout()
        size_label = QLabel("Import size:")
        size_label.setFixedWidth(88)
        size_row.addWidget(size_label)
        self.import_size = QLineEdit(str(self.cfg.get("import_size", "keep")))
        self.import_size.setPlaceholderText("keep or 640x480")
        self.import_size.setToolTip(
            "Target size for imported images: 'keep' keeps the original size; "
            "640x480 letterboxes (gray padding) and transforms YOLO labels.")
        self.import_size.setFixedWidth(150)
        size_row.addWidget(self.import_size)
        size_row.addStretch(1)
        ds_lay.addLayout(size_row)
        self.btn_import = QPushButton("Import images from folder…")
        self.btn_import.setToolTip(
            "Copy JPEG/PNG/BMP images into the dataset. YOLO .txt labels are "
            "imported too (next to the image or in the sibling labels/ folder); "
            "a data.yaml/classes.txt in the source folder maps class ids by name.")
        self.btn_import.clicked.connect(self._import_images)
        accent(self.btn_import)
        ds_lay.addWidget(self.btn_import)
        self.import_status = QLabel("")
        self.import_status.setObjectName("Muted")
        self.import_status.setWordWrap(True)
        self.import_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        ds_lay.addWidget(self.import_status)

        # --- camera
        cam_box = QGroupBox("Camera")
        cam_lay = QFormLayout(cam_box)
        cam_lay.setSpacing(8)
        self.src_combo = QComboBox()
        self.src_combo.addItem("Board (clean stream)", "board")
        self.src_combo.addItem("USB camera", "usb")
        idx = self.src_combo.findData(self.cfg.get("camera_source"))
        self.src_combo.setCurrentIndex(max(0, idx))
        cam_lay.addRow("Source:", self.src_combo)

        self.usb_row = QWidget()
        usb_lay = QHBoxLayout(self.usb_row)
        usb_lay.setContentsMargins(0, 0, 0, 0)
        usb_lay.addWidget(QLabel("/dev/video"))
        self.usb_index = QSpinBox()
        self.usb_index.setRange(0, 16)
        self.usb_index.setValue(int(self.cfg.get("usb_index", 0)))
        usb_lay.addWidget(self.usb_index)
        usb_lay.addStretch(1)
        cam_lay.addRow(self.usb_row)

        self.board_row = QWidget()
        board_lay = QHBoxLayout(self.board_row)
        board_lay.setContentsMargins(0, 0, 0, 0)
        self.btn_prepare = QPushButton("Clean stream")
        self.btn_prepare.setToolTip(
            "Stop the YOLO stream on the board and start a stream without detection or boxes")
        self.btn_prepare.clicked.connect(self._prepare_board)
        self.btn_normal = QPushButton("Restore YOLO")
        self.btn_normal.setToolTip("Stop capture and restore the normal stream with boxes")
        self.btn_normal.clicked.connect(self._restore_board)
        board_lay.addWidget(self.btn_prepare)
        board_lay.addWidget(self.btn_normal)
        cam_lay.addRow(self.board_row)

        conn_row = QHBoxLayout()
        self.btn_connect = QPushButton("Connect camera")
        self.btn_connect.clicked.connect(self._connect_camera)
        primary(self.btn_connect)
        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.clicked.connect(self._disconnect_camera)
        conn_row.addWidget(self.btn_connect)
        conn_row.addWidget(self.btn_disconnect)
        cam_lay.addRow(conn_row)
        self.cam_status = QLabel("camera not connected")
        self.cam_status.setObjectName("Muted")
        self.cam_status.setWordWrap(True)
        cam_lay.addRow(self.cam_status)
        self.src_combo.currentIndexChanged.connect(self._source_changed)
        self._source_changed()

        top.addWidget(ds_box, 1)
        top.addWidget(cam_box, 1)
        root.addLayout(top)

        # --- video + scenario
        mid = QHBoxLayout()
        mid.setSpacing(12)
        self.view = FrameView()
        self.view.set_editable(False)
        mid.addWidget(self.view, 3)

        scen = QGroupBox("Capture scenario")
        scen_lay = QVBoxLayout(scen)
        scen_lay.setSpacing(8)
        self.steps_list = QListWidget()
        self.steps_list.setSpacing(2)
        self.steps_list.setMinimumHeight(190)
        for i, (name, _prompt, sec) in enumerate(STEPS):
            item = QListWidgetItem(f"{i + 1}. {name} — {sec:g} s")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.steps_list.addItem(item)
        self.steps_list.setCurrentRow(0)
        scen_lay.addWidget(self.steps_list)

        self.prompt = QLabel("Select a step and click “Start step”.\n"
                             "Frames are saved automatically while the countdown runs.")
        self.prompt.setObjectName("Prompt")
        self.prompt.setWordWrap(True)
        scen_lay.addWidget(self.prompt)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        scen_lay.addWidget(self.progress)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start step")
        self.btn_start.clicked.connect(self._start_step)
        primary(self.btn_start)
        self.btn_skip = QPushButton("Skip")
        self.btn_skip.clicked.connect(self._skip_step)
        self.btn_snap = QPushButton("Snapshot")
        self.btn_snap.clicked.connect(self._snapshot)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_skip)
        btn_row.addWidget(self.btn_snap)
        scen_lay.addLayout(btn_row)

        self.capture_stats = QLabel("saved this session: 0 · skipped: 0")
        self.capture_stats.setObjectName("Muted")
        self.capture_stats.setWordWrap(True)
        scen_lay.addWidget(self.capture_stats)
        scen_lay.addStretch(1)

        mid.addWidget(scen, 2)
        root.addLayout(mid, 1)

    # ---------- dataset ----------

    @property
    def dataset(self) -> Dataset | None:
        return self.main.dataset

    def _new_dataset(self) -> None:
        name, ok = QInputDialog.getText(self, "New dataset", "Dataset folder name:")
        if not ok or not name.strip():
            return
        cls, ok = QInputDialog.getText(
            self, "Object classes",
            "Comma-separated class names (Latin characters are easier):",
            text="object")
        if not ok:
            return
        class_names = [part.strip() for part in re.split(r"[,;]", cls)
                       if part.strip()] or ["object"]
        root = Path(str(self.cfg.get("dataset_root"))).expanduser() / name.strip()
        if root.exists() and any(root.iterdir()):
            if QMessageBox.question(self, "Dataset exists",
                                    f"{root} is not empty. Open it?") \
                    != QMessageBox.StandardButton.Yes:
                return
            try:
                ds = Dataset(root)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Error", str(exc))
                return
        else:
            try:
                ds = Dataset(root, create=True, class_names=class_names)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Error", str(exc))
                return
        self.main.set_dataset(ds)
        self._update_dataset_ui()

    def _open_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Open dataset", str(self.cfg.get("dataset_root")))
        if not path:
            return
        try:
            ds = Dataset(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Not a dataset", str(exc))
            return
        self.main.set_dataset(ds)
        self._update_dataset_ui()

    def _save_class(self) -> None:
        ds = self.dataset
        if ds is None:
            return
        names = [part.strip() for part in re.split(r"[,;]", self.cls_edit.text())
                 if part.strip()]
        if names and names != ds.class_names:
            ds.update_class_names(names)
            self._update_dataset_ui()
            self.main.notify_dataset_changed()

    def _import_images(self) -> None:
        ds = self.dataset
        if ds is None:
            QMessageBox.information(self, "No dataset",
                                    "Create or open a dataset first.")
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Import images", str(self.cfg.get("dataset_root")))
        if not folder:
            return
        try:
            size = parse_size(self.import_size.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Import size", str(exc))
            return
        self.cfg["import_size"] = self.import_size.text().strip() or "keep"
        self.cfg.save()
        self.import_status.setText("importing…")

        def task(cancel=None, on_line=None):
            def progress(done: int, total: int, name: str) -> None:
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("cancelled")
                if on_line is not None:
                    on_line(f"importing {done + 1}/{total}: {name}")
            return ds.import_images(folder, size=size, progress=progress)

        def done(res) -> None:
            text = (f"imported {res['images']} of {res['total']} · "
                    f"with labels: {res['labeled']} · skipped: {res['skipped']}")
            if res["size"]:
                text += f" · size: {res['size'][0]}×{res['size'][1]}"
            else:
                text += " · original size"
            self.import_status.setText(text)
            self._update_dataset_ui()
            self.main.notify_dataset_changed()
            if res["error_count"]:
                details = "\n".join(res["errors"][:10])
                QMessageBox.warning(
                    self, "Import finished with warnings",
                    f"{res['error_count']} file(s) were skipped or not labeled."
                    + (f"\n\n{details}" if details else ""))

        self._run(task, done, "Import failed",
                  on_error=self.import_status.setText,
                  on_line=self.import_status.setText)

    def _source_changed(self) -> None:
        board = self.src_combo.currentData() == "board"
        self.usb_row.setVisible(not board)
        self.board_row.setVisible(board)

    def _update_dataset_ui(self) -> None:
        ds = self.dataset
        if ds is None:
            self.ds_path.setText("not selected — create or open a dataset")
            self.ds_stats.setText("")
            self.cls_edit.clear()
            self.import_status.clear()
        else:
            self.ds_path.setText(str(ds.root))
            self.cls_edit.setText(", ".join(ds.class_names))
            st = ds.stats()
            self.ds_stats.setText(
                f"frames: {st['images']} · labeled: {st['labeled']} · "
                f"boxes: {st['boxes']}")
        self._update_buttons()

    def _update_buttons(self) -> None:
        has_ds = self.dataset is not None
        busy = self._worker is not None
        self.btn_start.setEnabled(has_ds and not busy and self._camera is not None
                                  and self._active_step is None)
        self.btn_snap.setEnabled(has_ds and not busy and self._camera is not None)
        self.btn_prepare.setEnabled(not busy and self._camera is None)
        self.btn_connect.setEnabled(not busy)
        self.btn_new.setEnabled(not busy)
        self.btn_open.setEnabled(not busy)
        self.btn_class.setEnabled(has_ds and not busy)
        self.btn_import.setEnabled(has_ds and not busy)
        self.import_size.setEnabled(has_ds and not busy)

    # ---------- board ----------

    def _board(self) -> Board:
        return self.main.board()

    def _prepare_board(self) -> None:
        if QMessageBox.question(
                self, "Capture mode",
                "The current YOLO stream on the board will be stopped "
                "and a clean stream without boxes will be started. Continue?") \
                != QMessageBox.StandardButton.Yes:
            return
        board = self._board()
        port = int(self.cfg.get("capture_port", 7778))
        self.cam_status.setText("preparing the board: stopping the stream…")

        def task(cancel=None, on_line=None):
            if on_line:
                on_line("stopping the stream…")
            board.stop_all()
            if on_line:
                on_line(f"starting the clean stream on :{port}…")
            board.start_clean(port)
            return board.stream_url(port)

        self._run(task, self._board_prepared, "Failed to prepare the board")

    def _restore_board(self) -> None:
        if QMessageBox.question(
                self, "Restore YOLO",
                "Stop the camera and restore the normal YOLO stream on the board?") \
                != QMessageBox.StandardButton.Yes:
            return
        self._stop_camera()
        board = self._board()
        self.cam_status.setText("restoring the normal stream…")

        def task(cancel=None, on_line=None):
            board.stop_all()
            board.start_normal()
            return "ok"

        def done(_res):
            self._clean_started = False
            self.cam_status.setText(
                f"normal stream: {board.stream_url(int(self.cfg.get('stream_port', 7777)))}")
            self._update_buttons()

        self._run(task, done, "Failed to restore the stream")

    def _board_prepared(self, url: str) -> None:
        self._clean_started = True
        self.cam_status.setText(f"clean stream ready: {url}")
        # the worker is still alive when this callback runs — connect on the next tick
        QTimer.singleShot(0, self._connect_camera)

    # ---------- camera ----------

    def _connect_camera(self) -> None:
        if self._worker is not None:
            return
        src = self.src_combo.currentData()
        self.cfg["camera_source"] = src
        self.cfg["usb_index"] = int(self.usb_index.value())
        self.cfg.save()
        self._stop_camera()
        if src == "board":
            if not self._clean_started:
                self._prepare_board()
                return
            url = self._board().stream_url(int(self.cfg.get("capture_port", 7778)))
            self._camera = CameraThread(url)
        else:
            self._camera = CameraThread(int(self.usb_index.value()))
        self._camera.frame.connect(self._on_frame)
        self._camera.status.connect(self.cam_status.setText)
        self._camera.start()
        self._update_buttons()
        self.cam_status.setText("connecting…")

    def _stop_camera(self) -> None:
        if self._camera is not None:
            cam, self._camera = self._camera, None
            if isinstance(cam, CameraThread):
                try:
                    cam.frame.disconnect(self._on_frame)
                    cam.status.disconnect()
                except (RuntimeError, TypeError):
                    pass
                cam.stop()
            self._last_frame = None
            self.view.clear()
        self._update_buttons()

    def _disconnect_camera(self) -> None:
        self._stop_camera()
        self.cam_status.setText("camera disconnected")

    def _on_frame(self, frame: np.ndarray) -> None:
        self._last_frame = frame
        now = time.monotonic()
        if now - self._last_show >= 0.05:
            self._last_show = now
            self.view.set_frame(frame)

    # ---------- scenario ----------

    def _start_step(self) -> None:
        ds = self.dataset
        if ds is None:
            QMessageBox.information(self, "No dataset", "Create or open a dataset.")
            return
        if self._last_frame is None:
            QMessageBox.information(self, "No frames",
                                    "Connect the camera first.")
            return
        row = self.steps_list.currentRow()
        if row < 0:
            row = self._next_unfinished()
        if row < 0:
            row = 0
        self._active_step = row
        self._step_start = time.monotonic()
        ds.reset_dedup()
        name, prompt, sec = STEPS[row]
        self.prompt.setText(f"{name}: {prompt} — {sec:g} s")
        self.progress.setValue(0)
        self._timer.start()
        self.btn_start.setEnabled(False)
        self.btn_skip.setEnabled(True)
        self._update_buttons()

    def _capture_tick(self) -> None:
        if self._active_step is None:
            return
        ds = self.dataset
        if ds is None:
            self._timer.stop()
            return
        _name, _prompt, sec = STEPS[self._active_step]
        elapsed = time.monotonic() - self._step_start
        if elapsed >= sec:
            self._finish_step()
            return
        self.progress.setValue(int(100.0 * elapsed / sec))
        if self._last_frame is None:
            return
        path, reason = ds.add_frame(self._last_frame)
        if path is not None:
            self._saved_session += 1
        else:
            self._skipped_session += 1
        self.capture_stats.setText(
            f"saved this session: {self._saved_session} · "
            f"skipped: {self._skipped_session} ({reason})")

    def _finish_step(self) -> None:
        self._timer.stop()
        idx = self._active_step
        self._active_step = None
        if idx is not None:
            item = self.steps_list.item(idx)
            item.setText(f"✓ {idx + 1}. {STEPS[idx][0]} — {STEPS[idx][2]:g} s")
            item.setForeground(QColor("#15803d"))
            self._next_unfinished(select=True)
        self.progress.setValue(100)
        nxt = self.steps_list.currentRow()
        if nxt >= 0 and nxt != idx:
            self.prompt.setText(
                f"Done. Next step: “{STEPS[nxt][0]}” — "
                "prepare the object and click “Start step”.")
        else:
            self.prompt.setText("Scenario complete. You can repeat a step or capture more frames.")
        self.btn_start.setEnabled(True)
        self.btn_skip.setEnabled(True)
        self._update_dataset_ui()

    def _skip_step(self) -> None:
        self._timer.stop()
        idx = self._active_step
        self._active_step = None
        if idx is not None:
            item = self.steps_list.item(idx)
            item.setText(f"— {idx + 1}. {STEPS[idx][0]} (skipped)")
            item.setForeground(QColor("#9aa4b2"))
            self._next_unfinished(select=True)
        self.progress.setValue(0)
        self.prompt.setText("Step skipped. Select the next one and click “Start step”.")
        self.btn_start.setEnabled(True)
        self._update_buttons()

    def _next_unfinished(self, select: bool = False) -> int:
        for i in range(self.steps_list.count()):
            text = self.steps_list.item(i).text()
            if not (text.startswith("✓") or text.startswith("—")):
                if select:
                    self.steps_list.setCurrentRow(i)
                return i
        if select:
            self.steps_list.setCurrentRow(-1)
        return -1

    def _snapshot(self) -> None:
        ds = self.dataset
        if ds is None or self._last_frame is None:
            return
        path, reason = ds.add_frame(self._last_frame, dedup=False)
        if path is not None:
            self._saved_session += 1
            self.capture_stats.setText(
                f"saved this session: {self._saved_session} · "
                f"skipped: {self._skipped_session}")
            self._update_dataset_ui()
        else:
            self.cam_status.setText(f"snapshot not saved: {reason}")

    # ---------- background tasks ----------

    def _run(self, fn, on_done, error_title: str,
             on_error=None, on_line=None) -> None:
        if self._worker is not None:
            return
        worker = Worker(fn)
        self._worker = worker
        self._update_buttons()
        if on_line is not None:
            worker.line.connect(on_line)
        if on_error is None:
            on_error = self.cam_status.setText

        def cleanup() -> None:
            self._worker = None
            self._update_buttons()

        worker.done.connect(lambda res: (on_done(res), cleanup()))
        worker.failed.connect(lambda msg: (QMessageBox.critical(self, error_title, msg),
                                           on_error(msg), cleanup()))
        worker.start()

    # ---------- misc ----------

    def refresh(self) -> None:
        self._update_dataset_ui()

    def showEvent(self, event) -> None:
        self._update_dataset_ui()
        super().showEvent(event)

    def shutdown(self) -> None:
        """Stops the camera and background tasks before the window closes."""
        self._timer.stop()
        self._stop_camera()
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(3000)
