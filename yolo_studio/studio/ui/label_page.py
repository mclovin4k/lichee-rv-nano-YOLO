"""Labeling page: draw boxes with the mouse and track them with optical flow."""

from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QComboBox, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QProgressDialog, QPushButton, QVBoxLayout,
    QWidget,
)

from ..dataset import Box, Dataset, imread
from ..tracking import track_boxes
from .widgets import FrameView


class LabelPage(QWidget):
    def __init__(self, main) -> None:
        super().__init__()
        self.main = main
        self._images: list[Path] = []
        self._current_img: Path | None = None
        self._current_row = -1
        self._boxes: list[Box] = []
        self._size: tuple[int, int] = (0, 0)
        self._dirty = False
        self._loading = False
        self._selected_box = -1
        self._build_ui()
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=self._next_image)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=self._prev_image)

    @property
    def dataset(self) -> Dataset | None:
        return self.main.dataset

    # ---------- UI ----------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        left.addWidget(QLabel("Frames (✓ — labeled)"))
        self.list = QListWidget()
        self.list.setMinimumWidth(230)
        self.list.currentRowChanged.connect(self._row_changed)
        left.addWidget(self.list, 1)
        self.stats = QLabel("")
        self.stats.setWordWrap(True)
        left.addWidget(self.stats)
        root.addLayout(left)

        center = QVBoxLayout()
        self.view = FrameView()
        self.view.set_editable(True)
        self.view.box_added.connect(self._box_added)
        self.view.box_selected.connect(self._box_selected)
        self.view.box_delete_requested.connect(self._box_deleted)
        center.addWidget(self.view, 1)

        nav = QHBoxLayout()
        btn_prev = QPushButton("← Previous")
        btn_prev.clicked.connect(self._prev_image)
        btn_next = QPushButton("Next →")
        btn_next.clicked.connect(self._next_image)
        nav.addWidget(btn_prev)
        nav.addWidget(btn_next)
        nav.addStretch(1)
        self.hint = QLabel("Draw a box with the mouse. Delete removes the selected one.")
        nav.addWidget(self.hint)
        center.addLayout(nav)
        root.addLayout(center, 1)

        right = QVBoxLayout()
        box = QGroupBox("Labeling")
        bl = QVBoxLayout(box)
        bl.addWidget(QLabel("Box class:"))
        self.cls_combo = QComboBox()
        self.cls_combo.setToolTip(
            "Class for new boxes. To change the class of an existing box, "
            "select it and change the class in the list.")
        self.cls_combo.currentIndexChanged.connect(self._class_changed)
        bl.addWidget(self.cls_combo)
        self.cls_label = QLabel("")
        bl.addWidget(self.cls_label)

        self.btn_save = QPushButton("Save boxes")
        self.btn_save.clicked.connect(self._save_current)
        bl.addWidget(self.btn_save)

        self.btn_track = QPushButton("Track forward →")
        self.btn_track.setToolTip(
            "Propagate boxes to the following frames using optical flow and save")
        self.btn_track.clicked.connect(self._track_forward)
        bl.addWidget(self.btn_track)

        self.btn_clear = QPushButton("Clear boxes")
        self.btn_clear.clicked.connect(self._clear_boxes)
        bl.addWidget(self.btn_clear)

        self.btn_delete = QPushButton("Delete frame")
        self.btn_delete.clicked.connect(self._delete_image)
        bl.addWidget(self.btn_delete)

        bl.addStretch(1)
        self.box_info = QLabel("boxes: 0")
        bl.addWidget(self.box_info)
        right.addWidget(box)
        root.addLayout(right)

    # ---------- list ----------

    def refresh(self) -> None:
        """Re-reads the frame list while keeping the current frame and unsaved boxes."""
        ds = self.dataset
        old_img = self._current_img
        old_dirty = self._dirty
        self._images = ds.image_paths() if ds else []
        self._loading = True
        self.list.clear()
        for img in self._images:
            labeled = ds.is_labeled(img) if ds else False
            count = len(ds.load_label(img)) if labeled else 0
            item = QListWidgetItem(
                f"{'✓' if labeled else '•'} {img.name}" +
                (f" · {count}" if count else ""))
            item.setData(Qt.ItemDataRole.UserRole, str(img))
            if labeled:
                item.setForeground(QColor("#31d17c"))
            self.list.addItem(item)

        row = -1
        if old_img is not None and old_img in self._images:
            row = self._images.index(old_img)
        elif self._images:
            row = 0
        if row >= 0:
            self.list.setCurrentRow(row)
        self._loading = False

        same = old_img is not None and row >= 0 and self._images[row] == old_img
        if same and old_dirty:
            self._current_row = row
            self._current_img = old_img
            item = self.list.item(row)
            if item is not None:
                item.setText(f"✓ {old_img.name} · {len(self._boxes)}"
                             if self._boxes else f"✓ {old_img.name}")
                item.setForeground(QColor("#31d17c"))
        elif row >= 0:
            self._current_row = row
            self._load_image(self._images[row])
        else:
            self._current_img = None
            self._current_row = -1
            self._boxes = []
            self._dirty = False
            self._selected_box = -1
            self.view.clear()
            self.view.set_boxes([], self._class_names())

        self._fill_classes()
        self._update_stats()

    def _class_names(self) -> list[str]:
        ds = self.dataset
        return ds.class_names if ds else ["object"]

    def _fill_classes(self) -> None:
        names = self._class_names()
        current = self.cls_combo.currentIndex()
        self.cls_combo.blockSignals(True)
        self.cls_combo.clear()
        for i, name in enumerate(names):
            self.cls_combo.addItem(f"{i}: {name}", i)
        if not 0 <= current < len(names):
            current = 0
        self.cls_combo.setCurrentIndex(current)
        self.cls_combo.blockSignals(False)
        self.view.set_new_class(current)
        self.cls_label.setText(f"classes in dataset: {len(names)}")

    def _class_changed(self, index: int) -> None:
        if index < 0:
            return
        self.view.set_new_class(index)
        if 0 <= self._selected_box < len(self._boxes):
            self._boxes[self._selected_box].cls = index
            self.view.set_box_class(self._selected_box, index)
            self._dirty = True
            self.hint.setText(f"box class: {self.cls_combo.currentText()}")

    def _box_selected(self, index: int) -> None:
        self._selected_box = index
        if 0 <= index < len(self._boxes):
            cls = self._boxes[index].cls
            self.cls_combo.blockSignals(True)
            if 0 <= cls < self.cls_combo.count():
                self.cls_combo.setCurrentIndex(cls)
            self.cls_combo.blockSignals(False)
            self.view.set_new_class(cls)

    def _update_stats(self) -> None:
        ds = self.dataset
        if ds is None:
            self.stats.setText("no dataset selected")
            return
        st = ds.stats()
        self.stats.setText(
            f"frames: {st['images']}\nlabeled: {st['labeled']}\n"
            f"unlabeled: {st['unlabeled']}\nboxes: {st['boxes']}")
        self.box_info.setText(f"boxes on frame: {len(self._boxes)}")

    def showEvent(self, event) -> None:
        self.refresh()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        if self._dirty:
            self._save_current(silent=True)
        super().hideEvent(event)

    # ---------- navigation ----------

    def _row_changed(self, row: int) -> None:
        if self._loading:
            return
        if self._dirty and self._current_img is not None:
            self._save_current(silent=True)
        self._current_row = row
        if 0 <= row < len(self._images):
            self._load_image(self._images[row])
        else:
            self._current_img = None
            self._boxes = []
            self.view.clear()

    def _load_image(self, img: Path) -> None:
        ds = self.dataset
        frame = imread(img)
        if ds is None or frame is None:
            self.hint.setText(f"cannot read {img.name}")
            return
        ih, iw = frame.shape[:2]
        self._current_img = img
        self._size = (iw, ih)
        self._boxes = ds.load_label(img)
        self._dirty = False
        self._selected_box = -1
        self.view.set_frame(frame)
        self.view.set_boxes(self._boxes, self._class_names())
        self.view.set_new_class(self.cls_combo.currentIndex())
        self.hint.setText(f"{img.name} · {iw}×{ih} · "
                          "draw a box with the mouse, Delete removes it")
        self._update_stats()

    def _prev_image(self) -> None:
        row = self.list.currentRow()
        if row > 0:
            self.list.setCurrentRow(row - 1)

    def _next_image(self) -> None:
        row = self.list.currentRow()
        if row < len(self._images) - 1:
            self.list.setCurrentRow(row + 1)

    # ---------- boxes ----------

    def _box_added(self, box: Box) -> None:
        box.cls = self.cls_combo.currentIndex()
        self._boxes.append(box)
        self._selected_box = len(self._boxes) - 1
        self._dirty = True
        self.view.set_new_class(box.cls)
        self._update_stats()

    def _box_deleted(self, index: int) -> None:
        if 0 <= index < len(self._boxes):
            self._boxes.pop(index)
        self._selected_box = -1
        self._dirty = True
        self.view.set_boxes(self._boxes, self._class_names())
        self._update_stats()

    def _clear_boxes(self) -> None:
        self._boxes = []
        self._selected_box = -1
        self._dirty = True
        self.view.set_boxes([], self._class_names())
        self._update_stats()

    def _save_current(self, silent: bool = False) -> None:
        ds = self.dataset
        if ds is None or self._current_img is None:
            return
        try:
            ds.save_label(self._current_img, self._boxes, self._size)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save error", str(exc))
            return
        self._dirty = False
        if 0 <= self._current_row < self.list.count():
            item = self.list.item(self._current_row)
            img = self._current_img
            if item is not None:
                item.setText(f"✓ {img.name} · {len(self._boxes)}"
                             if self._boxes else f"✓ {img.name}")
                item.setForeground(QColor("#31d17c"))
        if not silent:
            self.hint.setText("labels saved")
        self._update_stats()

    def _delete_image(self) -> None:
        ds = self.dataset
        img = self._current_img
        if ds is None or img is None:
            return
        if QMessageBox.question(self, "Delete frame", f"Delete {img.name}?") \
                != QMessageBox.StandardButton.Yes:
            return
        row = self._current_row
        ds.delete_image(img)
        self._current_img = None
        self._dirty = False
        self.refresh()
        if self._images:
            self.list.setCurrentRow(min(max(0, row), len(self._images) - 1))

    # ---------- tracking ----------

    def _track_forward(self) -> None:
        ds = self.dataset
        if ds is None or self._current_img is None:
            return
        if not self._boxes:
            QMessageBox.information(self, "No boxes",
                                    "Draw a box on the current frame first.")
            return
        self._save_current(silent=True)

        row = self._current_row
        last = len(self._images) - 1
        if row >= last:
            self.hint.setText("this is the last frame")
            return

        prev_frame = imread(self._current_img)
        if prev_frame is None:
            return
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        boxes = list(self._boxes)

        progress = QProgressDialog("Tracking boxes…", "Cancel", row, last, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(300)
        progress.setValue(row)

        lost = 0
        processed = row
        for i in range(row + 1, last + 1):
            if progress.wasCanceled():
                break
            img = self._images[i]
            frame = imread(img)
            if frame is None:
                break
            ih, iw = frame.shape[:2]
            if (iw, ih) != self._size:
                QMessageBox.warning(
                    self, "Size mismatch",
                    f"{img.name}: {iw}×{ih}, expected {self._size[0]}×{self._size[1]}.\n"
                    "Tracking stopped.")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes, quality = track_boxes(prev_gray, gray, boxes)
            lost = lost + 1 if max(quality) <= 0.05 else 0
            if lost >= 3:
                QMessageBox.warning(
                    self, "Tracking lost",
                    "Object lost for three consecutive frames.\n"
                    "Label the frame manually and continue.")
                break
            try:
                ds.save_label(img, boxes, (iw, ih))
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Error", str(exc))
                break
            prev_gray = gray
            processed = i
            progress.setValue(i)
            QApplication.processEvents()

        progress.setValue(last)
        progress.close()
        self.refresh()
        if processed > row:
            self.list.setCurrentRow(processed)
            self.hint.setText(f"tracking: labeled {processed - row} frames")
