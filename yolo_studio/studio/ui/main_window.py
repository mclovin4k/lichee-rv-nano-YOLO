"""Main window: tabs, shared dataset, settings."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog, QLabel, QMainWindow, QMessageBox, QSizePolicy, QTabWidget,
    QToolBar, QVBoxLayout, QWidget,
)

from ..board import Board
from ..config import Config
from ..dataset import Dataset
from .board_page import BoardPage
from .capture_page import CapturePage
from .label_page import LabelPage
from .model_page import ModelPage
from .settings_dialog import SettingsDialog


def classes_text(ds: Dataset) -> str:
    word = "class" if len(ds.class_names) == 1 else "classes"
    return f"{word}: {', '.join(ds.class_names)}"


class MainWindow(QMainWindow):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.dataset: Dataset | None = None
        self.last_cvimodel: Path | None = None
        self.setWindowTitle("Lichee YOLO Studio — dataset and cvimodel for LicheeRV Nano")
        self.setMinimumSize(1120, 720)
        self.resize(1380, 880)
        self._build_ui()
        self._load_last()

    def _build_ui(self) -> None:
        bar = QToolBar("Main")
        bar.setObjectName("AppBar")
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(bar)

        dot = QLabel("●")
        dot.setObjectName("BrandDot")
        bar.addWidget(dot)
        brand = QLabel("Lichee YOLO Studio")
        brand.setObjectName("Brand")
        bar.addWidget(brand)
        tagline = QLabel("dataset → label → train → deploy")
        tagline.setObjectName("BrandSub")
        bar.addWidget(tagline)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)

        self.path_label = QLabel("no dataset selected")
        self.path_label.setObjectName("PathChip")
        self.path_label.setMaximumWidth(420)
        self.path_label.setToolTip("Open a dataset to see its frames and classes")
        bar.addWidget(self.path_label)
        bar.addSeparator()

        act_open = QAction("Open dataset…", self)
        act_open.triggered.connect(self._open_dataset)
        bar.addAction(act_open)
        act_settings = QAction("Settings…", self)
        act_settings.triggered.connect(self._settings)
        bar.addAction(act_settings)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 14, 16, 8)
        layout.setSpacing(10)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.capture = CapturePage(self)
        self.label = LabelPage(self)
        self.model = ModelPage(self)
        self.board_page = BoardPage(self)
        self.tabs.addTab(self.capture, "1. Capture")
        self.tabs.addTab(self.label, "2. Label")
        self.tabs.addTab(self.model, "3. Model")
        self.tabs.addTab(self.board_page, "4. Board")
        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Ready")
        device = QLabel("LicheeRV Nano · SG2002/CV181x")
        device.setObjectName("DeviceChip")
        self.statusBar().addPermanentWidget(device)

    # ---------- shared access ----------

    def board(self) -> Board:
        return Board(
            str(self.config.get("board_ip")),
            str(self.config.get("board_user")) or "root",
            str(self.config.get("board_password")),
            stream_port=int(self.config.get("stream_port", 7777)))

    @staticmethod
    def _dataset_chip(ds: Dataset) -> str:
        return f"{ds.root.name}  ·  {classes_text(ds)}"

    def set_dataset(self, ds: Dataset) -> None:
        self.dataset = ds
        self.config["last_dataset"] = str(ds.root)
        self.config.save()
        self.path_label.setText(self._dataset_chip(ds))
        self.path_label.setToolTip(str(ds.root))
        self.capture._update_dataset_ui()
        self.label.refresh()
        self.model.refresh()
        self.statusBar().showMessage(f"dataset: {ds.root}")

    def notify_dataset_changed(self) -> None:
        if self.dataset is not None:
            self.path_label.setText(self._dataset_chip(self.dataset))
            self.path_label.setToolTip(str(self.dataset.root))
        self.label.refresh()
        self.model.refresh()

    def go_to_board(self, model_path: Path) -> None:
        self.board_page.set_model_file(model_path)
        self.tabs.setCurrentWidget(self.board_page)

    # ---------- menu ----------

    def _open_dataset(self) -> None:
        initial = str(self.config.get("dataset_root"))
        path = QFileDialog.getExistingDirectory(self, "Open dataset", initial)
        if not path:
            return
        try:
            ds = Dataset(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Not a dataset", str(exc))
            return
        self.set_dataset(ds)

    def _settings(self) -> None:
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            self.capture.refresh()
            self.model.refresh()
            self.statusBar().showMessage("settings saved")

    def _load_last(self) -> None:
        last = str(self.config.get("last_dataset") or "")
        if not last:
            return
        try:
            ds = Dataset(last)
        except Exception:  # noqa: BLE001
            return
        self.dataset = ds
        self.path_label.setText(self._dataset_chip(ds))
        self.path_label.setToolTip(str(ds.root))
        self.capture._update_dataset_ui()
        self.label.refresh()
        self.model.refresh()

    def closeEvent(self, event) -> None:
        self.capture.shutdown()
        self.model.shutdown()
        self.board_page.shutdown()
        super().closeEvent(event)
