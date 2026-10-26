"""Главное окно: вкладки, общий датасет, настройки."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog, QLabel, QMainWindow, QMessageBox, QTabWidget, QToolBar,
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
    word = "класс" if len(ds.class_names) == 1 else "классы"
    return f"{word}: {', '.join(ds.class_names)}"


class MainWindow(QMainWindow):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.dataset: Dataset | None = None
        self.last_cvimodel: Path | None = None
        self.setWindowTitle("Lichee YOLO Studio — датасет и cvimodel для LicheeRV Nano")
        self.resize(1380, 880)
        self._build_ui()
        self._load_last()

    def _build_ui(self) -> None:
        bar = QToolBar("Главная")
        bar.setMovable(False)
        self.addToolBar(bar)

        act_open = QAction("Открыть датасет…", self)
        act_open.triggered.connect(self._open_dataset)
        bar.addAction(act_open)
        act_settings = QAction("Настройки…", self)
        act_settings.triggered.connect(self._settings)
        bar.addAction(act_settings)
        bar.addSeparator()
        self.path_label = QLabel("датасет не выбран")
        bar.addWidget(self.path_label)

        self.tabs = QTabWidget()
        self.capture = CapturePage(self)
        self.label = LabelPage(self)
        self.model = ModelPage(self)
        self.board_page = BoardPage(self)
        self.tabs.addTab(self.capture, "1. Съёмка")
        self.tabs.addTab(self.label, "2. Разметка")
        self.tabs.addTab(self.model, "3. Модель")
        self.tabs.addTab(self.board_page, "4. Плата")
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage("готово")

    # ---------- общий доступ ----------

    def board(self) -> Board:
        return Board(
            str(self.config.get("board_ip")),
            str(self.config.get("board_user")) or "root",
            str(self.config.get("board_password")),
            stream_port=int(self.config.get("stream_port", 7777)))

    def set_dataset(self, ds: Dataset) -> None:
        self.dataset = ds
        self.config["last_dataset"] = str(ds.root)
        self.config.save()
        self.path_label.setText(f"{ds.root}  ·  {classes_text(ds)}")
        self.capture._update_dataset_ui()
        self.label.refresh()
        self.model.refresh()
        self.statusBar().showMessage(f"датасет: {ds.root}")

    def notify_dataset_changed(self) -> None:
        if self.dataset is not None:
            self.path_label.setText(
                f"{self.dataset.root}  ·  {classes_text(self.dataset)}")
        self.label.refresh()
        self.model.refresh()

    def go_to_board(self, model_path: Path) -> None:
        self.board_page.set_model_file(model_path)
        self.tabs.setCurrentWidget(self.board_page)

    # ---------- меню ----------

    def _open_dataset(self) -> None:
        initial = str(self.config.get("dataset_root"))
        path = QFileDialog.getExistingDirectory(self, "Открыть датасет", initial)
        if not path:
            return
        try:
            ds = Dataset(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Это не датасет", str(exc))
            return
        self.set_dataset(ds)

    def _settings(self) -> None:
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            self.capture.refresh()
            self.model.refresh()
            self.statusBar().showMessage("настройки сохранены")

    def _load_last(self) -> None:
        last = str(self.config.get("last_dataset") or "")
        if not last:
            return
        try:
            ds = Dataset(last)
        except Exception:  # noqa: BLE001
            return
        self.dataset = ds
        self.path_label.setText(f"{ds.root}  ·  {classes_text(ds)}")
        self.capture._update_dataset_ui()
        self.label.refresh()
        self.model.refresh()

    def closeEvent(self, event) -> None:
        self.capture.shutdown()
        self.model.shutdown()
        self.board_page.shutdown()
        super().closeEvent(event)
