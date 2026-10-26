"""Страница съёмки: подключение камеры, сценарий ракурсов, сохранение кадров."""

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
from ..dataset import Dataset
from .widgets import FrameView, Worker

STEPS: list[tuple[str, str, float]] = [
    ("Спереди", "Держите объект по центру кадра, покажите переднюю сторону", 3.0),
    ("Слева", "Поверните объект левой стороной к камере", 3.0),
    ("Справа", "Поверните объект правой стороной к камере", 3.0),
    ("Сверху", "Наклоните объект и покажите верхнюю сторону", 2.5),
    ("Снизу", "Покажите нижнюю сторону объекта", 2.5),
    ("Близко", "Поднесите объект ближе — примерно на полкадра", 2.0),
    ("Дальше", "Отодвиньте объект — средний план", 2.0),
    ("На фоне", "Покажите объект на фоне окружения", 3.0),
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
        top = QHBoxLayout()

        # --- датасет
        ds_box = QGroupBox("Датасет")
        ds_lay = QVBoxLayout(ds_box)
        self.ds_path = QLabel("не выбран")
        self.ds_path.setWordWrap(True)
        self.ds_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        ds_lay.addWidget(self.ds_path)
        row = QHBoxLayout()
        btn_new = QPushButton("Новый…")
        btn_new.clicked.connect(self._new_dataset)
        btn_open = QPushButton("Открыть…")
        btn_open.clicked.connect(self._open_dataset)
        row.addWidget(btn_new)
        row.addWidget(btn_open)
        ds_lay.addLayout(row)
        cls_row = QHBoxLayout()
        cls_row.addWidget(QLabel("Классы:"))
        self.cls_edit = QLineEdit()
        self.cls_edit.setPlaceholderText("например, drone или drone, person")
        self.cls_edit.setToolTip(
            "Имена классов через запятую. Порядок = id: первый — 0, второй — 1.")
        cls_row.addWidget(self.cls_edit, 1)
        btn_cls = QPushButton("OK")
        btn_cls.setFixedWidth(44)
        btn_cls.clicked.connect(self._save_class)
        cls_row.addWidget(btn_cls)
        ds_lay.addLayout(cls_row)
        self.ds_stats = QLabel("")
        ds_lay.addWidget(self.ds_stats)

        # --- камера
        cam_box = QGroupBox("Камера")
        cam_lay = QFormLayout(cam_box)
        self.src_combo = QComboBox()
        self.src_combo.addItem("Плата (чистый поток)", "board")
        self.src_combo.addItem("USB-камера", "usb")
        idx = self.src_combo.findData(self.cfg.get("camera_source"))
        self.src_combo.setCurrentIndex(max(0, idx))
        cam_lay.addRow("Источник:", self.src_combo)

        usb_row = QWidget()
        usb_lay = QHBoxLayout(usb_row)
        usb_lay.setContentsMargins(0, 0, 0, 0)
        usb_lay.addWidget(QLabel("/dev/video"))
        self.usb_index = QSpinBox()
        self.usb_index.setRange(0, 16)
        self.usb_index.setValue(int(self.cfg.get("usb_index", 0)))
        usb_lay.addWidget(self.usb_index)
        usb_lay.addStretch(1)
        cam_lay.addRow("", usb_row)

        board_row = QHBoxLayout()
        self.btn_prepare = QPushButton("Чистый поток")
        self.btn_prepare.setToolTip(
            "Остановить yolo/face на плате и поднять поток без детекции и рамок")
        self.btn_prepare.clicked.connect(self._prepare_board)
        self.btn_normal = QPushButton("Вернуть yolo")
        self.btn_normal.setToolTip("Остановить съёмку и вернуть обычный поток с рамками")
        self.btn_normal.clicked.connect(self._restore_board)
        board_row.addWidget(self.btn_prepare)
        board_row.addWidget(self.btn_normal)
        cam_lay.addRow(board_row)

        conn_row = QHBoxLayout()
        self.btn_connect = QPushButton("Подключить камеру")
        self.btn_connect.clicked.connect(self._connect_camera)
        self.btn_disconnect = QPushButton("Отключить")
        self.btn_disconnect.clicked.connect(self._disconnect_camera)
        conn_row.addWidget(self.btn_connect)
        conn_row.addWidget(self.btn_disconnect)
        cam_lay.addRow(conn_row)
        self.cam_status = QLabel("камера не подключена")
        self.cam_status.setWordWrap(True)
        cam_lay.addRow(self.cam_status)

        top.addWidget(ds_box, 1)
        top.addWidget(cam_box, 1)
        root.addLayout(top)

        # --- видео + сценарий
        mid = QHBoxLayout()
        self.view = FrameView()
        self.view.set_editable(False)
        mid.addWidget(self.view, 3)

        scen = QGroupBox("Сценарий съёмки")
        scen_lay = QVBoxLayout(scen)
        self.steps_list = QListWidget()
        self.steps_list.setMinimumHeight(190)
        for i, (name, _prompt, sec) in enumerate(STEPS):
            item = QListWidgetItem(f"{i + 1}. {name} — {sec:g} с")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.steps_list.addItem(item)
        self.steps_list.setCurrentRow(0)
        scen_lay.addWidget(self.steps_list)

        self.prompt = QLabel("Выберите шаг и нажмите «Начать шаг».\n"
                             "Кадры сохраняются автоматически, пока идёт отсчёт.")
        self.prompt.setWordWrap(True)
        self.prompt.setStyleSheet("font-weight: 600;")
        scen_lay.addWidget(self.prompt)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        scen_lay.addWidget(self.progress)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Начать шаг")
        self.btn_start.clicked.connect(self._start_step)
        self.btn_skip = QPushButton("Пропустить")
        self.btn_skip.clicked.connect(self._skip_step)
        self.btn_snap = QPushButton("Снимок")
        self.btn_snap.clicked.connect(self._snapshot)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_skip)
        btn_row.addWidget(self.btn_snap)
        scen_lay.addLayout(btn_row)

        self.capture_stats = QLabel("сохранено за сессию: 0 · пропущено: 0")
        self.capture_stats.setWordWrap(True)
        scen_lay.addWidget(self.capture_stats)
        scen_lay.addStretch(1)

        mid.addWidget(scen, 2)
        root.addLayout(mid, 1)

    # ---------- датасет ----------

    @property
    def dataset(self) -> Dataset | None:
        return self.main.dataset

    def _new_dataset(self) -> None:
        name, ok = QInputDialog.getText(self, "Новый датасет", "Имя папки датасета:")
        if not ok or not name.strip():
            return
        cls, ok = QInputDialog.getText(
            self, "Классы объекта",
            "Имена классов через запятую (латиницей удобнее):",
            text="object")
        if not ok:
            return
        class_names = [part.strip() for part in re.split(r"[,;]", cls)
                       if part.strip()] or ["object"]
        root = Path(str(self.cfg.get("dataset_root"))).expanduser() / name.strip()
        if root.exists() and any(root.iterdir()):
            if QMessageBox.question(self, "Датасет существует",
                                    f"{root} уже не пуст. Открыть его?") \
                    != QMessageBox.StandardButton.Yes:
                return
            try:
                ds = Dataset(root)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Ошибка", str(exc))
                return
        else:
            try:
                ds = Dataset(root, create=True, class_names=class_names)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Ошибка", str(exc))
                return
        self.main.set_dataset(ds)
        self._update_dataset_ui()

    def _open_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Открыть датасет", str(self.cfg.get("dataset_root")))
        if not path:
            return
        try:
            ds = Dataset(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Это не датасет", str(exc))
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

    def _update_dataset_ui(self) -> None:
        ds = self.dataset
        if ds is None:
            self.ds_path.setText("не выбран — создайте или откройте датасет")
            self.ds_stats.setText("")
            self.cls_edit.clear()
        else:
            self.ds_path.setText(str(ds.root))
            self.cls_edit.setText(", ".join(ds.class_names))
            st = ds.stats()
            self.ds_stats.setText(
                f"кадров: {st['images']} · размечено: {st['labeled']} · "
                f"рамок: {st['boxes']}")
        self._update_buttons()

    def _update_buttons(self) -> None:
        has_ds = self.dataset is not None
        self.btn_start.setEnabled(has_ds and self._camera is not None
                                  and self._active_step is None)
        self.btn_snap.setEnabled(has_ds and self._camera is not None)
        self.btn_prepare.setEnabled(self._worker is None and self._camera is None)
        self.btn_connect.setEnabled(self._worker is None)

    # ---------- плата ----------

    def _board(self) -> Board:
        return self.main.board()

    def _prepare_board(self) -> None:
        if QMessageBox.question(
                self, "Режим съёмки",
                "На плате будет остановлен текущий поток (yolo или face) "
                "и запущен чистый поток без рамок. Продолжить?") \
                != QMessageBox.StandardButton.Yes:
            return
        board = self._board()
        port = int(self.cfg.get("capture_port", 7778))
        self.cam_status.setText("готовлю плату: останавливаю поток…")

        def task(cancel=None, on_line=None):
            if on_line:
                on_line("остановка потока…")
            board.stop_all()
            if on_line:
                on_line(f"запуск чистого потока на :{port}…")
            board.start_clean(port)
            return board.stream_url(port)

        self._run(task, self._board_prepared, "Не удалось подготовить плату")

    def _restore_board(self) -> None:
        if QMessageBox.question(
                self, "Вернуть yolo",
                "Остановить камеру и вернуть обычный поток yolo на плате?") \
                != QMessageBox.StandardButton.Yes:
            return
        self._stop_camera()
        board = self._board()
        self.cam_status.setText("возвращаю обычный поток…")

        def task(cancel=None, on_line=None):
            board.stop_all()
            board.start_normal()
            return "ok"

        def done(_res):
            self._clean_started = False
            self.cam_status.setText(
                f"обычный поток: {board.stream_url(int(self.cfg.get('stream_port', 7777)))}")
            self._update_buttons()

        self._run(task, done, "Не удалось вернуть поток")

    def _board_prepared(self, url: str) -> None:
        self._clean_started = True
        self.cam_status.setText(f"чистый поток готов: {url}")
        # воркер ещё жив в момент этого колбэка — подключаемся следующим тиком
        QTimer.singleShot(0, self._connect_camera)

    # ---------- камера ----------

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
        self.cam_status.setText("подключаюсь…")

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
        self.cam_status.setText("камера отключена")

    def _on_frame(self, frame: np.ndarray) -> None:
        self._last_frame = frame
        now = time.monotonic()
        if now - self._last_show >= 0.05:
            self._last_show = now
            self.view.set_frame(frame)

    # ---------- сценарий ----------

    def _start_step(self) -> None:
        ds = self.dataset
        if ds is None:
            QMessageBox.information(self, "Нет датасета", "Создайте или откройте датасет.")
            return
        if self._last_frame is None:
            QMessageBox.information(self, "Нет кадров",
                                    "Сначала подключите камеру.")
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
        self.prompt.setText(f"{name}: {prompt} — {sec:g} с")
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
            f"сохранено за сессию: {self._saved_session} · "
            f"пропущено: {self._skipped_session} ({reason})")

    def _finish_step(self) -> None:
        self._timer.stop()
        idx = self._active_step
        self._active_step = None
        if idx is not None:
            item = self.steps_list.item(idx)
            item.setText(f"✓ {idx + 1}. {STEPS[idx][0]} — {STEPS[idx][2]:g} с")
            item.setForeground(QColor("#31d17c"))
            self._next_unfinished(select=True)
        self.progress.setValue(100)
        nxt = self.steps_list.currentRow()
        if nxt >= 0 and nxt != idx:
            self.prompt.setText(
                f"Готово. Следующий шаг: «{STEPS[nxt][0]}» — "
                "подготовьте объект и нажмите «Начать шаг».")
        else:
            self.prompt.setText("Сценарий пройден. Можно повторить шаг или снять ещё кадры.")
        self.btn_start.setEnabled(True)
        self.btn_skip.setEnabled(True)
        self._update_dataset_ui()

    def _skip_step(self) -> None:
        self._timer.stop()
        idx = self._active_step
        self._active_step = None
        if idx is not None:
            item = self.steps_list.item(idx)
            item.setText(f"— {idx + 1}. {STEPS[idx][0]} (пропущен)")
            item.setForeground(QColor("#9aa4b2"))
            self._next_unfinished(select=True)
        self.progress.setValue(0)
        self.prompt.setText("Шаг пропущен. Выберите следующий и нажмите «Начать шаг».")
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
                f"сохранено за сессию: {self._saved_session} · "
                f"пропущено: {self._skipped_session}")
            self._update_dataset_ui()
        else:
            self.cam_status.setText(f"снимок не сохранён: {reason}")

    # ---------- фоновые задачи ----------

    def _run(self, fn, on_done, error_title: str) -> None:
        if self._worker is not None:
            return
        self.btn_prepare.setEnabled(False)
        self.btn_connect.setEnabled(False)
        worker = Worker(fn)
        self._worker = worker

        def cleanup() -> None:
            self._worker = None
            self._update_buttons()

        worker.done.connect(lambda res: (on_done(res), cleanup()))
        worker.failed.connect(lambda msg: (QMessageBox.critical(self, error_title, msg),
                                           self.cam_status.setText(msg), cleanup()))
        worker.start()

    # ---------- прочее ----------

    def refresh(self) -> None:
        self._update_dataset_ui()

    def showEvent(self, event) -> None:
        self._update_dataset_ui()
        super().showEvent(event)

    def shutdown(self) -> None:
        """Останавливает камеру и фоновые задачи перед закрытием окна."""
        self._timer.stop()
        self._stop_camera()
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(3000)
