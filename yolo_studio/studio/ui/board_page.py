"""Страница платы: проверка, заливка модели, запуск потока."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMessageBox, QPushButton, QSpinBox, QVBoxLayout,
    QWidget,
)

from ..board import Board, BoardError
from .widgets import LogPane, Worker


class BoardPage(QWidget):
    def __init__(self, main) -> None:
        super().__init__()
        self.main = main
        self.cfg = main.config
        self._worker: Worker | None = None
        self._build_ui()

    # ---------- UI ----------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        left = QVBoxLayout()

        conn = QGroupBox("Плата")
        form = QFormLayout(conn)
        self.ip = QLineEdit(str(self.cfg.get("board_ip")))
        form.addRow("IP:", self.ip)
        self.user = QLineEdit(str(self.cfg.get("board_user")))
        form.addRow("Пользователь:", self.user)
        self.password = QLineEdit(str(self.cfg.get("board_password")))
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Пароль:", self.password)
        self.stream_port = QSpinBox()
        self.stream_port.setRange(1, 65535)
        self.stream_port.setValue(int(self.cfg.get("stream_port", 7777)))
        form.addRow("Порт потока:", self.stream_port)
        for w in (self.ip, self.user, self.password):
            w.editingFinished.connect(self._save_fields)
        self.stream_port.valueChanged.connect(self._save_fields)
        row = QHBoxLayout()
        self.btn_check = QPushButton("Проверить связь")
        self.btn_check.clicked.connect(self._check)
        row.addWidget(self.btn_check)
        btn_refresh = QPushButton("Обновить модели")
        btn_refresh.clicked.connect(self._refresh_models)
        row.addWidget(btn_refresh)
        form.addRow(row)
        self.status = QLabel("нажмите «Проверить связь»")
        self.status.setWordWrap(True)
        form.addRow(self.status)
        left.addWidget(conn)

        models = QGroupBox("Модели на плате")
        ml = QVBoxLayout(models)
        self.models = QListWidget()
        self.models.setMinimumHeight(140)
        ml.addWidget(self.models)
        hint = QLabel("start_stream.sh сканирует /root/models — залитая модель "
                      "сама появится в меню.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #6b7280;")
        ml.addWidget(hint)
        row2 = QHBoxLayout()
        self.btn_run = QPushButton("Запустить выбранную")
        self.btn_run.clicked.connect(self._run_selected)
        row2.addWidget(self.btn_run)
        self.btn_auto = QPushButton("Вернуть auto-поток")
        self.btn_auto.clicked.connect(self._run_auto)
        row2.addWidget(self.btn_auto)
        ml.addLayout(row2)
        left.addWidget(models)

        upload = QGroupBox("Залить модель")
        ul = QVBoxLayout(upload)
        file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("путь к .cvimodel")
        file_row.addWidget(self.file_edit, 1)
        btn_file = QPushButton("…")
        btn_file.setFixedWidth(36)
        btn_file.clicked.connect(self._pick_file)
        file_row.addWidget(btn_file)
        ul.addLayout(file_row)
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("имя на плате:"))
        self.remote_name = QLineEdit()
        name_row.addWidget(self.remote_name, 1)
        ul.addLayout(name_row)
        self.overwrite = QCheckBox("перезаписать, если уже есть")
        ul.addWidget(self.overwrite)
        self.btn_upload = QPushButton("Залить на плату")
        self.btn_upload.clicked.connect(self._upload)
        ul.addWidget(self.btn_upload)
        left.addWidget(upload)
        left.addStretch(1)

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(380)
        root.addWidget(left_w)

        right = QVBoxLayout()
        head = QHBoxLayout()
        head.addWidget(QLabel("Лог"))
        head.addStretch(1)
        btn_clear = QPushButton("Очистить")
        btn_clear.clicked.connect(lambda: self.log.clear())
        head.addWidget(btn_clear)
        right.addLayout(head)
        self.log = LogPane()
        right.addWidget(self.log, 1)
        root.addLayout(right, 1)

    # ---------- поля ----------

    def _save_fields(self) -> None:
        self.cfg["board_ip"] = self.ip.text().strip()
        self.cfg["board_user"] = self.user.text().strip() or "root"
        self.cfg["board_password"] = self.password.text()
        self.cfg["stream_port"] = int(self.stream_port.value())
        self.cfg.save()

    def _board(self) -> Board:
        self._save_fields()
        return Board(
            self.ip.text().strip(), self.user.text().strip() or "root",
            self.password.text(), stream_port=int(self.stream_port.value()))

    def set_model_file(self, path: str | Path) -> None:
        self.file_edit.setText(str(path))
        self.remote_name.setText(Path(path).name)

    # ---------- действия ----------

    def _check(self) -> None:
        board = self._board()

        def task(cancel=None, on_line=None):
            info = board.check()
            info["http"] = board.wait_http(board.stream_port, timeout=3)
            info["capture"] = board.wait_http(
                int(self.cfg.get("capture_port", 7778)), timeout=2)
            return info

        self._run(task, self._check_done, "Проверка не удалась")

    def _check_done(self, info: dict) -> None:
        yolo = info.get("stream_yolo") or "не запущен"
        http = "ок" if info.get("http") else "нет ответа"
        cap = "ок" if info.get("capture") else "нет"
        text = (f"SSH: ок · {info.get('uname', '')}\n"
                f"stream_yolo: {yolo}\n"
                f"поток :{self.stream_port.value()}: {http} · "
                f"съёмочный :{self.cfg.get('capture_port')}: {cap}\n"
                f"{info.get('uptime', '')}")
        self.status.setText(text)
        self.log.append_line(text)

    def _refresh_models(self) -> None:
        board = self._board()

        def task(cancel=None, on_line=None):
            return board.list_models()

        self._run(task, self._models_done, "Не удалось получить список")

    def _models_done(self, names: list[str]) -> None:
        current = self.models.currentItem()
        keep = current.text() if current else None
        self.models.clear()
        for n in names:
            self.models.addItem(n)
            if n == keep:
                self.models.setCurrentRow(self.models.count() - 1)
        self.status.setText(f"моделей на плате: {len(names)}")
        self.log.append_line("модели: " + ", ".join(names))

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите cvimodel", "", "cvimodel (*.cvimodel)")
        if path:
            self.file_edit.setText(path)
            self.remote_name.setText(Path(path).name)

    def _upload(self) -> None:
        local = self.file_edit.text().strip()
        if not local or not Path(local).is_file():
            QMessageBox.information(self, "Нет файла", "Выберите .cvimodel.")
            return
        board = self._board()
        name = self.remote_name.text().strip() or Path(local).name
        overwrite = self.overwrite.isChecked()

        def task(cancel=None, on_line=None):
            remote = board.upload_model(local, name, overwrite=overwrite)
            return remote, board.list_models()

        self._run(task, self._upload_done, "Заливка не удалась")

    def _upload_done(self, result) -> None:
        remote, names = result
        self.log.append_line(f"залито: {remote}")
        self.models.clear()
        for n in names:
            self.models.addItem(n)
        target = Path(remote).name
        for i in range(self.models.count()):
            if self.models.item(i).text() == target:
                self.models.setCurrentRow(i)
                break
        self.status.setText(f"залито: {remote}. Модель появится в конце меню "
                            "start_stream.sh.")

    def _run_selected(self) -> None:
        item = self.models.currentItem()
        if item is None:
            QMessageBox.information(self, "Модель не выбрана",
                                    "Выберите модель из списка.")
            return
        name = item.text()
        if QMessageBox.question(
                self, "Запустить модель",
                f"Поток на плате будет перезапущен с моделью\n{name}.\n"
                "Текущая съёмка будет остановлена. Продолжить?") \
                != QMessageBox.StandardButton.Yes:
            return
        board = self._board()
        stem = name[:-len(".cvimodel")] if name.endswith(".cvimodel") else name

        def task(cancel=None, on_line=None):
            out = board.restart_with_model(stem)
            ok = board.wait_http(board.stream_port, timeout=40)
            return out, ok, board.tail("/root/log/stream.log", 8)

        self._run(task, self._run_done, "Не удалось запустить модель")

    def _run_done(self, result) -> None:
        out, ok, tail = result
        self.log.append_line(out)
        if tail:
            self.log.append_line("--- stream.log ---\n" + tail)
        if ok:
            self.status.setText(f"поток запущен: {self._board().stream_url()}")
        else:
            self.status.setText("модель запущена, но поток не ответил — смотрите лог")

    def _run_auto(self) -> None:
        board = self._board()

        def task(cancel=None, on_line=None):
            board.stop_all()
            board.start_normal()
            ok = board.wait_http(board.stream_port, timeout=40)
            return ok, board.tail("/root/log/stream.log", 8)

        self._run(task, self._auto_done, "Не удалось вернуть поток")

    def _auto_done(self, result) -> None:
        ok, tail = result
        if tail:
            self.log.append_line("--- stream.log ---\n" + tail)
        self.status.setText("обычный поток запущен" if ok else "поток не ответил")

    # ---------- фон ----------

    def _run(self, fn, on_done, error_title: str) -> None:
        if self._worker is not None:
            return
        worker = Worker(fn)
        self._worker = worker

        def done(res) -> None:
            try:
                on_done(res)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Ошибка", str(exc))
            self._cleanup()

        def failed(msg: str) -> None:
            self.log.append_line("ОШИБКА: " + msg)
            QMessageBox.critical(self, error_title, msg)
            self._cleanup()

        worker.done.connect(done)
        worker.failed.connect(failed)
        worker.line.connect(self.log.append_line)
        worker.start()

    def _cleanup(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    def showEvent(self, event) -> None:
        self.ip.setText(str(self.cfg.get("board_ip")))
        self.user.setText(str(self.cfg.get("board_user")))
        self.password.setText(str(self.cfg.get("board_password")))
        super().showEvent(event)

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.wait(3000)
