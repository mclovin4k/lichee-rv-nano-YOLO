"""Board page: connection check, model upload, stream start."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMessageBox, QPushButton, QSpinBox, QVBoxLayout,
    QWidget,
)

from ..board import Board, BoardError
from .theme import accent, primary
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
        root.setSpacing(12)
        left = QVBoxLayout()
        left.setSpacing(10)

        conn = QGroupBox("Board")
        form = QFormLayout(conn)
        form.setSpacing(8)
        self.ip = QLineEdit(str(self.cfg.get("board_ip")))
        form.addRow("IP:", self.ip)
        self.user = QLineEdit(str(self.cfg.get("board_user")))
        form.addRow("User:", self.user)
        self.password = QLineEdit(str(self.cfg.get("board_password")))
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Password:", self.password)
        self.stream_port = QSpinBox()
        self.stream_port.setRange(1, 65535)
        self.stream_port.setValue(int(self.cfg.get("stream_port", 7777)))
        form.addRow("Stream port:", self.stream_port)
        for w in (self.ip, self.user, self.password):
            w.editingFinished.connect(self._save_fields)
        self.stream_port.valueChanged.connect(self._save_fields)
        row = QHBoxLayout()
        self.btn_check = QPushButton("Check connection")
        self.btn_check.clicked.connect(self._check)
        accent(self.btn_check)
        row.addWidget(self.btn_check)
        btn_refresh = QPushButton("Refresh models")
        btn_refresh.clicked.connect(self._refresh_models)
        row.addWidget(btn_refresh)
        form.addRow(row)
        self.status = QLabel("click “Check connection”")
        self.status.setObjectName("Muted")
        self.status.setWordWrap(True)
        self.status.setAlignment(Qt.AlignmentFlag.AlignTop
                                 | Qt.AlignmentFlag.AlignLeft)
        form.addRow(self.status)
        left.addWidget(conn)

        models = QGroupBox("Models on board")
        ml = QVBoxLayout(models)
        ml.setSpacing(8)
        self.models = QListWidget()
        self.models.setMinimumHeight(180)
        self.models.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.models.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ml.addWidget(self.models)
        hint = QLabel("start_stream.sh scans /root/models — an uploaded model "
                      "appears in the menu automatically.")
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        ml.addWidget(hint)
        row2 = QHBoxLayout()
        self.btn_run = QPushButton("Start selected")
        self.btn_run.clicked.connect(self._run_selected)
        row2.addWidget(self.btn_run)
        self.btn_auto = QPushButton("Restore auto stream")
        self.btn_auto.clicked.connect(self._run_auto)
        row2.addWidget(self.btn_auto)
        ml.addLayout(row2)
        left.addWidget(models)

        left.addStretch(1)

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(460)
        root.addWidget(left_w)

        right = QVBoxLayout()
        right.setSpacing(10)

        upload = QGroupBox("Upload model")
        uf = QFormLayout(upload)
        uf.setSpacing(8)
        file_row = QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("path to .cvimodel")
        file_row.addWidget(self.file_edit, 1)
        btn_file = QPushButton("…")
        btn_file.setFixedWidth(40)
        btn_file.clicked.connect(self._pick_file)
        file_row.addWidget(btn_file)
        uf.addRow("File:", file_row)
        self.remote_name = QLineEdit()
        uf.addRow("Name on board:", self.remote_name)
        action_row = QHBoxLayout()
        self.overwrite = QCheckBox("overwrite if it already exists")
        action_row.addWidget(self.overwrite)
        action_row.addStretch(1)
        self.btn_upload = QPushButton("Upload to board")
        self.btn_upload.clicked.connect(self._upload)
        primary(self.btn_upload)
        action_row.addWidget(self.btn_upload)
        uf.addRow(action_row)
        right.addWidget(upload)

        head = QHBoxLayout()
        head.addWidget(QLabel("Log"))
        head.addStretch(1)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(lambda: self.log.clear())
        head.addWidget(btn_clear)
        right.addLayout(head)
        self.log = LogPane()
        right.addWidget(self.log, 1)
        root.addLayout(right, 1)

    # ---------- fields ----------

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

    # ---------- actions ----------

    def _set_status(self, text: str) -> None:
        self.status.setText(text)
        lines = text.count("\n") + 1
        self.status.setMinimumHeight(
            self.status.fontMetrics().lineSpacing() * lines + 10)

    def _check(self) -> None:
        board = self._board()

        def task(cancel=None, on_line=None):
            info = board.check()
            info["http"] = board.wait_http(board.stream_port, timeout=3)
            info["capture"] = board.wait_http(
                int(self.cfg.get("capture_port", 7778)), timeout=2)
            return info

        self._run(task, self._check_done, "Connection check failed")

    def _check_done(self, info: dict) -> None:
        yolo = info.get("stream_yolo") or "not running"
        http = "ok" if info.get("http") else "no response"
        cap = "ok" if info.get("capture") else "down"
        text = (f"SSH: ok · {info.get('uname', '')}\n"
                f"stream_yolo: {yolo}\n"
                f"stream :{self.stream_port.value()}: {http} · "
                f"capture :{self.cfg.get('capture_port')}: {cap}\n"
                f"{info.get('uptime', '')}")
        self._set_status(text)
        self.log.append_line(text)

    def _refresh_models(self) -> None:
        board = self._board()

        def task(cancel=None, on_line=None):
            return board.list_models()

        self._run(task, self._models_done, "Failed to get the model list")

    def _models_done(self, names: list[str]) -> None:
        current = self.models.currentItem()
        keep = current.text() if current else None
        self.models.clear()
        for n in names:
            self.models.addItem(n)
            self.models.item(self.models.count() - 1).setToolTip(n)
            if n == keep:
                self.models.setCurrentRow(self.models.count() - 1)
        self._set_status(f"models on board: {len(names)}")
        self.log.append_line("models: " + ", ".join(names))

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select cvimodel", "", "cvimodel (*.cvimodel)")
        if path:
            self.file_edit.setText(path)
            self.remote_name.setText(Path(path).name)

    def _upload(self) -> None:
        local = self.file_edit.text().strip()
        if not local or not Path(local).is_file():
            QMessageBox.information(self, "No file", "Select a .cvimodel file.")
            return
        board = self._board()
        name = self.remote_name.text().strip() or Path(local).name
        overwrite = self.overwrite.isChecked()

        def task(cancel=None, on_line=None):
            remote = board.upload_model(local, name, overwrite=overwrite)
            return remote, board.list_models()

        self._run(task, self._upload_done, "Upload failed")

    def _upload_done(self, result) -> None:
        remote, names = result
        self.log.append_line(f"uploaded: {remote}")
        self.models.clear()
        for n in names:
            self.models.addItem(n)
            self.models.item(self.models.count() - 1).setToolTip(n)
        target = Path(remote).name
        for i in range(self.models.count()):
            if self.models.item(i).text() == target:
                self.models.setCurrentRow(i)
                break
        self._set_status(f"uploaded: {remote}. The model will appear at the end of "
                         "the start_stream.sh menu.")

    def _run_selected(self) -> None:
        item = self.models.currentItem()
        if item is None:
            QMessageBox.information(self, "No model selected",
                                    "Select a model from the list.")
            return
        name = item.text()
        if QMessageBox.question(
                self, "Start model",
                f"The board stream will be restarted with model\n{name}.\n"
                "Current capture will be stopped. Continue?") \
                != QMessageBox.StandardButton.Yes:
            return
        board = self._board()
        stem = name[:-len(".cvimodel")] if name.endswith(".cvimodel") else name

        def task(cancel=None, on_line=None):
            out = board.restart_with_model(stem)
            ok = board.wait_http(board.stream_port, timeout=40)
            return out, ok, board.tail("/root/log/stream.log", 8)

        self._run(task, self._run_done, "Failed to start model")

    def _run_done(self, result) -> None:
        out, ok, tail = result
        self.log.append_line(out)
        if tail:
            self.log.append_line("--- stream.log ---\n" + tail)
        if ok:
            self._set_status(f"stream started: {self._board().stream_url()}")
        else:
            self._set_status("model started, but the stream did not respond — check the log")

    def _run_auto(self) -> None:
        board = self._board()

        def task(cancel=None, on_line=None):
            board.stop_all()
            board.start_normal()
            ok = board.wait_http(board.stream_port, timeout=40)
            return ok, board.tail("/root/log/stream.log", 8)

        self._run(task, self._auto_done, "Failed to restore the stream")

    def _auto_done(self, result) -> None:
        ok, tail = result
        if tail:
            self.log.append_line("--- stream.log ---\n" + tail)
        self._set_status("normal stream started" if ok else "stream did not respond")

    # ---------- background ----------

    def _run(self, fn, on_done, error_title: str) -> None:
        if self._worker is not None:
            return
        worker = Worker(fn)
        self._worker = worker

        def done(res) -> None:
            try:
                on_done(res)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Error", str(exc))
            self._cleanup()

        def failed(msg: str) -> None:
            self.log.append_line("ERROR: " + msg)
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
