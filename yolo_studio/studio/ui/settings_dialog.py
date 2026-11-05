"""Settings dialog."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QWidget,
)

from .. import pyenv
from ..config import Config


class SettingsDialog(QDialog):
    def __init__(self, cfg: Config, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("Settings")
        self.setMinimumWidth(640)
        form = QFormLayout(self)
        form.setSpacing(10)
        form.setContentsMargins(18, 18, 18, 18)

        self.ip = QLineEdit(str(cfg.get("board_ip")))
        form.addRow("Board IP:", self.ip)
        self.user = QLineEdit(str(cfg.get("board_user")))
        form.addRow("User:", self.user)
        self.password = QLineEdit(str(cfg.get("board_password")))
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Password:", self.password)
        self.stream_port = QSpinBox()
        self.stream_port.setRange(1, 65535)
        self.stream_port.setValue(int(cfg.get("stream_port", 7777)))
        form.addRow("Normal stream port:", self.stream_port)
        self.capture_port = QSpinBox()
        self.capture_port.setRange(1, 65535)
        self.capture_port.setValue(int(cfg.get("capture_port", 7778)))
        form.addRow("Capture stream port:", self.capture_port)

        self.ultra = self._path_row(form, "Python Ultralytics:", "ultra_python",
                                    file_mode=True, detect_kind="ultra")
        self.tpu = self._path_row(form, "Python tpu-mlir:", "tpu_python",
                                  file_mode=True, detect_kind="tpu")
        self.datasets = self._path_row(form, "Datasets directory:", "dataset_root",
                                       file_mode=False)

        note = QWidget()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(note)
        form.addRow(buttons)

    def _path_row(self, form: QFormLayout, title: str, key: str,
                  file_mode: bool, detect_kind: str = "") -> QLineEdit:
        row = QHBoxLayout()
        edit = QLineEdit(str(self.cfg.get(key)))
        wsl_tpu = detect_kind == "tpu" and pyenv.is_win()
        if detect_kind and pyenv.bundled(detect_kind):
            edit.setPlaceholderText("bundled in this build")
            edit.setToolTip("This build already contains the package; the "
                            "field can stay empty.")
        elif wsl_tpu:
            edit.setPlaceholderText(
                "/home/<user>/.venvs/tpu310/bin/python (WSL)")
            edit.setToolTip("tpu-mlir runs inside WSL 2. Enter the Linux path "
                            "to its interpreter, or click Detect.")
        elif detect_kind:
            edit.setPlaceholderText("detected automatically")
        row.addWidget(edit, 1)
        if not wsl_tpu:
            btn = QPushButton("…")
            btn.setFixedWidth(40)
            row.addWidget(btn)
        if detect_kind:
            detect = QPushButton("Detect")
            detect.setToolTip(
                "Search inside the default WSL distribution for an "
                "interpreter that has tpu-mlir" if wsl_tpu else
                "Search for an interpreter that has the package")
            row.addWidget(detect)

            def run_detect() -> None:
                exe = pyenv.detect(detect_kind, edit.text().strip(),
                                   force=True)
                if exe is None:
                    QMessageBox.information(self, "Not found",
                                            pyenv.missing_help(detect_kind))
                    return
                edit.setText(str(exe))

            detect.clicked.connect(run_detect)
        form.addRow(title, row)
        if wsl_tpu:
            hint = QLabel("Used inside WSL 2. The Windows file picker cannot "
                          "browse Linux paths — click Detect or type the path.")
            hint.setWordWrap(True)
            form.addRow("", hint)

        if not wsl_tpu:
            def browse() -> None:
                if file_mode:
                    path, _ = QFileDialog.getOpenFileName(self, title,
                                                          edit.text())
                else:
                    path = QFileDialog.getExistingDirectory(self, title,
                                                            edit.text())
                if path:
                    edit.setText(path)

            btn.clicked.connect(browse)
        return edit

    def _accept(self) -> None:
        self.cfg["board_ip"] = self.ip.text().strip()
        self.cfg["board_user"] = self.user.text().strip() or "root"
        self.cfg["board_password"] = self.password.text()
        self.cfg["stream_port"] = int(self.stream_port.value())
        self.cfg["capture_port"] = int(self.capture_port.value())
        self.cfg["ultra_python"] = self.ultra.text().strip()
        self.cfg["tpu_python"] = self.tpu.text().strip()
        self.cfg["dataset_root"] = self.datasets.text().strip()
        self.cfg.save()
        self.accept()
