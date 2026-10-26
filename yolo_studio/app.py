#!/usr/bin/env python3
"""Lichee YOLO Studio — запуск."""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from PySide6.QtWidgets import QApplication  # noqa: E402

from studio.config import Config  # noqa: E402
from studio.ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Lichee YOLO Studio")
    app.setOrganizationName("lichee")
    app.setStyle("Fusion")
    window = MainWindow(Config())
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
