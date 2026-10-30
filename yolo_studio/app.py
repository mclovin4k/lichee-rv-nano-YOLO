#!/usr/bin/env python3
"""Lichee YOLO Studio."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _check_linux_qt() -> None:
    """Warn about the system Qt/X11 dependency before QApplication aborts."""
    if sys.platform != "linux":
        return
    if os.environ.get("YOLO_STUDIO_SKIP_QT_CHECK") == "1":
        return
    platform = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
    if platform.startswith(("offscreen", "minimal", "wayland")):
        return
    try:
        import ctypes.util
        if ctypes.util.find_library("xcb-cursor") is not None:
            return
    except Exception:  # noqa: BLE001 - the check must never block startup
        return
    print(
        "warning: libxcb-cursor was not found.\n"
        "Qt 6.5+ needs it for the X11 (xcb) platform plugin. Install it with:\n"
        "  Debian/Ubuntu: sudo apt install libxcb-cursor0\n"
        "  Fedora:        sudo dnf install xcb-util-cursor\n"
        "  Arch:          sudo pacman -S xcb-util-cursor\n"
        "On Wayland you can try: QT_QPA_PLATFORM=wayland python app.py\n"
        "Set YOLO_STUDIO_SKIP_QT_CHECK=1 to skip this check.",
        file=sys.stderr,
    )


def main() -> int:
    _check_linux_qt()
    smoke_test = "--smoke-test" in sys.argv
    if smoke_test:
        sys.argv.remove("--smoke-test")

    from PySide6.QtCore import QTimer  # noqa: E402
    from PySide6.QtWidgets import QApplication  # noqa: E402

    from studio.config import Config  # noqa: E402
    from studio.ui.main_window import MainWindow  # noqa: E402

    app = QApplication(sys.argv)
    app.setApplicationName("Lichee YOLO Studio")
    app.setOrganizationName("lichee")
    app.setStyle("Fusion")
    window = MainWindow(Config())
    window.show()
    if smoke_test:
        QTimer.singleShot(750, app.quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
