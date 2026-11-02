#!/usr/bin/env python3
"""Lichee YOLO Studio."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


# Shared libraries Qt links against that are not part of the bundle. They are
# present on every normal desktop install, but a minimal image can lack them
# and Qt then fails with a bare ImportError.
_QT_LIBS = (
    ("libEGL.so.1", "libegl1"),
    ("libGL.so.1", "libgl1"),
    ("libxkbcommon.so.0", "libxkbcommon0"),
    ("libfontconfig.so.1", "libfontconfig1"),
)

_INSTALL_HINT = (
    "  Debian/Ubuntu: sudo apt install libegl1 libgl1 libxkbcommon0 "
    "libfontconfig1 libdbus-1-3 libxcb-cursor0\n"
    "  Fedora:        sudo dnf install mesa-libEGL mesa-libGL libxkbcommon "
    "fontconfig dbus-libs xcb-util-cursor\n"
    "  Arch:          sudo pacman -S mesa libxkbcommon fontconfig dbus "
    "xcb-util-cursor"
)


def _missing_libs(sonames: tuple[str, ...]) -> list[str]:
    """Return the sonames that cannot be loaded on this system."""
    import ctypes

    missing = []
    for soname in sonames:
        try:
            ctypes.CDLL(soname)
        except OSError:
            missing.append(soname)
    return missing


def _have_xcb_cursor() -> bool:
    """True when libxcb-cursor is loadable (system, PySide6 wheel or bundle)."""
    try:
        import ctypes

        ctypes.CDLL("libxcb-cursor.so.0")
        return True
    except OSError:
        pass
    try:
        import ctypes.util

        if ctypes.util.find_library("xcb-cursor") is not None:
            return True
    except Exception:  # noqa: BLE001 - the check must never block startup
        pass
    # PySide6 ships its own copy in the wheel, and PyInstaller unpacks one into
    # sys._MEIPASS, so a bare soname lookup is not enough.
    roots = []
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        roots.append(Path(bundle))
    try:
        import PySide6

        roots.append(Path(PySide6.__file__).resolve().parent)
    except Exception:  # noqa: BLE001 - the check must never block startup
        pass
    for root in roots:
        try:
            if any(root.rglob("libxcb-cursor.so*")):
                return True
        except OSError:
            continue
    return False


def _check_linux_qt() -> None:
    """Warn about missing system Qt/X11 libraries before Qt aborts."""
    if sys.platform != "linux":
        return
    if os.environ.get("YOLO_STUDIO_SKIP_QT_CHECK") == "1":
        return
    try:
        missing = _missing_libs(tuple(name for name, _ in _QT_LIBS))
    except Exception:  # noqa: BLE001 - the check must never block startup
        return
    if missing:
        packages = ", ".join(
            pkg for name, pkg in _QT_LIBS if name in missing
        )
        print(
            f"warning: missing system libraries: {', '.join(missing)}\n"
            f"Qt needs them ({packages}). Install with:\n"
            f"{_INSTALL_HINT}\n"
            "Set YOLO_STUDIO_SKIP_QT_CHECK=1 to skip this check.",
            file=sys.stderr,
        )

    platform = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
    if platform.startswith(("offscreen", "minimal", "wayland")):
        return
    if _have_xcb_cursor():
        return
    print(
        "warning: libxcb-cursor was not found.\n"
        "Qt 6.5+ needs it for the X11 (xcb) platform plugin. Install it with:\n"
        "  Debian/Ubuntu: sudo apt install libxcb-cursor0\n"
        "  Fedora:        sudo dnf install xcb-util-cursor\n"
        "  Arch:          sudo pacman -S xcb-util-cursor\n"
        "On Wayland you can try: QT_QPA_PLATFORM=wayland ./LicheeYOLOStudio\n"
        "Set YOLO_STUDIO_SKIP_QT_CHECK=1 to skip this check.",
        file=sys.stderr,
    )


def _run_helper(argv: list[str]) -> int | None:
    """Hidden modes used by the pipeline of a build that bundles ultralytics.

    A frozen application has no separate interpreter to spawn, so training and
    the ONNX export re-enter this executable:

        LicheeYOLOStudio --yolo-cli detect train data=... model=...
        LicheeYOLOStudio --export-onnx best.pt 640
    """
    if argv[:1] == ["--yolo-cli"]:
        sys.argv = ["yolo"] + argv[1:]
        try:
            from ultralytics.cfg import entrypoint
        except ImportError as exc:
            print(f"error: ultralytics is not bundled in this build: {exc}",
                  file=sys.stderr)
            return 2
        entrypoint()
        return 0
    if argv[:1] == ["--have-ml"]:
        # Diagnostic used by CI to check that the full build really contains
        # torch/ultralytics (importing them is the only reliable test).
        try:
            import ultralytics  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - any failure means "no"
            print(f"no: {exc}", file=sys.stderr)
            return 1
        print("ok")
        return 0
    if argv[:1] == ["--export-onnx"]:
        try:
            from studio.export_yolo11 import main as export_main
        except ImportError as exc:
            print(f"error: ultralytics is not bundled in this build: {exc}",
                  file=sys.stderr)
            return 2
        return export_main(argv[1:])
    return None


def _ensure_stdio() -> None:
    """A windowed PyInstaller build has no stdout/stderr (both are None)."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def main() -> int:
    _ensure_stdio()
    helper = _run_helper(sys.argv[1:])
    if helper is not None:
        return helper

    _check_linux_qt()
    smoke_test = "--smoke-test" in sys.argv
    if smoke_test:
        sys.argv.remove("--smoke-test")

    try:
        from PySide6.QtCore import QTimer  # noqa: E402
        from PySide6.QtWidgets import QApplication  # noqa: E402
    except ImportError as exc:  # noqa: BLE001 - report instead of a traceback
        print(
            f"error: cannot load Qt: {exc}\n"
            "Some system libraries Qt links against are missing:\n"
            f"{_INSTALL_HINT}\n"
            "Set YOLO_STUDIO_SKIP_QT_CHECK=1 to skip the preflight check.",
            file=sys.stderr,
        )
        return 2

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
