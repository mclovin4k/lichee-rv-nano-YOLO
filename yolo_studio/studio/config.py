"""Application settings: ~/.config/yolo-studio/config.json."""

from __future__ import annotations

import json
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "board_ip": "",  # each board has its own address — set it in Settings
    "board_user": "root",
    "board_password": "root",
    "stream_port": 7777,
    "capture_port": 7778,
    "camera_source": "board",  # board | usb
    "usb_index": 0,
    # Empty means "detect": see studio/pyenv.py. Never hard code a path here —
    # the default used to be ~/.venvs/ultra/bin/python, which does not exist on
    # Windows and broke training with "Python not found".
    "ultra_python": "",
    "tpu_python": "",
    "dataset_root": str(Path.home() / "YOLO Studio" / "datasets"),
    "last_dataset": "",
    "train_base": "yolo11n.pt",
}


class Config:
    def __init__(self, path: Path | None = None) -> None:
        env = os.environ.get("YOLO_STUDIO_CONFIG")
        self.path = Path(env) if env else (
            path or Path.home() / ".config" / "yolo-studio" / "config.json"
        )
        self.data: dict = dict(DEFAULTS)
        self.load()

    def load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.data.update(raw)
        except (OSError, ValueError):
            pass
        self._drop_stale_paths()

    def _drop_stale_paths(self) -> None:
        """Forget configured interpreters that are no longer on disk.

        An old settings file (or one copied from another machine) must not
        stop the application from finding a working Python by itself.
        """
        for key in ("ultra_python", "tpu_python"):
            value = str(self.data.get(key) or "").strip()
            if value and not Path(value).expanduser().is_file():
                self.data[key] = ""

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def get(self, key: str, default=None):
        return self.data.get(key, DEFAULTS.get(key, default))

    def set(self, key: str, value) -> None:
        self.data[key] = value

    def __getitem__(self, key: str):
        return self.get(key)

    def __setitem__(self, key: str, value) -> None:
        self.set(key, value)
