"""Датасет: кадры, YOLO-разметка, сплит для обучения и калибровки."""

from __future__ import annotations

import json
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def imread(path: str | Path) -> np.ndarray | None:
    """cv2.imread, устойчивый к не-ASCII путям."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path: str | Path, frame: np.ndarray, quality: int = 92) -> bool:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return False
    try:
        Path(path).write_bytes(buf.tobytes())
    except OSError:
        return False
    return True

_TRANSLIT = str.maketrans(
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    "abvgdeejzijklmnoprstufhccss_y_eua",
)


def slugify(name: str) -> str:
    """ASCII-имя для файлов моделей: «Квадрокоптер» -> kvadrokopter."""
    s = (name or "").strip().lower().translate(_TRANSLIT)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "object"


def clean_classes(names) -> list[str]:
    """Список имён классов без пустых строк; порядок = id класса."""
    if isinstance(names, str):
        names = [names]
    out = [str(n).strip() for n in names if str(n).strip()]
    return out or ["object"]


@dataclass
class Box:
    """Прямоугольник в пикселях кадра."""

    x: float
    y: float
    w: float
    h: float
    cls: int = 0
    conf: float | None = None

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    def clipped(self, iw: int, ih: int) -> "Box":
        x1 = max(0.0, min(float(iw - 1), float(self.x)))
        y1 = max(0.0, min(float(ih - 1), float(self.y)))
        x2 = max(x1 + 1.0, min(float(iw), float(self.x + self.w)))
        y2 = max(y1 + 1.0, min(float(ih), float(self.y + self.h)))
        return Box(x1, y1, x2 - x1, y2 - y1, self.cls, self.conf)

    def to_yolo(self, iw: int, ih: int) -> tuple[int, float, float, float, float]:
        cx = (self.x + self.w / 2.0) / iw
        cy = (self.y + self.h / 2.0) / ih
        return self.cls, cx, cy, self.w / iw, self.h / ih

    @classmethod
    def from_yolo(cls, class_id: int, cx: float, cy: float, w: float, h: float,
                  iw: int, ih: int) -> "Box":
        return cls_new(class_id, (cx - w / 2.0) * iw, (cy - h / 2.0) * ih,
                       w * iw, h * ih)


def cls_new(cls: int, x: float, y: float, w: float, h: float) -> Box:
    return Box(float(x), float(y), float(w), float(h), int(cls))


class Dataset:
    """Каталог датасета:

    images/   — исходные кадры (JPEG)
    labels/   — YOLO-txt рядом по имени
    calib/    — кадры для INT8-калибровки tpu-mlir
    model/    — .pt / .onnx / .cvimodel
    splits/   — служебные списки
    """

    META = "dataset.json"

    def __init__(self, root: str | Path, create: bool = False,
                 class_name: str = "object",
                 class_names: list[str] | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.class_names = clean_classes(class_names if class_names is not None
                                         else [class_name])
        self._last_thumb: np.ndarray | None = None
        if create:
            self._create(self.class_names)
        else:
            self._load()

    # ---------- создание / загрузка ----------

    def _create(self, class_names: list[str]) -> None:
        for d in ("images", "labels", "model", "calib", "splits"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        meta_path = self.root / self.META
        if meta_path.exists():
            self._load()
            return
        self.class_names = clean_classes(class_names)
        self._write_meta({"version": 2, "classes": self.class_names,
                          "class_name": self.class_names[0],
                          "created": time.strftime("%Y-%m-%d %H:%M:%S")})

    def _load(self) -> None:
        meta_path = self.root / self.META
        if not meta_path.exists():
            raise FileNotFoundError(f"это не датасет: нет {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw = meta.get("classes") or meta.get("class_names")
        if not raw:
            raw = [meta.get("class_name") or "object"]
        self.class_names = clean_classes(raw)

    def _write_meta(self, meta: dict) -> None:
        (self.root / self.META).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @property
    def class_name(self) -> str:
        """Первый класс — для старых мест, где нужен один класс."""
        return self.class_names[0]

    def update_class_names(self, names) -> None:
        self.class_names = clean_classes(names)
        meta: dict = {}
        try:
            old = json.loads((self.root / self.META).read_text(encoding="utf-8"))
            if isinstance(old, dict):
                meta = old
        except (OSError, ValueError):
            pass
        meta.update({"version": 2, "classes": self.class_names,
                     "class_name": self.class_names[0],
                     "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
        self._write_meta(meta)
        # если датасет уже разбит, обновляем список классов и в data.yaml
        if self.data_yaml.is_file():
            try:
                import yaml
                data = yaml.safe_load(self.data_yaml.read_text(encoding="utf-8")) or {}
                if isinstance(data, dict):
                    data["names"] = {i: name
                                     for i, name in enumerate(self.class_names)}
                    self.data_yaml.write_text(
                        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
            except (OSError, ValueError):
                pass

    def update_class_name(self, name: str) -> None:
        """Совместимость: задать единственный класс."""
        self.update_class_names([name])

    # ---------- пути ----------

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    @property
    def calib_dir(self) -> Path:
        return self.root / "calib"

    @property
    def data_yaml(self) -> Path:
        return self.root / "data.yaml"

    @property
    def slug(self) -> str:
        parts = [slugify(name) for name in self.class_names]
        return "_".join(parts)[:64].strip("_") or "model"

    # ---------- кадры ----------

    def image_paths(self) -> list[Path]:
        if not self.images_dir.is_dir():
            return []
        files = [p for p in self.images_dir.iterdir()
                 if p.suffix.lower() in _IMAGE_EXTS]
        return sorted(files, key=lambda p: p.name)

    def _next_index(self) -> int:
        best = 0
        for p in self.image_paths():
            m = re.search(r"(\d+)$", p.stem)
            if m:
                best = max(best, int(m.group(1)))
        return best + 1

    def reset_dedup(self) -> None:
        """Новый ракурс/сцена — первый кадр сохраняем всегда."""
        self._last_thumb = None

    def add_frame(self, frame: np.ndarray | None, dedup: bool = True,
                  min_sharpness: float = 12.0) -> tuple[Path | None, str]:
        """Сохраняет кадр. Возвращает (путь, причина пропуска)."""
        if frame is None or frame.size == 0:
            return None, "пусто"
        h, w = frame.shape[:2]
        if w < 64 or h < 64:
            return None, "мало"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharp < min_sharpness:
            return None, "мутно"
        thumb = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
        if dedup and self._last_thumb is not None:
            diff = float(np.abs(thumb - self._last_thumb).mean())
            if diff < 1.2:
                return None, "дубль"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        path = self.images_dir / f"frame_{self._next_index():05d}.jpg"
        if not imwrite(path, frame):
            return None, "ошибка записи"
        self._last_thumb = thumb
        return path, "ok"

    def delete_image(self, img: Path) -> None:
        img.unlink(missing_ok=True)
        self.label_path(img).unlink(missing_ok=True)
        for split in ("train", "val"):
            for base in ("images", "labels"):
                cand = self.root / base / split / img.name
                cand.unlink(missing_ok=True)
            cand = self.root / "labels" / split / (img.stem + ".txt")
            cand.unlink(missing_ok=True)

    # ---------- разметка ----------

    def label_path(self, img: Path) -> Path:
        return self.labels_dir / (img.stem + ".txt")

    def is_labeled(self, img: Path) -> bool:
        return self.label_path(img).exists()

    def save_label(self, img: Path, boxes: list[Box],
                   size: tuple[int, int] | None = None) -> Path:
        if size is None:
            frame = imread(img)
            if frame is None:
                raise ValueError(f"не читается кадр {img}")
            ih, iw = frame.shape[:2]
        else:
            iw, ih = size
        lines: list[str] = []
        for b in boxes:
            b = b.clipped(iw, ih)
            cls, cx, cy, bw, bh = b.to_yolo(iw, ih)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        path = self.label_path(img)
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path

    def load_label(self, img: Path) -> list[Box]:
        path = self.label_path(img)
        if not path.exists():
            return []
        frame = imread(img)
        if frame is None:
            return []
        ih, iw = frame.shape[:2]
        boxes: list[Box] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
            except ValueError:
                continue
            boxes.append(Box.from_yolo(cls, cx, cy, w, h, iw, ih))
        return boxes

    def stats(self) -> dict:
        images = self.image_paths()
        labeled = 0
        boxes = 0
        for img in images:
            lp = self.label_path(img)
            if lp.exists():
                labeled += 1
                try:
                    boxes += sum(1 for ln in lp.read_text(encoding="utf-8").splitlines()
                                 if ln.strip())
                except OSError:
                    pass
        return {"images": len(images), "labeled": labeled,
                "unlabeled": len(images) - labeled, "boxes": boxes}

    # ---------- сплит и калибровка ----------

    def build_split(self, val_ratio: float = 0.2, seed: int = 42,
                    calib_max: int = 200) -> dict:
        labeled = [p for p in self.image_paths() if self.is_labeled(p)]
        if not labeled:
            raise ValueError("нет размеченных кадров")
        rng = random.Random(seed)
        rng.shuffle(labeled)

        n = len(labeled)
        n_val = max(1, int(round(n * max(0.05, min(0.5, val_ratio))))) if n >= 2 else 0
        val = labeled[:n_val]
        train = labeled[n_val:]
        if not train:
            train = labeled
            val = labeled

        for split in ("train", "val"):
            for base in ("images", "labels"):
                d = self.root / base / split
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)

        for split, items in (("train", train), ("val", val)):
            for img in items:
                shutil.copy2(img, self.root / "images" / split / img.name)
                lp = self.label_path(img)
                if lp.exists():
                    shutil.copy2(lp, self.root / "labels" / split / lp.name)

        names = {i: name for i, name in enumerate(self.class_names)}
        import yaml  # локальный импорт: yaml нужен только здесь
        data = {
            "path": str(self.root),
            "train": "images/train",
            "val": "images/val",
            "names": names,
        }
        self.data_yaml.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8")

        # калибровка INT8: без меток, просто кадры (до calib_max)
        if self.calib_dir.exists():
            shutil.rmtree(self.calib_dir)
        self.calib_dir.mkdir(parents=True, exist_ok=True)
        calib_src = list(labeled)
        rng.shuffle(calib_src)
        calib_src = calib_src[:calib_max]
        for i, img in enumerate(calib_src):
            ext = ".jpg"
            dst = self.calib_dir / f"calib_{i:04d}{ext}"
            shutil.copy2(img, dst)

        return {"train": len(train), "val": len(val), "calib": len(calib_src),
                "yaml": str(self.data_yaml)}

    def best_pt(self) -> Path | None:
        p = self.model_dir / "best.pt"
        if p.exists():
            return p
        found = sorted(self.model_dir.glob("*.pt")) if self.model_dir.is_dir() else []
        return found[0] if found else None

    def best_onnx(self) -> Path | None:
        found = sorted(self.model_dir.glob("*.onnx")) if self.model_dir.is_dir() else []
        return found[0] if found else None

    def best_cvimodel(self) -> Path | None:
        found = sorted(self.model_dir.glob("*.cvimodel")) if self.model_dir.is_dir() else []
        return found[-1] if found else None
