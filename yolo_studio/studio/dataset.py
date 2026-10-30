"""Dataset: frames, YOLO labels, split for training and calibration."""

from __future__ import annotations

import json
import random
import re
import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def imread(path: str | Path) -> np.ndarray | None:
    """cv2.imread that tolerates non-ASCII paths."""
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


_PAD_COLOR = (114, 114, 114)


def parse_size(text: str | None) -> tuple[int, int] | None:
    """Parses an import size: empty/keep -> None, otherwise WIDTHxHEIGHT."""
    s = (text or "").strip().lower().replace("\u00d7", "x").replace("*", "x")
    if not s or s in {"keep", "original", "none", "as is", "as-is"}:
        return None
    match = re.fullmatch(r"(\d{2,5})\s*x\s*(\d{2,5})", s)
    if match is None:
        raise ValueError("image size must be 'keep' or WIDTHxHEIGHT, e.g. 640x480")
    width, height = int(match.group(1)), int(match.group(2))
    if not (32 <= width <= 8192 and 32 <= height <= 8192):
        raise ValueError("image size must be between 32x32 and 8192x8192")
    return width, height


def letterbox(frame: np.ndarray, width: int, height: int,
              color: tuple[int, int, int] = _PAD_COLOR
              ) -> tuple[np.ndarray, float, int, int]:
    """Fits a frame into width x height, keeping the aspect ratio.

    Returns (frame, scale, pad_x, pad_y); the padding is centered and filled
    with gray. The same scale and offsets are used to transform YOLO labels.
    """
    src_h, src_w = frame.shape[:2]
    if (src_w, src_h) == (width, height):
        return frame, 1.0, 0, 0
    scale = min(width / float(src_w), height / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    canvas = np.full((height, width, 3), color, dtype=np.uint8)
    pad_x = (width - new_w) // 2
    pad_y = (height - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


_TRANSLIT = str.maketrans(
    # Cyrillic lowercase letters, escaped to keep the source ASCII-only.
    "\u0430\u0431\u0432\u0433\u0434\u0435\u0451\u0436\u0437\u0438\u0439\u043a"
    "\u043b\u043c\u043d\u043e\u043f\u0440\u0441\u0442\u0443\u0444\u0445\u0446"
    "\u0447\u0448\u0449\u044a\u044b\u044c\u044d\u044e\u044f",
    "abvgdeejzijklmnoprstufhccss_y_eua",
)


def slugify(name: str) -> str:
    """ASCII name for model files (Cyrillic names are transliterated)."""
    s = (name or "").strip().lower().translate(_TRANSLIT)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "object"


def clean_classes(names) -> list[str]:
    """List of class names without empty strings; order = class id."""
    if isinstance(names, str):
        names = [names]
    out = [str(n).strip() for n in names if str(n).strip()]
    return out or ["object"]


@dataclass
class Box:
    """Rectangle in frame pixels."""

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
    """Dataset directory:

    images/   — source frames (JPEG)
    labels/   — YOLO txt files next to them by name
    calib/    — frames for tpu-mlir INT8 calibration
    model/    — .pt / .onnx / .cvimodel
    splits/   — auxiliary lists
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

    # ---------- create / load ----------

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
            raise FileNotFoundError(f"not a dataset: {meta_path} is missing")
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
        """First class — for legacy code paths that need a single class."""
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
        # if the dataset is already split, update the class list in data.yaml too
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
        """Compatibility: set a single class."""
        self.update_class_names([name])

    # ---------- paths ----------

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

    # ---------- frames ----------

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
        """New angle/scene — always save the first frame."""
        self._last_thumb = None

    def add_frame(self, frame: np.ndarray | None, dedup: bool = True,
                  min_sharpness: float = 12.0) -> tuple[Path | None, str]:
        """Saves a frame. Returns (path, skip reason)."""
        if frame is None or frame.size == 0:
            return None, "empty"
        h, w = frame.shape[:2]
        if w < 64 or h < 64:
            return None, "too small"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharp < min_sharpness:
            return None, "blurry"
        thumb = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
        if dedup and self._last_thumb is not None:
            diff = float(np.abs(thumb - self._last_thumb).mean())
            if diff < 1.2:
                return None, "duplicate"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        path = self.images_dir / f"frame_{self._next_index():05d}.jpg"
        if not imwrite(path, frame):
            return None, "write error"
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

    # ---------- labels ----------

    def label_path(self, img: Path) -> Path:
        return self.labels_dir / (img.stem + ".txt")

    def is_labeled(self, img: Path) -> bool:
        return self.label_path(img).exists()

    def save_label(self, img: Path, boxes: list[Box],
                   size: tuple[int, int] | None = None) -> Path:
        if size is None:
            frame = imread(img)
            if frame is None:
                raise ValueError(f"cannot read frame {img}")
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

    # ---------- import ----------

    @staticmethod
    def _source_classes(source: Path) -> list[str] | None:
        """Reads class names from a YOLO data.yaml / classes.txt if present."""
        for name in ("data.yaml", "data.yml"):
            path = source / name
            if not path.is_file():
                continue
            try:
                import yaml
            except ImportError:
                continue
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, yaml.YAMLError):
                continue
            names = data.get("names") if isinstance(data, dict) else None
            if isinstance(names, dict):
                try:
                    keys = sorted(names, key=lambda k: int(k))
                except (TypeError, ValueError):
                    keys = sorted(names, key=str)
                names = [names[k] for k in keys]
            if isinstance(names, (list, tuple)) and names:
                return [str(n).strip() or f"class_{i}"
                        for i, n in enumerate(names)]
        for name in ("classes.txt", "obj.names", "names.txt"):
            path = source / name
            if not path.is_file():
                continue
            try:
                names = [line.strip() for line in
                         path.read_text(encoding="utf-8").splitlines()
                         if line.strip()]
            except (OSError, UnicodeDecodeError):
                continue
            if names:
                return names
        return None

    @staticmethod
    def _label_sidecar(img: Path) -> Path | None:
        """Finds a YOLO .txt next to the image or in the sibling labels/ dir."""
        direct = img.with_suffix(".txt")
        if direct.is_file():
            return direct
        parts = img.parts
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            candidate = Path(*parts[:idx], "labels",
                             *parts[idx + 1:]).with_suffix(".txt")
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _read_yolo_label(
            path: Path) -> list[tuple[int, float, float, float, float]]:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
        boxes: list[tuple[int, float, float, float, float]] = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            if bw <= 0.0 or bh <= 0.0:
                continue
            boxes.append((cls, cx, cy, bw, bh))
        return boxes

    def import_images(self, source: str | Path | Iterable[str | Path],
                      size: tuple[int, int] | str | None = None,
                      recursive: bool = True,
                      progress: Callable[[int, int, str], None] | None = None
                      ) -> dict:
        """Copies external images (and YOLO labels) into the dataset.

        ``source`` may be a folder or a list of image paths. JPEG/PNG/BMP images
        are copied into ``images/`` as ``frame_XXXXX.jpg``. If a YOLO ``.txt``
        with the same name exists next to an image (or in the sibling
        ``labels/`` folder), it is imported and renamed accordingly.

        ``size`` is ``(width, height)`` or a string such as ``"640x480"``. Images
        are letterboxed into that size (aspect ratio preserved, gray padding)
        and label coordinates are transformed with the same scale and offsets.
        ``None``/``"keep"`` keeps the original size.

        Class names are taken from ``data.yaml`` / ``classes.txt`` in the source
        folder when present and merged into this dataset by name, so label ids
        are remapped correctly.

        Returns counters: images, labeled, skipped, total, errors, classes, size.
        """
        if isinstance(size, str):
            size = parse_size(size)
        target: tuple[int, int] | None = None
        if size is not None:
            try:
                tw, th = int(size[0]), int(size[1])
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError("size must be (width, height)") from exc
            if not (32 <= tw <= 8192 and 32 <= th <= 8192):
                raise ValueError("image size must be between 32x32 and 8192x8192")
            target = (tw, th)

        source_names: list[str] | None = None
        if isinstance(source, (str, Path)):
            src = Path(source).expanduser()
            if src.is_dir():
                iterator = src.rglob("*") if recursive else src.iterdir()
                images = sorted(
                    (p for p in iterator
                     if p.is_file() and p.suffix.lower() in _IMAGE_EXTS),
                    key=lambda p: str(p).lower())
                source_names = self._source_classes(src)
            elif src.is_file():
                images = [src]
            else:
                raise FileNotFoundError(f"no such file or folder: {src}")
        else:
            images = [Path(p).expanduser() for p in source
                      if Path(p).suffix.lower() in _IMAGE_EXTS]
        if not images:
            raise ValueError("no images found (expected .jpg, .jpeg, .png or .bmp)")

        class_map: dict[int, int] = {}
        if source_names:
            names = list(self.class_names)
            for source_id, raw_name in enumerate(source_names):
                name = str(raw_name).strip() or f"class_{source_id}"
                if name not in names:
                    names.append(name)
                class_map[source_id] = names.index(name)
            if names != self.class_names:
                self.update_class_names(names)

        def ensure_classes(count: int) -> None:
            if count > len(self.class_names):
                self.update_class_names(
                    self.class_names
                    + [f"class_{i}" for i in range(len(self.class_names), count)])

        self.images_dir.mkdir(parents=True, exist_ok=True)
        next_index = self._next_index()
        imported = 0
        labeled = 0
        skipped = 0
        errors: list[str] = []
        error_count = 0
        total = len(images)

        for done, img_path in enumerate(images):
            if progress is not None:
                progress(done, total, img_path.name)
            frame = imread(img_path)
            if frame is None or frame.size == 0:
                skipped += 1
                error_count += 1
                if len(errors) < 30:
                    errors.append(f"{img_path}: cannot read the image")
                continue
            old_h, old_w = frame.shape[:2]

            label_src = self._label_sidecar(img_path)
            labels = self._read_yolo_label(label_src) if label_src else []
            if labels:
                ensure_classes(max(class_map.get(cls, cls) for cls, *_ in labels) + 1)

            scale, pad_x, pad_y = 1.0, 0, 0
            if target is not None:
                frame, scale, pad_x, pad_y = letterbox(frame, target[0], target[1])
                tw, th = target
            else:
                tw, th = old_w, old_h

            dest = self.images_dir / f"frame_{next_index:05d}.jpg"
            while dest.exists():
                next_index += 1
                dest = self.images_dir / f"frame_{next_index:05d}.jpg"
            if not imwrite(dest, frame, quality=95):
                skipped += 1
                error_count += 1
                if len(errors) < 30:
                    errors.append(f"{img_path}: cannot write {dest.name}")
                continue
            next_index += 1
            imported += 1

            if label_src is not None:
                lines: list[str] = []
                for cls, cx, cy, bw, bh in labels:
                    new_cls = class_map.get(cls, cls)
                    if new_cls >= len(self.class_names):
                        continue
                    cx_t = (cx * old_w * scale + pad_x) / tw
                    cy_t = (cy * old_h * scale + pad_y) / th
                    bw_t = bw * old_w * scale / tw
                    bh_t = bh * old_h * scale / th
                    x1 = max(0.0, min(1.0, cx_t - bw_t / 2.0))
                    y1 = max(0.0, min(1.0, cy_t - bh_t / 2.0))
                    x2 = max(0.0, min(1.0, cx_t + bw_t / 2.0))
                    y2 = max(0.0, min(1.0, cy_t + bh_t / 2.0))
                    if x2 - x1 < 1e-6 or y2 - y1 < 1e-6:
                        continue
                    lines.append(
                        f"{new_cls} {(x1 + x2) / 2.0:.6f} {(y1 + y2) / 2.0:.6f} "
                        f"{x2 - x1:.6f} {y2 - y1:.6f}")
                try:
                    self.labels_dir.mkdir(parents=True, exist_ok=True)
                    self.label_path(dest).write_text(
                        "\n".join(lines) + ("\n" if lines else ""),
                        encoding="utf-8")
                    labeled += 1
                except OSError as exc:
                    error_count += 1
                    if len(errors) < 30:
                        errors.append(f"{label_src}: cannot write the label: {exc}")

        return {
            "images": imported,
            "labeled": labeled,
            "skipped": skipped,
            "total": total,
            "error_count": error_count,
            "errors": errors,
            "classes": list(self.class_names),
            "size": target,
        }

    # ---------- split and calibration ----------

    def build_split(self, val_ratio: float = 0.2, seed: int = 42,
                    calib_max: int = 200) -> dict:
        labeled = [p for p in self.image_paths() if self.is_labeled(p)]
        if not labeled:
            raise ValueError("no labeled frames")
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
        import yaml  # local import: yaml is only needed here
        data = {
            "path": str(self.root),
            "train": "images/train",
            "val": "images/val",
            "names": names,
        }
        self.data_yaml.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8")

        # INT8 calibration: no labels, just frames (up to calib_max)
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
