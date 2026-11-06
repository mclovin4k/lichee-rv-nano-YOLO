"""Model page: dataset → training → ONNX → cvimodel."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget,
)

from .. import pipeline, pyenv
from ..dataset import Dataset
from .theme import accent, danger, primary
from .widgets import LogPane, Worker


class ModelPage(QWidget):
    def __init__(self, main) -> None:
        super().__init__()
        self.main = main
        self.cfg = main.config
        self._worker: Worker | None = None
        self._busy_buttons: list[QPushButton] = []
        self._build_ui()

    @property
    def dataset(self) -> Dataset | None:
        return self.main.dataset

    # ---------- UI ----------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        top = QHBoxLayout()

        left = QVBoxLayout()
        left.setSpacing(10)

        ds_box = QGroupBox("1. Dataset")
        dl = QVBoxLayout(ds_box)
        dl.setSpacing(8)
        self.ds_info = QLabel("no dataset selected")
        self.ds_info.setWordWrap(True)
        self.ds_info.setMinimumHeight(38)
        self.ds_info.setAlignment(Qt.AlignmentFlag.AlignTop)
        dl.addWidget(self.ds_info)
        cls_row = QHBoxLayout()
        cls_row.addWidget(QLabel("Classes:"))
        self.class_edit = QLineEdit()
        self.class_edit.setPlaceholderText("e.g. drone, person")
        self.class_edit.setToolTip(
            "Comma-separated class names. Order = id: the first is 0, the second is 1. "
            "Updates dataset.json and data.yaml.")
        self.class_edit.returnPressed.connect(self._save_class)
        cls_row.addWidget(self.class_edit, 1)
        self.btn_class = QPushButton("Save")
        self.btn_class.setToolTip("Save the class list to the dataset")
        self.btn_class.clicked.connect(self._save_class)
        cls_row.addWidget(self.btn_class)
        dl.addLayout(cls_row)
        self.btn_split = QPushButton("Prepare dataset (train/val + calibration)")
        self.btn_split.clicked.connect(self._prepare_split)
        primary(self.btn_split)
        dl.addWidget(self.btn_split)
        self.split_info = QLabel("")
        self.split_info.setObjectName("Muted")
        self.split_info.setWordWrap(True)
        dl.addWidget(self.split_info)
        left.addWidget(ds_box)

        pt_box = QGroupBox("2. Weights")
        pl = QVBoxLayout(pt_box)
        pl.setSpacing(8)
        self.pt_info = QLabel("weights: none")
        self.pt_info.setObjectName("Muted")
        self.pt_info.setWordWrap(True)
        pl.addWidget(self.pt_info)
        self.btn_import = QPushButton("Import ready .pt…")
        self.btn_import.clicked.connect(self._import_pt)
        accent(self.btn_import)
        pl.addWidget(self.btn_import)
        left.addWidget(pt_box)

        train_box = QGroupBox("3. Training")
        tl = QFormLayout(train_box)
        tl.setSpacing(8)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(60)
        tl.addRow("Epochs:", self.epochs)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(256, 960)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)
        tl.addRow("imgsz:", self.imgsz)
        self.batch = QSpinBox()
        self.batch.setRange(1, 64)
        self.batch.setValue(8)
        tl.addRow("batch:", self.batch)
        self.base_model = QLineEdit(str(self.cfg.get("train_base", "yolo11n.pt")))
        tl.addRow("base:", self.base_model)
        self.btn_train = QPushButton("Train on CPU")
        self.btn_train.clicked.connect(self._train)
        primary(self.btn_train)
        tl.addWidget(self.btn_train)
        warn = QLabel("CPU training is slow: for initial experiments, a small "
                      "dataset and imgsz 416–640 are enough.\n"
                      "Base model: yolo11n.pt (requires internet or a local "
                      "file) or yolo11n.yaml — train from scratch without downloading.")
        warn.setObjectName("Warning")
        warn.setWordWrap(True)
        tl.addWidget(warn)
        left.addWidget(train_box)

        exp_box = QGroupBox("4. Export and conversion")
        el = QVBoxLayout(exp_box)
        el.setSpacing(8)
        self.btn_onnx = QPushButton("Export ONNX (6 outputs)")
        self.btn_onnx.clicked.connect(self._export_onnx)
        accent(self.btn_onnx)
        el.addWidget(self.btn_onnx)
        self.onnx_info = QLabel("ONNX: none")
        self.onnx_info.setObjectName("Muted")
        self.onnx_info.setWordWrap(True)
        el.addWidget(self.onnx_info)
        name_row = QVBoxLayout()
        name_row.addWidget(QLabel("Model file name:"))
        self.out_name = QLineEdit()
        name_row.addWidget(self.out_name)
        el.addLayout(name_row)
        self.btn_cvimodel = QPushButton("Build cvimodel")
        self.btn_cvimodel.clicked.connect(self._build_cvimodel)
        primary(self.btn_cvimodel)
        el.addWidget(self.btn_cvimodel)
        self.cv_info = QLabel("cvimodel: none")
        self.cv_info.setObjectName("Muted")
        self.cv_info.setWordWrap(True)
        el.addWidget(self.cv_info)
        left.addWidget(exp_box)
        left.addStretch(1)

        left_w = QWidget()
        left_w.setLayout(left)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(424)
        scroll.setWidget(left_w)
        top.addWidget(scroll)

        right = QVBoxLayout()
        right.setSpacing(8)
        log_head = QHBoxLayout()
        log_head.addWidget(QLabel("Build log"))
        log_head.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        danger(self.btn_cancel)
        log_head.addWidget(self.btn_cancel)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(lambda: self.log.clear())
        log_head.addWidget(btn_clear)
        right.addLayout(log_head)
        self.log = LogPane()
        right.addWidget(self.log, 1)
        top.addLayout(right, 1)
        root.addLayout(top, 1)

        self._busy_buttons = [self.btn_split, self.btn_import, self.btn_train,
                              self.btn_onnx, self.btn_cvimodel, self.btn_class]

    # ---------- state ----------

    def refresh(self) -> None:
        ds = self.dataset
        if ds is None:
            self.ds_info.setText("no dataset selected")
            self.class_edit.clear()
        else:
            if not self.class_edit.hasFocus():
                self.class_edit.setText(", ".join(ds.class_names))
            st = ds.stats()
            self.ds_info.setText(
                f"{ds.root}\nclasses: {len(ds.class_names)} · frames: {st['images']} · "
                f"labeled: {st['labeled']} · boxes: {st['boxes']}"
                + ("" if st["unlabeled"] == 0 else
                   f" · unlabeled: {st['unlabeled']}"))
            if ds.data_yaml.is_file():
                self.split_info.setText(f"data.yaml: {ds.data_yaml}")
            else:
                self.split_info.setText("dataset has not been split into train/val yet")
        pt = ds.best_pt() if ds else None
        self.pt_info.setText(f"weights: {pt}" if pt else "weights: none (train or import a .pt)")
        onnx = ds.best_onnx() if ds else None
        self.onnx_info.setText(f"ONNX: {onnx}" if onnx else "ONNX: none")
        calib = pipeline.count_calib(ds.calib_dir) if ds else 0
        cv = ds.best_cvimodel() if ds else None
        self.cv_info.setText(
            (f"cvimodel: {cv}\n" if cv else "cvimodel: none\n") +
            f"calibration frames: {calib}")
        if ds is not None:
            imgsz = int(self.imgsz.value())
            if not self.out_name.text().strip() or \
                    self.out_name.text().startswith("yolov11n_det_"):
                self.out_name.setText(
                    f"yolov11n_det_{ds.slug}_{imgsz}_{imgsz}_INT8_cv181x.cvimodel")
        self._update_enabled()

    def _update_enabled(self) -> None:
        busy = self._worker is not None
        ds = self.dataset
        for b in self._busy_buttons:
            b.setEnabled(not busy)
        self.class_edit.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.btn_train.setEnabled(not busy and ds is not None and
                                  ds.data_yaml.is_file())
        self.btn_onnx.setEnabled(not busy and ds is not None and ds.best_pt() is not None)
        self.btn_cvimodel.setEnabled(not busy and ds is not None and
                                     ds.best_onnx() is not None and
                                     pipeline.count_calib(ds.calib_dir) > 0)

    def showEvent(self, event) -> None:
        self.refresh()
        super().showEvent(event)

    # ---------- actions ----------

    def _sync_classes(self) -> bool:
        """Applies the class list from the field to the dataset. False means error."""
        ds = self.dataset
        if ds is None:
            return True
        names = [part.strip() for part in re.split(r"[,;]", self.class_edit.text())
                 if part.strip()]
        if not names:
            QMessageBox.warning(self, "Classes",
                                "Enter at least one class name.")
            return False
        if names == ds.class_names:
            self.class_edit.setText(", ".join(names))
            return True
        try:
            ds.update_class_names(names)
        except OSError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return False
        self.main.set_dataset(ds)
        return True

    def _save_class(self) -> None:
        ds = self.dataset
        if ds is None:
            QMessageBox.information(self, "No dataset", "Create a dataset first.")
            return
        if not self._sync_classes():
            return
        self.main.statusBar().showMessage(
            f"classes: {', '.join(ds.class_names)} "
            f"(id 0…{len(ds.class_names) - 1})")

    def _prepare_split(self) -> None:
        ds = self.dataset
        if ds is None:
            QMessageBox.information(self, "No dataset", "Create a dataset first.")
            return
        if not self._sync_classes():
            return
        try:
            info = ds.build_split()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error", str(exc))
            return
        self.split_info.setText(
            f"done: train {info['train']}, val {info['val']}, "
            f"calibration {info['calib']}\n{info['yaml']}")
        self.log.append_line(
            f"dataset: train={info['train']} val={info['val']} calib={info['calib']}")
        self.refresh()

    def _import_pt(self) -> None:
        ds = self.dataset
        if ds is None:
            QMessageBox.information(self, "No dataset", "Create a dataset first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select .pt", "", "PyTorch (*.pt)")
        if not path:
            return
        src = Path(path)
        ds.model_dir.mkdir(parents=True, exist_ok=True)
        dst = ds.model_dir / src.name
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return
        self.log.append_line(f"imported weights: {dst}")
        self.refresh()

    def _train(self) -> None:
        ds = self.dataset
        if ds is None:
            return
        if not self._sync_classes():
            return
        if not ds.data_yaml.is_file():
            QMessageBox.information(self, "No data.yaml",
                                    "Click “Prepare dataset” first.")
            return
        ultra = self._python("ultra")
        if ultra is None:
            return
        self.cfg["train_base"] = self.base_model.text().strip() or "yolo11n.pt"
        self.cfg.save()
        if QMessageBox.question(
                self, "Train on CPU",
                f"Train {self.base_model.text()} on CPU: "
                f"{self.epochs.value()} epochs, imgsz {self.imgsz.value()}?\n"
                "This may take hours.") != QMessageBox.StandardButton.Yes:
            return
        self.log.append_line(
            f"$ training: epochs={self.epochs.value()} imgsz={self.imgsz.value()} "
            f"batch={self.batch.value()}")
        self._run(
            pipeline.train_local,
            ds.data_yaml, ds.model_dir / "runs", ultra,
            epochs=self.epochs.value(), imgsz=self.imgsz.value(),
            batch=self.batch.value(), base_model=self.cfg["train_base"],
            on_done=self._train_done)

    def _train_done(self, best: Path) -> None:
        ds = self.dataset
        if ds is None:
            return
        dst = ds.model_dir / "best.pt"
        try:
            shutil.copy2(best, dst)
        except OSError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return
        self.log.append_line(f"best weights: {dst}")
        self.refresh()

    def _export_onnx(self) -> None:
        ds = self.dataset
        pt = ds.best_pt() if ds else None
        if ds is None or pt is None:
            return
        ultra = self._python("ultra")
        if ultra is None:
            return
        self.log.append_line(f"$ ONNX export (imgsz={self.imgsz.value()})…")
        self._run(pipeline.export_onnx, pt, ultra, imgsz=self.imgsz.value(),
                  on_done=self._onnx_done)

    def _onnx_done(self, onnx: Path) -> None:
        self.log.append_line(f"ONNX: {onnx}")
        ds = self.dataset
        if ds is not None:
            imgsz = int(self.imgsz.value())
            self.out_name.setText(
                f"yolov11n_det_{ds.slug}_{imgsz}_{imgsz}_INT8_cv181x.cvimodel")
        self.refresh()

    def _build_cvimodel(self) -> None:
        ds = self.dataset
        onnx = ds.best_onnx() if ds else None
        if ds is None or onnx is None:
            return
        tpu = self._python("tpu")
        if tpu is None:
            return
        name = self.out_name.text().strip()
        if not name:
            name = f"model_{ds.slug}.cvimodel"
        if not name.endswith(".cvimodel"):
            name += ".cvimodel"
        self.log.append_line(f"$ cvimodel build: {name}")
        self._run(
            pipeline.build_cvimodel, onnx, ds.calib_dir, name, tpu,
            ds.model_dir / "build", imgsz=self.imgsz.value(),
            on_done=self._cv_done)

    def _cv_done(self, path: Path) -> None:
        self.main.last_cvimodel = path
        self.log.append_line(f"done: {path}")
        ds = self.dataset
        if ds is not None:
            names = path.with_suffix(".names")
            try:
                names.write_text("\n".join(ds.class_names) + "\n", encoding="utf-8")
                self.log.append_line(f"class names: {names.name}")
            except OSError as exc:
                self.log.append_line(f"failed to write {names.name}: {exc}")
        self.refresh()
        if QMessageBox.question(self, "Model built",
                                "cvimodel is ready. Upload it to the board now?") \
                == QMessageBox.StandardButton.Yes:
            self.main.go_to_board(path)

    # ---------- external python ----------

    def _python(self, kind: str) -> str | None:
        """The configured or detected interpreter, or None (after telling the
        user what to install). Detection results are saved back to settings."""
        info = pyenv.KINDS[kind]
        key = info["config_key"]
        if pyenv.bundled(kind):
            # The full build carries ultralytics itself: nothing to configure.
            return str(pyenv.resolve(kind))
        configured = str(self.cfg.get(key) or "")
        exe = pyenv.resolve(kind, configured)
        if exe is None:
            self._python_missing(kind)
            return None
        if str(exe) != configured:
            self.cfg[key] = str(exe)
            self.cfg.save()
            self.log.append_line(f"{info['label']}: {exe}")
        return str(exe)

    def _python_missing(self, kind: str) -> None:
        info = pyenv.KINDS[kind]
        wsl_tpu = kind == "tpu" and pyenv.is_win()
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Python not found")
        box.setText(f"No Python environment with {info['install']} was found, "
                    f"so {info['purpose']} cannot run.")
        box.setInformativeText(pyenv.missing_help(kind))
        locate = None
        if not wsl_tpu:
            locate = box.addButton("Locate interpreter…",
                                   QMessageBox.ButtonRole.ActionRole)
        create = box.addButton(
            "Create WSL environment…" if wsl_tpu else "Create environment…",
            QMessageBox.ButtonRole.ActionRole)
        search = box.addButton("Search again",
                               QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is create:
            target = pyenv.venv_target(kind)
            if wsl_tpu:
                question = (
                    f"Create {target} in the default WSL 2 distribution and "
                    f"install {info['install']} into it?\n\n"
                    "No sudo, apt or manual pip is needed: if WSL does not "
                    "have Python 3.10, the Studio downloads a private copy "
                    "with uv.\n\n"
                    "About 1.5 GB will be downloaded once (about 2.5 GB on "
                    "disk) and the installation can take several minutes.\n\n"
                    "WSL 2 itself must already be installed: if it is not, "
                    "cancel and run \"wsl --install\" in PowerShell first.")
            elif kind == "tpu":
                question = (
                    f"Create {target} and install tpu-mlir into it?\n\n"
                    "No sudo or system packages are needed: if Python 3.10 "
                    "is missing, the Studio downloads a private copy with "
                    "uv.\n\n"
                    "About 1.5 GB will be downloaded once (about 2.5 GB on "
                    "disk) and the installation can take several minutes.")
            else:
                question = (
                    f"Create {target} and install {info['install']} into it?\n\n"
                    "About 400 MB will be downloaded once; PyTorch is large "
                    "and the installation can take several minutes.")
            if QMessageBox.question(self, "Create environment", question) \
                    != QMessageBox.StandardButton.Yes:
                return
            self.log.append_line(f"creating {target}…")
            self._run(pyenv.create_env, kind,
                      on_done=lambda path: self._env_created(kind, path))
            return
        if locate is not None and clicked is locate:
            path, _ = QFileDialog.getOpenFileName(
                self, info["label"], str(Path.home()))
            if not path:
                return
            self.cfg[info["config_key"]] = path
            self.cfg.save()
            self.log.append_line(f"{info['label']}: {path}")
        elif clicked is search:
            self.log.append_line(f"searching for {info['install']}…")
            exe = pyenv.detect(kind, force=True)
            if exe is None:
                self.log.append_line(f"{info['install']} was not found")
                return
            self.cfg[info["config_key"]] = str(exe)
            self.cfg.save()
            self.log.append_line(f"{info['label']}: {exe}")

    def _env_created(self, kind: str, path) -> None:
        info = pyenv.KINDS[kind]
        if path is None:
            return
        self.cfg[info["config_key"]] = str(path)
        self.cfg.save()
        self.log.append_line(f"{info['label']}: {path}")

    # ---------- background ----------

    def _run(self, fn, *args, on_done, **kwargs) -> None:
        if self._worker is not None:
            return
        worker = Worker(fn, *args, **kwargs)
        self._worker = worker
        self._update_enabled()
        worker.line.connect(self.log.append_line)
        worker.done.connect(lambda res: self._worker_done(on_done, res))
        worker.failed.connect(self._worker_failed)
        worker.finished.connect(self._worker_finished)
        worker.start()

    def _worker_done(self, on_done, result) -> None:
        try:
            on_done(result)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error", str(exc))

    def _worker_failed(self, message: str) -> None:
        self.log.append_line(f"ERROR: {message}")
        QMessageBox.critical(self, "Failed", message)

    def _worker_finished(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        self._update_enabled()

    def _cancel(self) -> None:
        if self._worker is not None:
            self.log.append_line("cancelling…")
            self._worker.cancel()

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(3000)
