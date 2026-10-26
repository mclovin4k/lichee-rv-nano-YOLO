"""Страница модели: датасет → обучение → ONNX → cvimodel."""

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

from .. import pipeline
from ..dataset import Dataset
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

        ds_box = QGroupBox("1. Датасет")
        dl = QVBoxLayout(ds_box)
        self.ds_info = QLabel("датасет не выбран")
        self.ds_info.setWordWrap(True)
        self.ds_info.setMinimumHeight(38)
        self.ds_info.setAlignment(Qt.AlignmentFlag.AlignTop)
        dl.addWidget(self.ds_info)
        cls_row = QHBoxLayout()
        cls_row.addWidget(QLabel("Классы:"))
        self.class_edit = QLineEdit()
        self.class_edit.setPlaceholderText("например, drone, person")
        self.class_edit.setToolTip(
            "Имена классов через запятую. Порядок = id: первый — 0, второй — 1. "
            "Обновляет dataset.json и data.yaml.")
        self.class_edit.returnPressed.connect(self._save_class)
        cls_row.addWidget(self.class_edit, 1)
        self.btn_class = QPushButton("Сохранить")
        self.btn_class.setToolTip("Сохранить список классов в датасете")
        self.btn_class.clicked.connect(self._save_class)
        cls_row.addWidget(self.btn_class)
        dl.addLayout(cls_row)
        self.btn_split = QPushButton("Подготовить датасет (train/val + калибровка)")
        self.btn_split.clicked.connect(self._prepare_split)
        dl.addWidget(self.btn_split)
        self.split_info = QLabel("")
        self.split_info.setWordWrap(True)
        dl.addWidget(self.split_info)
        left.addWidget(ds_box)

        pt_box = QGroupBox("2. Веса")
        pl = QVBoxLayout(pt_box)
        self.pt_info = QLabel("веса: нет")
        self.pt_info.setWordWrap(True)
        pl.addWidget(self.pt_info)
        self.btn_import = QPushButton("Импорт готового .pt…")
        self.btn_import.clicked.connect(self._import_pt)
        pl.addWidget(self.btn_import)
        left.addWidget(pt_box)

        train_box = QGroupBox("3. Обучение")
        tl = QFormLayout(train_box)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(60)
        tl.addRow("Эпох:", self.epochs)
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
        tl.addRow("базовая:", self.base_model)
        self.btn_train = QPushButton("Обучить на CPU")
        self.btn_train.clicked.connect(self._train)
        tl.addWidget(self.btn_train)
        warn = QLabel("CPU-обучение медленное: для первых опытов хватит "
                      "небольшого датасета и imgsz 416–640.\n"
                      "Базовая модель: yolo11n.pt (нужен интернет или локальный "
                      "файл) либо yolo11n.yaml — обучение с нуля без скачивания.")
        warn.setWordWrap(True)
        warn.setStyleSheet("color: #b45309;")
        tl.addWidget(warn)
        left.addWidget(train_box)

        exp_box = QGroupBox("4. Экспорт и конвертация")
        el = QVBoxLayout(exp_box)
        self.btn_onnx = QPushButton("Экспорт ONNX (6 выходов)")
        self.btn_onnx.clicked.connect(self._export_onnx)
        el.addWidget(self.btn_onnx)
        self.onnx_info = QLabel("ONNX: нет")
        self.onnx_info.setWordWrap(True)
        el.addWidget(self.onnx_info)
        name_row = QVBoxLayout()
        name_row.addWidget(QLabel("Имя файла модели:"))
        self.out_name = QLineEdit()
        name_row.addWidget(self.out_name)
        el.addLayout(name_row)
        self.btn_cvimodel = QPushButton("Собрать cvimodel")
        self.btn_cvimodel.clicked.connect(self._build_cvimodel)
        el.addWidget(self.btn_cvimodel)
        self.cv_info = QLabel("cvimodel: нет")
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
        scroll.setFixedWidth(402)
        scroll.setWidget(left_w)
        top.addWidget(scroll)

        right = QVBoxLayout()
        log_head = QHBoxLayout()
        log_head.addWidget(QLabel("Лог сборки"))
        log_head.addStretch(1)
        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        log_head.addWidget(self.btn_cancel)
        btn_clear = QPushButton("Очистить")
        btn_clear.clicked.connect(lambda: self.log.clear())
        log_head.addWidget(btn_clear)
        right.addLayout(log_head)
        self.log = LogPane()
        right.addWidget(self.log, 1)
        top.addLayout(right, 1)
        root.addLayout(top, 1)

        self._busy_buttons = [self.btn_split, self.btn_import, self.btn_train,
                              self.btn_onnx, self.btn_cvimodel, self.btn_class]

    # ---------- состояние ----------

    def refresh(self) -> None:
        ds = self.dataset
        if ds is None:
            self.ds_info.setText("датасет не выбран")
            self.class_edit.clear()
        else:
            if not self.class_edit.hasFocus():
                self.class_edit.setText(", ".join(ds.class_names))
            st = ds.stats()
            self.ds_info.setText(
                f"{ds.root}\nклассов: {len(ds.class_names)} · кадров: {st['images']} · "
                f"размечено: {st['labeled']} · рамок: {st['boxes']}"
                + ("" if st["unlabeled"] == 0 else
                   f" · без разметки: {st['unlabeled']}"))
            if ds.data_yaml.is_file():
                self.split_info.setText(f"data.yaml: {ds.data_yaml}")
            else:
                self.split_info.setText("датасет ещё не разбит на train/val")
        pt = ds.best_pt() if ds else None
        self.pt_info.setText(f"веса: {pt}" if pt else "веса: нет (обучите или импортируйте .pt)")
        onnx = ds.best_onnx() if ds else None
        self.onnx_info.setText(f"ONNX: {onnx}" if onnx else "ONNX: нет")
        calib = pipeline.count_calib(ds.calib_dir) if ds else 0
        cv = ds.best_cvimodel() if ds else None
        self.cv_info.setText(
            (f"cvimodel: {cv}\n" if cv else "cvimodel: нет\n") +
            f"кадров для калибровки: {calib}")
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

    # ---------- действия ----------

    def _sync_classes(self) -> bool:
        """Применяет список классов из поля к датасету. False — ошибка."""
        ds = self.dataset
        if ds is None:
            return True
        names = [part.strip() for part in re.split(r"[,;]", self.class_edit.text())
                 if part.strip()]
        if not names:
            QMessageBox.warning(self, "Классы",
                                "Укажите хотя бы одно имя класса.")
            return False
        if names == ds.class_names:
            self.class_edit.setText(", ".join(names))
            return True
        try:
            ds.update_class_names(names)
        except OSError as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return False
        self.main.set_dataset(ds)
        return True

    def _save_class(self) -> None:
        ds = self.dataset
        if ds is None:
            QMessageBox.information(self, "Нет датасета", "Сначала создайте датасет.")
            return
        if not self._sync_classes():
            return
        self.main.statusBar().showMessage(
            f"классы: {', '.join(ds.class_names)} "
            f"(id 0…{len(ds.class_names) - 1})")

    def _prepare_split(self) -> None:
        ds = self.dataset
        if ds is None:
            QMessageBox.information(self, "Нет датасета", "Сначала соберите датасет.")
            return
        if not self._sync_classes():
            return
        try:
            info = ds.build_split()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", str(exc))
            return
        self.split_info.setText(
            f"готово: train {info['train']}, val {info['val']}, "
            f"калибровка {info['calib']}\n{info['yaml']}")
        self.log.append_line(
            f"датасет: train={info['train']} val={info['val']} calib={info['calib']}")
        self.refresh()

    def _import_pt(self) -> None:
        ds = self.dataset
        if ds is None:
            QMessageBox.information(self, "Нет датасета", "Сначала создайте датасет.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Выберите .pt", "", "PyTorch (*.pt)")
        if not path:
            return
        src = Path(path)
        ds.model_dir.mkdir(parents=True, exist_ok=True)
        dst = ds.model_dir / src.name
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return
        self.log.append_line(f"импорт весов: {dst}")
        self.refresh()

    def _train(self) -> None:
        ds = self.dataset
        if ds is None:
            return
        if not self._sync_classes():
            return
        if not ds.data_yaml.is_file():
            QMessageBox.information(self, "Нет data.yaml",
                                    "Сначала нажмите «Подготовить датасет».")
            return
        ultra = str(self.cfg.get("ultra_python"))
        if not Path(ultra).is_file():
            QMessageBox.critical(self, "Нет Python",
                                 f"Не найден {ultra}. Укажите путь в настройках.")
            return
        self.cfg["train_base"] = self.base_model.text().strip() or "yolo11n.pt"
        self.cfg.save()
        if QMessageBox.question(
                self, "Обучение на CPU",
                f"Обучить {self.base_model.text()} на CPU: "
                f"{self.epochs.value()} эпох, imgsz {self.imgsz.value()}?\n"
                "Это может занять часы.") != QMessageBox.StandardButton.Yes:
            return
        self.log.append_line(
            f"$ обучение: epochs={self.epochs.value()} imgsz={self.imgsz.value()} "
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
            QMessageBox.critical(self, "Ошибка", str(exc))
            return
        self.log.append_line(f"лучшие веса: {dst}")
        self.refresh()

    def _export_onnx(self) -> None:
        ds = self.dataset
        pt = ds.best_pt() if ds else None
        if ds is None or pt is None:
            return
        ultra = str(self.cfg.get("ultra_python"))
        if not Path(ultra).is_file():
            QMessageBox.critical(self, "Нет Python",
                                 f"Не найден {ultra}. Укажите путь в настройках.")
            return
        self.log.append_line(f"$ экспорт ONNX (imgsz={self.imgsz.value()})…")
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
        tpu = str(self.cfg.get("tpu_python"))
        if not Path(tpu).is_file():
            QMessageBox.critical(self, "Нет Python",
                                 f"Не найден {tpu}. Укажите путь в настройках.")
            return
        name = self.out_name.text().strip()
        if not name:
            name = f"model_{ds.slug}.cvimodel"
        if not name.endswith(".cvimodel"):
            name += ".cvimodel"
        self.log.append_line(f"$ сборка cvimodel: {name}")
        self._run(
            pipeline.build_cvimodel, onnx, ds.calib_dir, name, tpu,
            ds.model_dir / "build", imgsz=self.imgsz.value(),
            on_done=self._cv_done)

    def _cv_done(self, path: Path) -> None:
        self.main.last_cvimodel = path
        self.log.append_line(f"готово: {path}")
        ds = self.dataset
        if ds is not None:
            names = path.with_suffix(".names")
            try:
                names.write_text("\n".join(ds.class_names) + "\n", encoding="utf-8")
                self.log.append_line(f"имена классов: {names.name}")
            except OSError as exc:
                self.log.append_line(f"не удалось записать {names.name}: {exc}")
        self.refresh()
        if QMessageBox.question(self, "Модель собрана",
                                "cvimodel готов. Перейти к заливке на плату?") \
                == QMessageBox.StandardButton.Yes:
            self.main.go_to_board(path)

    # ---------- фон ----------

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
            QMessageBox.critical(self, "Ошибка", str(exc))

    def _worker_failed(self, message: str) -> None:
        self.log.append_line(f"ОШИБКА: {message}")
        QMessageBox.critical(self, "Не получилось", message)

    def _worker_finished(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        self._update_enabled()

    def _cancel(self) -> None:
        if self._worker is not None:
            self.log.append_line("отмена…")
            self._worker.cancel()

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(3000)
