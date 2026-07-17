"""
labelme/yolo11_settings.py
Dialog "Réglages YOLO11" — panel dynamique progressif.
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

_SETTINGS_PATH = Path(__file__).parent.parent / "yolo11" / "ui_settings.json"

BUILTIN_PRESETS: dict[str, dict] = {
    "Rapide — YOLO body": {
        "tasks": {"boxes": True, "seg": False, "skeletons": True},
        "mode": "yolo",
        "skeleton": "openpose",
        "groups": {"body": True, "feet": False, "face": False, "hands": False},
        "min_conf": 0.3, "nms_iou": 0.5, "fps": 1.0, "rotation": 0,
    },
    "Standard — DWPose body+mains": {
        "tasks": {"boxes": True, "seg": False, "skeletons": True},
        "mode": "dwpose",
        "skeleton": "openpose",
        "groups": {"body": True, "feet": False, "face": False, "hands": True},
        "min_conf": 0.3, "nms_iou": 0.5, "fps": 1.0, "rotation": 0,
    },
    "Full — DWPose tout": {
        "tasks": {"boxes": True, "seg": True, "skeletons": True},
        "mode": "dwpose",
        "skeleton": "openpose",
        "groups": {"body": True, "feet": True, "face": True, "hands": True},
        "min_conf": 0.25, "nms_iou": 0.5, "fps": 1.0, "rotation": 0,
    },
}

DEFAULT_SETTINGS: dict = {
    "active_preset": "Standard — DWPose body+mains",
    "custom_presets": {},
    "current": BUILTIN_PRESETS["Standard — DWPose body+mains"].copy(),
}


def load_ui_settings() -> dict:
    if _SETTINGS_PATH.exists():
        try:
            with open(_SETTINGS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            for k, v in DEFAULT_SETTINGS.items():
                data.setdefault(k, v)
            return data
        except (json.JSONDecodeError, OSError):
            pass
    return json.loads(json.dumps(DEFAULT_SETTINGS))


def save_ui_settings(settings: dict) -> None:
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


def get_current_settings() -> dict:
    return load_ui_settings().get("current", DEFAULT_SETTINGS["current"])


class Yolo11SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Réglages YOLO11")
        self.setMinimumWidth(380)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._settings = load_ui_settings()
        self._all_presets = {**BUILTIN_PRESETS, **self._settings.get("custom_presets", {})}
        self._build_ui()
        self._load_from_settings()

    def _build_ui(self):
        self._main = QtWidgets.QVBoxLayout(self)
        self._main.setSpacing(8)

        # ── Presets ──────────────────────────────────────────────
        row_preset = QtWidgets.QHBoxLayout()
        self._preset_combo = QtWidgets.QComboBox()
        self._preset_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._refresh_preset_combo()
        self._preset_combo.currentTextChanged.connect(self._on_preset_selected)

        btn_save = QtWidgets.QPushButton("💾")
        btn_save.setFixedWidth(30)
        btn_save.setToolTip("Enregistrer comme preset")
        btn_save.clicked.connect(self._on_save_preset)

        self._btn_del = QtWidgets.QPushButton("🗑")
        self._btn_del.setFixedWidth(30)
        self._btn_del.setToolTip("Supprimer ce preset")
        self._btn_del.clicked.connect(self._on_delete_preset)

        row_preset.addWidget(QtWidgets.QLabel("Preset :"))
        row_preset.addWidget(self._preset_combo)
        row_preset.addWidget(btn_save)
        row_preset.addWidget(self._btn_del)
        self._main.addLayout(row_preset)

        self._main.addWidget(_hline())

        # ── Tâches ───────────────────────────────────────────────
        self._main.addWidget(QtWidgets.QLabel("<b>Tâches</b>"))
        row_tasks = QtWidgets.QHBoxLayout()
        self._chk_boxes = QtWidgets.QCheckBox("Boxes")
        self._chk_seg   = QtWidgets.QCheckBox("Segmentation")
        self._chk_skel  = QtWidgets.QCheckBox("Skeletons")
        self._chk_skel.stateChanged.connect(self._refresh_visibility)
        row_tasks.addWidget(self._chk_boxes)
        row_tasks.addWidget(self._chk_seg)
        row_tasks.addWidget(self._chk_skel)
        row_tasks.addStretch()
        self._main.addLayout(row_tasks)

        # ── Mode (visible si Skeletons coché) ────────────────────
        self._w_mode = QtWidgets.QWidget()
        row_mode = QtWidgets.QHBoxLayout(self._w_mode)
        row_mode.setContentsMargins(0, 0, 0, 0)
        row_mode.addWidget(QtWidgets.QLabel("Mode :"))
        self._radio_yolo   = QtWidgets.QRadioButton("YOLO  (17 pts)")
        self._radio_dwpose = QtWidgets.QRadioButton("DWPose  (133/134 pts)")
        self._radio_yolo.toggled.connect(self._refresh_visibility)
        row_mode.addWidget(self._radio_yolo)
        row_mode.addWidget(self._radio_dwpose)
        row_mode.addStretch()
        self._main.addWidget(self._w_mode)

        # ── Format (visible si DWPose) ────────────────────────────
        self._w_fmt = QtWidgets.QWidget()
        row_fmt = QtWidgets.QHBoxLayout(self._w_fmt)
        row_fmt.setContentsMargins(0, 0, 0, 0)
        row_fmt.addWidget(QtWidgets.QLabel("Format :"))
        self._radio_openpose = QtWidgets.QRadioButton("OpenPose  (134 pts)")
        self._radio_coco     = QtWidgets.QRadioButton("COCO  (133 pts)")
        row_fmt.addWidget(self._radio_openpose)
        row_fmt.addWidget(self._radio_coco)
        row_fmt.addStretch()
        self._main.addWidget(self._w_fmt)

        # ── Groupes (visible si DWPose) ───────────────────────────
        self._w_grp = QtWidgets.QWidget()
        row_grp = QtWidgets.QHBoxLayout(self._w_grp)
        row_grp.setContentsMargins(0, 0, 0, 0)
        row_grp.addWidget(QtWidgets.QLabel("Groupes :"))
        self._chk_body  = QtWidgets.QCheckBox("body")
        self._chk_feet  = QtWidgets.QCheckBox("feet")
        self._chk_face  = QtWidgets.QCheckBox("face")
        self._chk_hands = QtWidgets.QCheckBox("hands")
        for c in (self._chk_body, self._chk_feet, self._chk_face, self._chk_hands):
            row_grp.addWidget(c)
        row_grp.addStretch()
        self._main.addWidget(self._w_grp)

        self._main.addWidget(_hline())

        # ── Paramètres numériques ─────────────────────────────────
        self._main.addWidget(QtWidgets.QLabel("<b>Paramètres</b>"))
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self._spin_conf = _spin(0.01, 1.0, 0.05, 2)
        self._spin_iou  = _spin(0.01, 1.0, 0.05, 2)
        self._spin_fps  = _spin(0.1, 60.0, 0.5, 1)

        form.addRow("Confiance min :", self._spin_conf)
        form.addRow("NMS IoU :", self._spin_iou)
        form.addRow("FPS :", self._spin_fps)
        self._main.addLayout(form)

        # ── Rotation avant inférence ──────────────────────────────
        self._main.addWidget(_hline())
        self._main.addWidget(QtWidgets.QLabel("<b>Rotation avant inférence</b>"))
        row_rot = QtWidgets.QHBoxLayout()
        self._radio_rot = {}
        self._rot_group = QtWidgets.QButtonGroup(self)
        rot_labels = {0: "0°", 90: "90°R", 180: "180°", 270: "90°L"}
        for angle in [0, 270, 180, 90]:
            rb = QtWidgets.QRadioButton(rot_labels[angle])
            self._radio_rot[angle] = rb
            self._rot_group.addButton(rb)
            row_rot.addWidget(rb)
        self._radio_rot[0].setChecked(True)
        row_rot.addStretch()
        self._main.addLayout(row_rot)

        self._main.addWidget(_hline())

        # ── OK / Annuler ──────────────────────────────────────────
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        self._main.addWidget(btn_box)

    # ── Chargement ────────────────────────────────────────────────

    def _load_from_settings(self):
        self._apply(self._settings.get("current", DEFAULT_SETTINGS["current"]))
        active = self._settings.get("active_preset", "")
        idx = self._preset_combo.findText(active)
        if idx >= 0:
            self._preset_combo.blockSignals(True)
            self._preset_combo.setCurrentIndex(idx)
            self._preset_combo.blockSignals(False)
        self._update_delete_btn()

    def _apply(self, v: dict):
        tasks = v.get("tasks", {})
        self._chk_boxes.setChecked(tasks.get("boxes", True))
        self._chk_seg.setChecked(tasks.get("seg", False))
        self._chk_skel.setChecked(tasks.get("skeletons", True))

        self._radio_yolo.setChecked(v.get("mode", "yolo") == "yolo")
        self._radio_dwpose.setChecked(v.get("mode", "yolo") == "dwpose")
        self._radio_openpose.setChecked(v.get("skeleton", "openpose") == "openpose")
        self._radio_coco.setChecked(v.get("skeleton", "openpose") == "coco")

        grp = v.get("groups", {})
        self._chk_body.setChecked(grp.get("body", True))
        self._chk_feet.setChecked(grp.get("feet", False))
        self._chk_face.setChecked(grp.get("face", False))
        self._chk_hands.setChecked(grp.get("hands", False))

        self._spin_conf.setValue(v.get("min_conf", 0.3))
        self._spin_iou.setValue(v.get("nms_iou", 0.5))
        self._spin_fps.setValue(v.get("fps", 1.0))

        rot = int(v.get("rotation", 0)) % 360
        for angle, rb in self._radio_rot.items():
            rb.setChecked(angle == rot)

        self._refresh_visibility()

    def _collect(self) -> dict:
        return {
            "tasks": {
                "boxes":     self._chk_boxes.isChecked(),
                "seg":       self._chk_seg.isChecked(),
                "skeletons": self._chk_skel.isChecked(),
            },
            "mode":     "yolo" if self._radio_yolo.isChecked() else "dwpose",
            "skeleton": "openpose" if self._radio_openpose.isChecked() else "coco",
            "groups": {
                "body":  self._chk_body.isChecked(),
                "feet":  self._chk_feet.isChecked(),
                "face":  self._chk_face.isChecked(),
                "hands": self._chk_hands.isChecked(),
            },
            "min_conf":  self._spin_conf.value(),
            "nms_iou":   self._spin_iou.value(),
            "fps":       self._spin_fps.value(),
            "rotation":  next(a for a, rb in self._radio_rot.items() if rb.isChecked()),
        }

    # ── Visibilité dynamique ──────────────────────────────────────

    def _refresh_visibility(self):
        skel_on  = self._chk_skel.isChecked()
        dwpose   = self._radio_dwpose.isChecked()

        self._w_mode.setVisible(skel_on)
        self._w_fmt.setVisible(skel_on and dwpose)
        self._w_grp.setVisible(skel_on and dwpose)

        self.adjustSize()

    # ── Presets ───────────────────────────────────────────────────

    def _on_preset_selected(self, name: str):
        v = self._all_presets.get(name)
        if v:
            self._apply(v)
        self._update_delete_btn()

    def _on_save_preset(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Enregistrer le preset", "Nom :")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in BUILTIN_PRESETS:
            QtWidgets.QMessageBox.warning(self, "Nom réservé", f"'{name}' est un preset intégré.")
            return
        v = self._collect()
        self._settings.setdefault("custom_presets", {})[name] = v
        self._all_presets[name] = v
        self._refresh_preset_combo(select=name)
        self._update_delete_btn()

    def _on_delete_preset(self):
        name = self._preset_combo.currentText()
        if name in BUILTIN_PRESETS:
            return
        if QtWidgets.QMessageBox.question(
            self, "Supprimer", f"Supprimer '{name}' ?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        ) != QtWidgets.QMessageBox.Yes:
            return
        self._settings.get("custom_presets", {}).pop(name, None)
        self._all_presets.pop(name, None)
        self._refresh_preset_combo()
        self._update_delete_btn()

    def _on_accept(self):
        self._settings["current"] = self._collect()
        self._settings["active_preset"] = self._preset_combo.currentText()
        save_ui_settings(self._settings)
        self.accept()

    def _refresh_preset_combo(self, select=None):
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        for n in list(BUILTIN_PRESETS) + list(self._settings.get("custom_presets", {})):
            self._preset_combo.addItem(n)
        if select:
            idx = self._preset_combo.findText(select)
            if idx >= 0:
                self._preset_combo.setCurrentIndex(idx)
        self._preset_combo.blockSignals(False)

    def _update_delete_btn(self):
        self._btn_del.setEnabled(
            self._preset_combo.currentText() not in BUILTIN_PRESETS
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hline():
    f = QtWidgets.QFrame()
    f.setFrameShape(QtWidgets.QFrame.HLine)
    f.setFrameShadow(QtWidgets.QFrame.Sunken)
    return f

def _spin(mn, mx, step, dec):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(mn, mx)
    s.setSingleStep(step)
    s.setDecimals(dec)
    return s
