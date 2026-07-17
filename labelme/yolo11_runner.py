"""
labelme/yolo11_runner.py
Lancement direct de yolo11.run() depuis LabelMe (pas de subprocess).
"""
from __future__ import annotations
from pathlib import Path

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt


# ---------------------------------------------------------------------------
# Thread d'exécution
# ---------------------------------------------------------------------------

class _Yolo11Thread(QtCore.QThread):
    log_line  = QtCore.pyqtSignal(str)
    finished_ok  = QtCore.pyqtSignal(object)   # résultat run()
    finished_err = QtCore.pyqtSignal(str)       # message d'erreur

    def __init__(self, path: str, settings: dict):
        super().__init__()
        self._path     = path
        self._settings = settings

    def run(self):
        import traceback
        try:
            from yolo11.main import run
            from labelme.yolo11_settings import get_current_settings

            s = self._settings
            # Mapping nom UI -> nom attendu par inference.py
            _TASK_NAME_MAP = {"seg": "segmentation"}
            tasks    = [_TASK_NAME_MAP.get(k, k) for k, v in s.get("tasks", {}).items() if v]
            has_skel = s.get("tasks", {}).get("skeletons", False)

            # Si skeletons pas coché → on ignore mode/groups/skeleton
            if has_skel:
                mode   = s.get("mode", "yolo")
                groups = [k for k, v in s.get("groups", {}).items() if v] if mode == "dwpose" else None
                skeleton = s.get("skeleton", "openpose") if mode == "dwpose" else None
            else:
                mode     = "yolo"
                groups   = None
                skeleton = None

            self.log_line.emit(f"[YOLO11] path     : {self._path}")
            self.log_line.emit(f"[YOLO11] tasks    : {tasks}")
            self.log_line.emit(f"[YOLO11] mode     : {mode}")
            if mode == "dwpose":
                self.log_line.emit(f"[YOLO11] skeleton : {skeleton}")
                self.log_line.emit(f"[YOLO11] groups   : {groups}")

            rotation = int(s.get("rotation", 0)) % 360
            if rotation:
                self.log_line.emit(f"[YOLO11] rotation : {rotation}°")

            res = run(
                self._path,
                tasks=tasks,
                mode=mode,
                skeleton=skeleton,
                groups=groups,
                min_conf=s.get("min_conf", 0.3),
                nms_iou=s.get("nms_iou", 0.5),
                target_fps=s.get("fps", 1.0),
                rotation=rotation,
            )
            self.finished_ok.emit(res)
        except Exception:
            self.finished_err.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Dialog de progression
# ---------------------------------------------------------------------------

class _RunDialog(QtWidgets.QDialog):
    def __init__(self, path: str, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO11 — Traitement en cours…")
        self.setMinimumSize(560, 260)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self._input_path = path
        self._success    = False

        layout = QtWidgets.QVBoxLayout(self)

        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        layout.addWidget(self._log)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 0)
        layout.addWidget(self._progress)

        self._btn = QtWidgets.QPushButton("Annuler")
        self._btn.clicked.connect(self._on_cancel)
        layout.addWidget(self._btn, alignment=Qt.AlignRight)

        self._thread = _Yolo11Thread(path, settings)
        self._thread.log_line.connect(self._log.appendPlainText)
        self._thread.finished_ok.connect(self._on_ok)
        self._thread.finished_err.connect(self._on_err)
        self._thread.start()

    def _on_cancel(self):
        if self._thread.isRunning():
            self._thread.terminate()
        self.reject()

    def _on_ok(self, _res):
        self._success = True
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._log.appendPlainText("\n✅ Terminé.")
        self._btn.setText("Fermer")
        self._btn.clicked.disconnect()
        self._btn.clicked.connect(self.accept)
        # Fermeture automatique après 1 seconde
        QtCore.QTimer.singleShot(1000, self.accept)

    def _on_err(self, tb: str):
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._log.appendPlainText(f"\n❌ Erreur :\n{tb}")
        self._btn.setText("Fermer")

    def processed_dir(self) -> Path | None:
        p = Path(self._input_path).resolve()
        # processed/ est toujours créé dans le parent (que ce soit fichier ou dossier)
        d = p.parent / "processed"
        self._log.appendPlainText(f"[DEBUG] Recherche processed/ dans : {d}")
        return d if d.exists() else None


# ---------------------------------------------------------------------------
# Point d'entrée appelé depuis app.py
# ---------------------------------------------------------------------------

def run_yolo11(parent_window, pick_dir: bool = False):
    """
    pick_dir=False → sélection fichier unique
    pick_dir=True  → sélection dossier
    """
    if pick_dir:
        input_path = QtWidgets.QFileDialog.getExistingDirectory(
            parent_window, "YOLO11 — Choisir un dossier à traiter"
        )
    else:
        input_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent_window,
            "YOLO11 — Choisir un fichier à traiter",
            "",
            "Images & vidéos (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov);;Tous (*)",
        )

    if not input_path:
        return

    from labelme.yolo11_settings import get_current_settings
    settings = get_current_settings()
    dlg = _RunDialog(str(Path(input_path)), settings, parent=parent_window)
    result = dlg.exec_()

    if result != QtWidgets.QDialog.Accepted:
        return

    processed = dlg.processed_dir()
    if processed and processed.exists():
        parent_window.importDirImages(str(processed), load=False)
        if parent_window.fileListWidget.count() > 0:
            parent_window.fileListWidget.setCurrentRow(0)
            parent_window.fileSelectionChanged()
    else:
        QtWidgets.QMessageBox.information(
            parent_window, "YOLO11",
            "Traitement terminé.\nDossier processed/ introuvable — vérifiez la sortie.",
        )
