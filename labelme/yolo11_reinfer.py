"""
labelme/yolo11_reinfer.py
Relance DWPose sur l'image courante en utilisant les boxes en mémoire (canvas).
Bypasse saveFile() et les modèles inutiles (boxes, skeletons).

Workflow :
  1. Lit les boxes directement depuis canvas.shapes (pas de saveFile())
  2. Si l'image n'est PAS dans un dossier 'processed/', crée processed/ à côté
     des originaux et y copie l'image — sans jamais créer processed/processed/
  3. Écrit un JSON minimal avec les boxes + autres shapes non-skeleton
  4. Charge uniquement DWPose (+ segmentation si cochée) — pas de boxes/skeleton
  5. Lance reinfer_dwpose_from_labelme_boxes
  6. Recharge l'image depuis l'emplacement cible
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PyQt5 import QtWidgets

from labelme.yolo11_settings import get_current_settings


def reinfer_from_boxes(parent_window) -> None:
    if not getattr(parent_window, "imagePath", None):
        QtWidgets.QMessageBox.warning(
            parent_window, "Réinférence", "Aucune image ouverte."
        )
        return

    # ── 1. Lire les boxes depuis le canvas (en mémoire, pas de saveFile()) ──
    boxes_shapes = []
    other_shapes = []
    for shape in parent_window.canvas.shapes:
        if shape.shape_type == "rectangle":
            boxes_shapes.append(shape)
        elif shape.shape_type != "skeleton":
            other_shapes.append(shape)
        # skeletons ignorés → remplacés par la réinférence

    if not boxes_shapes:
        QtWidgets.QMessageBox.information(
            parent_window, "Réinférence",
            "Aucune box rectangle trouvée dans l'image courante.\n"
            "Dessinez des boxes autour des personnes avant de relancer."
        )
        return

    # ── 2. Déterminer l'emplacement cible ─────────────────────────────────
    image_path = Path(parent_window.imagePath)

    if image_path.parent.name == "processed":
        # Cas normal — image déjà dans processed/ → on reste là, on écrase
        out_image_path = image_path
        out_json_path  = image_path.with_suffix(".json")
    else:
        # Cas zigoto — image hors processed/ → créer processed/ à côté des originaux
        # (jamais processed/processed/ : on est forcément hors processed/ ici)
        processed_dir  = image_path.parent / "processed"
        processed_dir.mkdir(exist_ok=True)
        stem           = f"{image_path.stem}_annotated"
        out_image_path = processed_dir / f"{stem}{image_path.suffix}"
        out_json_path  = processed_dir / f"{stem}.json"
        if not out_image_path.exists():
            shutil.copy2(str(image_path), str(out_image_path))

    # ── 3. Écrire un JSON minimal avec boxes + autres shapes ──────────────
    w = parent_window.image.width()
    h = parent_window.image.height()

    def _shape_to_dict(shape):
        return {
            "label":        shape.label,
            "group_id":     shape.group_id,
            "description":  getattr(shape, "description", ""),
            "shape_type":   shape.shape_type,
            "flags":        getattr(shape, "flags", {}),
            "mask":         None,
            "points":       [[p.x(), p.y()] for p in shape.points],
            "skeleton_name": getattr(shape, "skeleton_name", None),
        }

    data = {
        "version":     "5.9.1",
        "flags":       {},
        "shapes":      [_shape_to_dict(s) for s in boxes_shapes + other_shapes],
        "imagePath":   out_image_path.name,
        "imageData":   None,
        "imageHeight": h,
        "imageWidth":  w,
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # ── 4. Construire le cfg depuis les réglages UI ───────────────────────
    from yolo11.config import load_config
    from yolo11.inference import reinfer_dwpose_from_labelme_boxes

    settings    = get_current_settings()
    cfg         = load_config()
    mode        = settings.get("mode", "dwpose")
    skeleton    = settings.get("skeleton", "openpose")
    seg_enabled = settings.get("tasks", {}).get("seg", False)

    cfg["pose_backend"]  = mode
    cfg["skeleton_name"] = "openpose_fullbody" if skeleton == "openpose" else "coco_fullbody"
    cfg["seg_enabled"]   = seg_enabled
    cfg["rotation"]      = settings.get("rotation", 0)
    cfg["dwpose_groups"] = settings.get("groups", {})
    cfg["min_conf"]      = settings.get("min_conf", 0.3)
    cfg["nms_iou"]       = settings.get("nms_iou", 0.5)

    # ── 5. Charger uniquement les modèles nécessaires ─────────────────────
    # Réinférence DWPose = pas besoin du modèle boxes ni skeleton YOLO
    # DWPose est chargé lazily par get_dwpose() dans reinfer_dwpose_from_labelme_boxes
    # On ne charge la segmentation que si cochée
    try:
        if seg_enabled:
            from yolo11.model import load_model
            load_model(["segmentation"])
    except Exception as e:
        QtWidgets.QMessageBox.critical(
            parent_window, "Réinférence",
            f"Erreur chargement modèle segmentation :\n{e}"
        )
        return

    # ── 6. Lancer la réinférence ──────────────────────────────────────────
    try:
        reinfer_dwpose_from_labelme_boxes(
            str(out_image_path),
            str(out_json_path),
            cfg=cfg,
        )
    except Exception as e:
        QtWidgets.QMessageBox.critical(
            parent_window, "Réinférence", f"Erreur DWPose :\n{e}"
        )
        return

    # ── 7. Recharger depuis l'emplacement cible ───────────────────────────
    parent_window.loadFile(str(out_image_path))

    msg = (f"Réinférence terminée depuis {len(boxes_shapes)} box(es).\n"
           f"Skeletons mis à jour.")
    if seg_enabled:
        msg += "\nMasks (segmentation) régénérés."
    if out_image_path != image_path:
        msg += f"\n\nFichiers dans :\n{out_image_path.parent}"
    QtWidgets.QMessageBox.information(parent_window, "Réinférence", msg)
