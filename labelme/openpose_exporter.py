"""
labelme/openpose_exporter.py
Export dataset OpenPose depuis les shapes skeleton cochées.

Génère en une seule action :
  PNG-{stem}.png   ← rendu OpenPose fond noir
  PNG-{stem}.json  ← JSON LabelMe avec uniquement les skeletons cochés
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

# ---------------------------------------------------------------------------
# Palette arc-en-ciel standard OpenPose/ControlNet (RGB)
# ---------------------------------------------------------------------------
_RAINBOW = [
    (255, 0,   0),   (255, 85,  0),   (255, 170, 0),   (255, 255, 0),
    (170, 255, 0),   (85,  255, 0),   (0,   255, 0),   (0,   255, 85),
    (0,   255, 170), (0,   255, 255), (0,   170, 255), (0,   85,  255),
    (0,   0,   255), (85,  0,   255), (170, 0,   255), (255, 0,   255),
    (255, 0,   170), (255, 0,   85),
]
_GROUP_PT_COLOR = {
    "feet":       (255, 170, 0),
    "left_hand":  (0,   255, 0),
    "right_hand": (0,   0,   255),
    "face":       (255, 255, 255),
}
_FEET_LEFT  = {18, 19, 20}
_FEET_RIGHT = {21, 22, 23}
_HAND_LEFT  = set(range(92, 113))
_HAND_RIGHT = set(range(113, 134))

# Silhouette (mask polygon) dessinée sous le skeleton — purement visuelle,
# jamais reportée dans le JSON exporté.
_MASK_FILL_COLOR_BGR = (90, 90, 90)
_MASK_ALPHA = 0.35

def _bgr(rgb): return (rgb[2], rgb[1], rgb[0])
def _rainbow(i): return _RAINBOW[i % len(_RAINBOW)]

_SKELETON_CACHE: dict[str, dict] = {}

def _load_skeleton(name: str) -> dict:
    if name not in _SKELETON_CACHE:
        spec_path = Path(__file__).parent.parent / "coco_specs" / f"{name}_skeleton.json"
        try:
            with open(spec_path, encoding="utf-8") as f:
                _SKELETON_CACHE[name] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _SKELETON_CACHE[name] = {}
    return _SKELETON_CACHE[name]

def _valid_pts(points) -> dict[int, tuple[int, int]]:
    return {
        i: (int(pt.x()), int(pt.y()))
        for i, pt in enumerate(points)
        if not (int(pt.x()) == 0 and int(pt.y()) == 0)
    }

def _group_complete(pts, id_set, threshold=1.0):
    return sum(1 for i in id_set if i in pts) >= len(id_set) * threshold

# ---------------------------------------------------------------------------
# Rendu PNG OpenPose
# ---------------------------------------------------------------------------

def render_openpose_png(shapes: list, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # --- Silhouettes (masks) en dessous, semi-transparentes -------------
    # Un mask n'est dessiné que s'il existe un skeleton de même label
    # dans la liste passée (= skeleton coché). Jamais écrit dans le JSON.
    # NB : on matche par "label" ("person_0", etc.), pas group_id —
    # le group_id n'est en pratique renseigné que sur le skeleton, jamais
    # sur le mask/box généré par la pipeline d'inférence.
    skeleton_labels = {
        getattr(s, "label", None)
        for s in shapes if getattr(s, "shape_type", None) == "skeleton"
    }
    for shape in shapes:
        if getattr(shape, "shape_type", None) != "polygon":
            continue
        if getattr(shape, "label", None) not in skeleton_labels:
            continue
        pts = np.array(
            [[int(pt.x()), int(pt.y())] for pt in shape.points], dtype=np.int32
        )
        if len(pts) < 3:
            continue
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], _MASK_FILL_COLOR_BGR)
        cv2.addWeighted(overlay, _MASK_ALPHA, canvas, 1 - _MASK_ALPHA, 0, canvas)

    for shape in shapes:
        if shape.shape_type != "skeleton":
            continue

        skeleton_name = getattr(shape, "skeleton_name", "human")
        spec = _load_skeleton(skeleton_name)
        pts = _valid_pts(shape.points)
        connections = spec.get("connections", {})

        # Body — arc-en-ciel par connexion
        body_conns = connections.get("body", [])
        kp_color: dict[int, tuple] = {}
        for ci, conn in enumerate(body_conns):
            f, t = conn["from"], conn["to"]
            if f not in kp_color: kp_color[f] = _rainbow(ci)
            if t not in kp_color: kp_color[t] = _rainbow(ci)
            if f in pts and t in pts:
                cv2.line(canvas, pts[f], pts[t], _bgr(_rainbow(ci)), 2, cv2.LINE_AA)
        for idx, (x, y) in pts.items():
            if idx in kp_color:
                cv2.circle(canvas, (x, y), 4, _bgr(kp_color[idx]), -1, cv2.LINE_AA)
                cv2.circle(canvas, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)

        # Feet — connexions si sous-groupe complet
        color_feet = _bgr(_GROUP_PT_COLOR["feet"])
        left_ok  = _group_complete(pts, _FEET_LEFT)
        right_ok = _group_complete(pts, _FEET_RIGHT)
        for conn in connections.get("feet", []):
            f, t = conn["from"], conn["to"]
            is_left = f in _FEET_LEFT or t in _FEET_LEFT
            if (left_ok if is_left else right_ok) and f in pts and t in pts:
                cv2.line(canvas, pts[f], pts[t], color_feet, 2, cv2.LINE_AA)
        for idx in _FEET_LEFT | _FEET_RIGHT:
            if idx in pts:
                cv2.circle(canvas, pts[idx], 3, color_feet, -1, cv2.LINE_AA)
                cv2.circle(canvas, pts[idx], 3, (255, 255, 255), 1, cv2.LINE_AA)

        # Hands — connexions si main complète à 90%
        for hand_ids, gkey, ckey in [
            (_HAND_LEFT,  "left_hand",  "left_hand"),
            (_HAND_RIGHT, "right_hand", "right_hand"),
        ]:
            color_hand = _bgr(_GROUP_PT_COLOR[gkey])
            hand_ok = _group_complete(pts, hand_ids, threshold=0.9)
            if hand_ok:
                for conn in connections.get(ckey, []):
                    f, t = conn["from"], conn["to"]
                    if f in pts and t in pts:
                        cv2.line(canvas, pts[f], pts[t], color_hand, 1, cv2.LINE_AA)
            for idx in hand_ids:
                if idx in pts:
                    cv2.circle(canvas, pts[idx], 2, color_hand, -1, cv2.LINE_AA)

        # Face — points uniquement
        face_start = spec.get("groups", {}).get("face", {}).get("start", 24)
        face_end   = spec.get("groups", {}).get("face", {}).get("end",   91)
        color_face = _bgr(_GROUP_PT_COLOR["face"])
        for idx in range(face_start, face_end + 1):
            if idx in pts:
                cv2.circle(canvas, pts[idx], 2, color_face, -1, cv2.LINE_AA)

    return canvas

# ---------------------------------------------------------------------------
# Export JSON LabelMe — skeletons cochés uniquement
# ---------------------------------------------------------------------------

def _shape_to_dict(shape) -> dict:
    """Convertit une Shape LabelMe en dict JSON."""
    points = []
    for i, pt in enumerate(shape.points):
        points.append([float(i), float(pt.x()), float(pt.y()), 1.0])
    return {
        "label":        shape.label,
        "group_id":     getattr(shape, "group_id", None),
        "description":  getattr(shape, "description", ""),
        "shape_type":   shape.shape_type,
        "flags":        getattr(shape, "flags", {}),
        "mask":         None,
        "points":       points,
        "skeleton_name": getattr(shape, "skeleton_name", None),
    }

def export_dataset(parent_window) -> None:
    """
    Génère PNG-{stem}.png + PNG-{stem}.json
    depuis les skeletons cochés dans Polygon Labels.
    """
    if not getattr(parent_window, "imagePath", None):
        QtWidgets.QMessageBox.warning(
            parent_window, "Export dataset OpenPose", "Aucune image ouverte."
        )
        return

    skeleton_shapes = []
    mask_shapes = []
    for item in parent_window.labelList:
        shape = item.shape()
        if shape.shape_type == "polygon":
            mask_shapes.append(shape)
        elif item.checkState() == Qt.Checked and shape.shape_type == "skeleton":
            skeleton_shapes.append(shape)

    if not skeleton_shapes:
        QtWidgets.QMessageBox.information(
            parent_window, "Export dataset OpenPose",
            "Aucun skeleton coché dans le panel Polygon Labels."
        )
        return

    image_path = Path(parent_window.imagePath)
    stem = f"PNG-{image_path.stem}"
    out_dir = image_path.parent

    png_path  = out_dir / f"{stem}.png"
    json_path = out_dir / f"{stem}.json"

    # PNG OpenPose (silhouettes mask sous les skeletons, non reportées en JSON)
    w = parent_window.image.width()
    h = parent_window.image.height()
    canvas = render_openpose_png(skeleton_shapes + mask_shapes, w, h)
    cv2.imwrite(str(png_path), canvas)

    # JSON LabelMe
    data = {
        "version":     "5.9.1",
        "flags":       {},
        "shapes":      [_shape_to_dict(s) for s in skeleton_shapes],
        "imagePath":   f"{stem}.png",
        "imageData":   None,
        "imageHeight": h,
        "imageWidth":  w,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    QtWidgets.QMessageBox.information(
        parent_window, "Export dataset OpenPose",
        f"Fichiers générés :\n{png_path.name}\n{json_path.name}"
    )


# Alias pour compatibilité avec app.py existant
def export_openpose_png(parent_window) -> None:
    export_dataset(parent_window)

# ---------------------------------------------------------------------------
# Export batch — dossier entier
# ---------------------------------------------------------------------------

def export_dataset_batch(parent_window) -> None:
    """
    Batch : génère PNG-{stem}.png + PNG-{stem}.json pour chaque image du dossier.
    Prend tous les skeletons de chaque JSON sans distinction.
    """
    folder = QtWidgets.QFileDialog.getExistingDirectory(
        parent_window, "Choisir le dossier à traiter"
    )
    if not folder:
        return

    folder_path = Path(folder)
    json_files  = sorted(folder_path.glob("*.json"))
    # Exclure les JSON déjà générés (préfixe PNG-)
    json_files  = [j for j in json_files if not j.stem.startswith("PNG-")]

    if not json_files:
        QtWidgets.QMessageBox.warning(
            parent_window, "Export batch", "Aucun JSON trouvé dans ce dossier."
        )
        return

    from PyQt5.QtCore import QPointF

    count = 0
    errors = []

    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            w = data.get("imageWidth",  0)
            h = data.get("imageHeight", 0)
            if not w or not h:
                errors.append(json_file.name)
                continue

            # Reconstruire les shapes skeleton + masks depuis le JSON
            skeleton_shapes = []
            mask_shapes = []
            for shape_data in data.get("shapes", []):
                st = shape_data.get("shape_type")
                if st not in ("skeleton", "polygon"):
                    continue

                raw_pts = shape_data.get("points", [])
                points  = []
                for pt in raw_pts:
                    # Format [id, x, y, conf] ou [x, y]
                    if len(pt) == 4:
                        points.append(QPointF(pt[1], pt[2]))
                    elif len(pt) == 2:
                        points.append(QPointF(pt[0], pt[1]))

                class _FakeShape:
                    pass
                s = _FakeShape()
                s.shape_type    = st
                s.points        = points
                s.label         = shape_data.get("label", "person")
                s.group_id      = shape_data.get("group_id", None)
                s.description   = shape_data.get("description", "")
                s.flags         = shape_data.get("flags", {})
                s.skeleton_name = shape_data.get("skeleton_name") or (
                    "openpose_fullbody" if len(points) == 134 else
                    "coco_fullbody"     if len(points) == 133 else
                    "human"
                )
                if st == "skeleton":
                    skeleton_shapes.append(s)
                else:
                    mask_shapes.append(s)

            if not skeleton_shapes:
                continue

            stem      = f"PNG-{json_file.stem}"
            png_path  = folder_path / f"{stem}.png"
            json_path = folder_path / f"{stem}.json"

            # PNG (silhouettes mask sous les skeletons, non reportées en JSON)
            canvas = render_openpose_png(skeleton_shapes + mask_shapes, w, h)
            cv2.imwrite(str(png_path), canvas)

            # JSON
            out_data = {
                "version":     "5.9.1",
                "flags":       {},
                "shapes":      [_shape_to_dict(s) for s in skeleton_shapes],
                "imagePath":   f"{stem}.png",
                "imageData":   None,
                "imageHeight": h,
                "imageWidth":  w,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=2, ensure_ascii=False)

            count += 1

        except Exception as e:
            errors.append(f"{json_file.name}: {e}")

    msg = f"{count} image(s) traitée(s) sur {len(json_files)}."
    if errors:
        msg += f"\n\nErreurs :\n" + "\n".join(errors[:10])
    QtWidgets.QMessageBox.information(parent_window, "Export batch terminé", msg)
