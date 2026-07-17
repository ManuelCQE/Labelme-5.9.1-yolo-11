"""
labelme/pose_caption.py
Génère une caption textuelle de pose depuis les skeletons cochés.
Uniquement les skeletons visibles (cochés) dans le Polygon Labels panel.

Produit des descriptions en anglais compatibles ComfyUI/ControlNet.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


# ---------------------------------------------------------------------------
# Helpers géométriques
# ---------------------------------------------------------------------------

def _pt(points, idx) -> tuple[float, float] | None:
    """Retourne (x, y) du keypoint idx, ou None si à (0,0)."""
    if idx >= len(points):
        return None
    pt = points[idx]
    x, y = pt.x(), pt.y()
    if x == 0 and y == 0:
        return None
    return (x, y)

def _dist(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return math.hypot(b[0] - a[0], b[1] - a[1])

def _midpoint(a, b) -> tuple | None:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

def _angle_deg(a, b) -> float | None:
    """Angle en degrés du vecteur a→b (0° = droite, 90° = bas)."""
    if a is None or b is None:
        return None
    dx, dy = b[0] - a[0], b[1] - a[1]
    return math.degrees(math.atan2(dy, dx))


# ---------------------------------------------------------------------------
# Analyse de pose — format openpose_fullbody (134pts)
# Indices : 0=Nose 1=Neck 2=RShoulder 3=RElbow 4=RWrist
#           5=LShoulder 6=LElbow 7=LWrist 8=RHip 9=RKnee 10=RAnkle
#           11=LHip 12=LKnee 13=LAnkle 14=REye 15=LEye 16=REar 17=LEar
# ---------------------------------------------------------------------------

def _analyze_openpose(points) -> dict:
    """Retourne un dict de propriétés de pose depuis les keypoints."""
    p = lambda i: _pt(points, i)

    nose      = p(0);  neck    = p(1)
    rsho      = p(2);  relbow  = p(3);  rwrist = p(4)
    lsho      = p(5);  lelbow  = p(6);  lwrist = p(7)
    rhip      = p(8);  rknee   = p(9);  rankle = p(10)
    lhip      = p(11); lknee   = p(12); lankle = p(13)
    reye      = p(14); leye    = p(15)
    rear      = p(16); lear    = p(17)

    res = {}

    # ── Orientation caméra ─────────────────────────────────────────────────
    if rsho and lsho:
        sho_width = abs(rsho[0] - lsho[0])
        sho_dist  = _dist(rsho, lsho)
        ratio = sho_width / sho_dist if sho_dist > 0 else 1.0
        if ratio > 0.8:
            res["orientation"] = "facing camera"
        elif ratio < 0.3:
            # Profil — lequel ?
            if rsho and lsho:
                res["orientation"] = "left profile" if rsho[0] > lsho[0] else "right profile"
            else:
                res["orientation"] = "profile"
        else:
            res["orientation"] = "three-quarter view"
    elif rear and not lear:
        res["orientation"] = "right profile"
    elif lear and not rear:
        res["orientation"] = "left profile"

    # ── Dos / face ────────────────────────────────────────────────────────
    if nose is None and neck is not None:
        res["facing"] = "back to camera"

    # ── Position globale (debout / assis / allongé) ───────────────────────
    if neck and rhip and lhip:
        hip_mid = _midpoint(rhip, lhip)
        torso_h = abs(neck[1] - hip_mid[1]) if hip_mid else 0

        if rankle and lankle:
            ankle_mid_y = (rankle[1] + lankle[1]) / 2
            leg_h = abs(hip_mid[1] - ankle_mid_y) if hip_mid else 0

            if torso_h > 0 and leg_h > 0:
                ratio = leg_h / torso_h
                if ratio < 0.4:
                    res["position"] = "lying down"
                elif ratio < 0.8:
                    res["position"] = "sitting"
                else:
                    res["position"] = "standing"
        elif rknee or lknee:
            res["position"] = "sitting"
        else:
            res["position"] = "standing"

    # ── Angle de vue caméra (angle de prise de vue vertical) ─────────────
    if neck and (rhip or lhip):
        hip_mid = _midpoint(rhip, lhip) or rhip or lhip
        dy = hip_mid[1] - neck[1]
        dx = hip_mid[0] - neck[0]
        vertical_ratio = abs(dy) / (abs(dx) + 1e-6)
        if vertical_ratio < 0.5:
            res["camera_angle"] = "extreme low angle" if neck[1] > hip_mid[1] else "extreme high angle"
        elif vertical_ratio < 1.2:
            res["camera_angle"] = "low angle" if neck[1] > hip_mid[1] else "high angle"
        # sinon angle neutre → pas mentionné

    # ── Bras ──────────────────────────────────────────────────────────────
    arms = []
    # Bras droit
    if rsho and relbow and rwrist:
        elbow_up = relbow[1] < rsho[1]
        wrist_up = rwrist[1] < relbow[1]
        if wrist_up and elbow_up:
            arms.append("right arm raised")
        elif rwrist[1] < rsho[1]:
            arms.append("right arm partially raised")
        elif rwrist[0] < lsho[0] if lsho else False:
            arms.append("right arm crossed")
    # Bras gauche
    if lsho and lelbow and lwrist:
        elbow_up = lelbow[1] < lsho[1]
        wrist_up = lwrist[1] < lelbow[1]
        if wrist_up and elbow_up:
            arms.append("left arm raised")
        elif lwrist[1] < lsho[1]:
            arms.append("left arm partially raised")
        elif lsho and lwrist[0] > rsho[0] if rsho else False:
            arms.append("left arm crossed")
    if arms:
        res["arms"] = ", ".join(arms)

    # ── Jambes ────────────────────────────────────────────────────────────
    legs = []
    # Jambe droite
    if rhip and rknee and rankle:
        knee_angle = _angle_deg(rhip, rknee)
        if knee_angle is not None and abs(knee_angle) < 30:
            legs.append("right leg extended")
        elif rknee[0] < lhip[0] if lhip else False:
            legs.append("right leg crossed")

    # Jambe gauche
    if lhip and lknee and lankle:
        knee_angle = _angle_deg(lhip, lknee)
        if knee_angle is not None and abs(knee_angle) < 30:
            legs.append("left leg extended")
        elif lknee[0] > rhip[0] if rhip else False:
            legs.append("left leg crossed")

    # Grand écart
    if rankle and lankle and rhip and lhip:
        ankle_spread = abs(rankle[0] - lankle[0])
        hip_spread   = abs(rhip[0]   - lhip[0])
        if ankle_spread > hip_spread * 2.5:
            legs = ["legs wide apart / split"]

    if legs:
        res["legs"] = ", ".join(legs)

    # ── Appui / poids ─────────────────────────────────────────────────────
    if rhip and lhip and rankle and lankle:
        hip_tilt = rhip[1] - lhip[1]  # positif = hanche droite plus basse
        if abs(hip_tilt) > 15:
            res["weight"] = "weight on right leg" if hip_tilt > 0 else "weight on left leg"

    return res


def _analyze_coco17(points) -> dict:
    """
    Analyse pour human_skeleton (COCO 17pts).
    Indices : 0=nose 1=left_eye 2=right_eye 3=left_ear 4=right_ear
              5=left_shoulder 6=right_shoulder 7=left_elbow 8=right_elbow
              9=left_wrist 10=right_wrist 11=left_hip 12=right_hip
              13=left_knee 14=right_knee 15=left_ankle 16=right_ankle
    """
    p = lambda i: _pt(points, i)

    nose  = p(0)
    lsho  = p(5);  rsho   = p(6)
    lelbow= p(7);  relbow = p(8)
    lwrist= p(9);  rwrist = p(10)
    lhip  = p(11); rhip   = p(12)
    lknee = p(13); rknee  = p(14)
    lankle= p(15); rankle = p(16)

    # Réutiliser l'analyse openpose en remappant les points dans le bon ordre
    # On reconstruit une liste factice dans l'ordre openpose
    class _FakePt:
        def __init__(self, xy):
            self._xy = xy
        def x(self): return self._xy[0] if self._xy else 0
        def y(self): return self._xy[1] if self._xy else 0

    # Mapping COCO17 → OpenPose18 (approximatif, sans Neck)
    neck_approx = _midpoint(lsho, rsho)
    remapped = [
        nose, neck_approx,        # 0=Nose, 1=Neck
        rsho, relbow, rwrist,     # 2-4 right arm
        lsho, lelbow, lwrist,     # 5-7 left arm
        rhip, rknee, rankle,      # 8-10 right leg
        lhip, lknee, lankle,      # 11-13 left leg
        None, None, None, None,   # 14-17 eyes/ears (pas dans COCO17 facilement)
    ]

    fake_points = [_FakePt(xy) for xy in remapped]
    return _analyze_openpose(fake_points)


# ---------------------------------------------------------------------------
# Génération de la caption
# ---------------------------------------------------------------------------

def generate_caption(shape) -> str:
    """Génère une caption anglaise depuis une shape skeleton."""
    points = shape.points
    skeleton_name = getattr(shape, "skeleton_name", "human")
    label = getattr(shape, "label", "person")

    if skeleton_name in ("openpose_fullbody", "coco_fullbody"):
        props = _analyze_openpose(points)
    else:
        props = _analyze_coco17(points)

    parts = [label]

    if "position" in props:
        parts.append(props["position"])
    if "orientation" in props:
        parts.append(props["orientation"])
    if "facing" in props:
        parts.append(props["facing"])
    if "camera_angle" in props:
        parts.append(props["camera_angle"])
    if "weight" in props:
        parts.append(props["weight"])
    if "arms" in props:
        parts.append(props["arms"])
    if "legs" in props:
        parts.append(props["legs"])

    return ", ".join(parts)


def generate_captions_from_json(json_path: Path) -> list[str]:
    """
    Génère les captions depuis un JSON LabelMe (pour le batch).
    Retourne une liste de captions (une par skeleton dans le fichier).
    """
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    captions = []
    for shape_data in data.get("shapes", []):
        if shape_data.get("shape_type") != "skeleton":
            continue

        from PyQt5.QtCore import QPointF
        points = [QPointF(pt[0], pt[1]) for pt in shape_data.get("points", [])]

        class _FakeShape:
            pass
        s = _FakeShape()
        s.points = points
        s.skeleton_name = shape_data.get("skeleton_name", "human")
        s.label = shape_data.get("label", "person")

        captions.append(generate_caption(s))

    return captions


# ---------------------------------------------------------------------------
# Export depuis LabelMe — image courante
# ---------------------------------------------------------------------------

def export_pose_caption(parent_window) -> None:
    if not getattr(parent_window, "imagePath", None):
        QtWidgets.QMessageBox.warning(
            parent_window, "Caption de pose", "Aucune image ouverte."
        )
        return

    skeleton_shapes = []
    for item in parent_window.labelList:
        if item.checkState() == Qt.Checked:
            shape = item.shape()
            if shape.shape_type == "skeleton":
                skeleton_shapes.append(shape)

    if not skeleton_shapes:
        QtWidgets.QMessageBox.information(
            parent_window, "Caption de pose",
            "Aucun skeleton coché dans le panel Polygon Labels."
        )
        return

    captions = [generate_caption(s) for s in skeleton_shapes]
    full_caption = " | ".join(captions)

    image_path = Path(parent_window.imagePath)
    out_path = image_path.parent / f"{image_path.stem}_caption.txt"

    out_str, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent_window, "Sauvegarder la caption",
        str(out_path), "Texte (*.txt)",
    )
    if not out_str:
        return

    Path(out_str).write_text(full_caption, encoding="utf-8")
    QtWidgets.QMessageBox.information(
        parent_window, "Caption de pose",
        f"Caption sauvegardée :\n{full_caption}\n\n→ {out_str}"
    )


# ---------------------------------------------------------------------------
# Export batch — dossier entier
# ---------------------------------------------------------------------------

def export_pose_captions_batch(parent_window) -> None:
    folder = QtWidgets.QFileDialog.getExistingDirectory(
        parent_window, "Choisir le dossier contenant les JSON LabelMe"
    )
    if not folder:
        return

    folder_path = Path(folder)
    json_files = sorted(folder_path.glob("*.json"))

    if not json_files:
        QtWidgets.QMessageBox.warning(
            parent_window, "Caption batch",
            "Aucun fichier JSON trouvé dans ce dossier."
        )
        return

    count = 0
    for json_file in json_files:
        captions = generate_captions_from_json(json_file)
        if not captions:
            continue
        txt_path = json_file.with_suffix(".txt")
        txt_path.write_text(" | ".join(captions), encoding="utf-8")
        count += 1

    QtWidgets.QMessageBox.information(
        parent_window, "Caption batch",
        f"{count} fichier(s) traité(s) sur {len(json_files)} JSON trouvés.\n"
        f"Captions sauvegardées dans :\n{folder}"
    )
