# yolo11/tracking_helpers.py
import numpy as np

def bbox_iou(boxA, boxB):
    """Calcule l'IoU entre deux boîtes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0.0


def mask_bbox(mask_points):
    """Retourne la bbox englobante d'un masque [[x,y], ...]."""
    pts = np.array(mask_points)
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    return [float(x1), float(y1), float(x2), float(y2)]


def keypoints_center(kps):
    """Retourne le centre (x,y) des keypoints valides."""
    pts = np.array([[x, y] for x, y, *_ in kps if x >= 0 and y >= 0])
    if len(pts) == 0:
        return None
    return np.mean(pts, axis=0)


def match_entities(boxes, masks, keypoints):
    """
    Fusionne spatialement boxes, masks et keypoints POUR UNE FRAME.
    Aucun tracking temporel ici, seulement association spatiale.
    """
    entities = []

    mask_boxes = [(mask_bbox(m["points"]), m) for m in masks]
    kp_centers = [(keypoints_center(k["keypoints"]), k) for k in keypoints]

    used_boxes = set()
    used_masks = set()

    # --- Associe masks ↔ boxes par IoU
    for mi, (mbox, m) in enumerate(mask_boxes):
        best_box = None
        best_iou = 0.0
        for bi, b in enumerate(boxes):
            if bi in used_boxes:
                continue
            iou = bbox_iou(mbox, b["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_box = bi
        if best_box is not None and best_iou > 0.3:
            entities.append({
                "box": boxes[best_box],
                "mask": m,
                "keypoints": None
            })
            used_boxes.add(best_box)
            used_masks.add(mi)

    # --- Boxes seules
    for bi, b in enumerate(boxes):
        if bi not in used_boxes:
            entities.append({"box": b, "mask": None, "keypoints": None})

    # --- Associe keypoints ↔ boxes par distance centre
    for kp_center, k in kp_centers:
        if kp_center is None:
            continue

        best_ent = None
        best_dist = float("inf")
        for ent in entities:
            if ent["box"] is None:
                continue
            bx1, by1, bx2, by2 = ent["box"]["bbox"]
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            dist = np.linalg.norm(kp_center - np.array([cx, cy]))
            if dist < best_dist:
                best_dist = dist
                best_ent = ent

        if best_ent is not None and best_dist < 80:  # seuil adaptable
            best_ent["keypoints"] = k

    return entities
