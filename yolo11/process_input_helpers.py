# yolo11/process_input_helpers.py
import os
import json

def save_labelme_json(image_path, shapes, image_width, image_height, out_json_path):
    """Sauvegarde un JSON LabelMe à partir des shapes."""
    doc = {
        "version": "5.9.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------
# BOXES
# --------------------------------------------------------------------
def _boxes_to_shapes(boxes, classes=None, confs=None, group_id=None):
    shapes = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b
        label = classes[i] if classes and i < len(classes) else "object"
        desc = ""
        if confs and i < len(confs) and confs[i] is not None:
            desc = f"conf={confs[i]:.3f}"
        shapes.append({
            "label": label,
            "points": [
                [float(x1), float(y1)],
                [float(x2), float(y2)]
            ],
            "group_id": group_id,
            "description": desc,
            "shape_type": "rectangle",
            "flags": {},
            "mask": None
        })
    return shapes


# --------------------------------------------------------------------
# MASKS
# --------------------------------------------------------------------
def _masks_to_shapes(masks, group_id=None, label="mask", labels=None):
    shapes = []
    for i, poly in enumerate(masks):
        lbl = labels[i] if labels and i < len(labels) else label
        shapes.append({
            "label": lbl,
            "points": [[float(x), float(y)] for x, y in poly],
            "group_id": group_id,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None
        })
    return shapes


# --------------------------------------------------------------------
# KEYPOINTS / SKELETON
# --------------------------------------------------------------------
def _keypoints_to_shapes(keypoints, group_id=None, label="skeleton", skeleton_name=None):
    """
    keypoints    : [ [num_part, x, y, conf], ... ]
    skeleton_name: nom du squelette (ex: "human", "openpose_fullbody", "coco_fullbody")
                   Sauvegardé dans le JSON pour éviter l'heuristique au rechargement.
    Retourne UN SEUL shape LabelMe contenant tous les points.
    """
    return [{
        "label": label,
        "points": [[float(k[0]), float(k[1]), float(k[2]), float(k[3])] for k in keypoints],
        "group_id": group_id,
        "shape_type": "skeleton",
        "flags": {},
        "description": "",
        "skeleton_name": skeleton_name or "human",
    }]
