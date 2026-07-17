# yolo11/inference.py
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image
import cv2
import numpy as np

from .coco import COCO_SKELETON, COCO_COLORS

def _rotate_frame(frame: np.ndarray, angle: int) -> np.ndarray:
    """Rotation horaire de l'image : 0, 90, 180, 270 degrés.
    Côté UI : 90 = "90°R" (droite, horaire), 270 = "90°L" (gauche, anti-horaire)."""
    if angle == 0:
        return frame
    elif angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def _rotate_keypoints(kpts: list, angle: int, w_orig: int, h_orig: int) -> list:
    """
    Retransforme les coordonnées keypoints depuis l'image tournée
    vers l'image originale.
    kpts    : liste de [id, x, y, conf] — coordonnées dans l'image TOURNÉE
    w_orig  : largeur de l'image ORIGINALE
    h_orig  : hauteur de l'image ORIGINALE
    Côté UI : angle=90 → "90°R", angle=270 → "90°L".
    """
    if angle == 0:
        return kpts
    result = []
    for k in kpts:
        kid, x, y, conf = k
        if x == 0.0 and y == 0.0:
            result.append(k)
            continue
        if angle == 90:
            x2, y2 = y, h_orig - 1 - x
        elif angle == 180:
            x2, y2 = w_orig - 1 - x, h_orig - 1 - y
        elif angle == 270:
            x2, y2 = w_orig - 1 - y, x
        else:
            x2, y2 = x, y
        result.append([kid, x2, y2, conf])
    return result

def _rotate_bbox(bbox: list, angle: int, w: int, h: int) -> list:
    """Retransforme une bbox [x1,y1,x2,y2] depuis image tournée vers originale.
    Côté UI : angle=90 → "90°R", angle=270 → "90°L"."""
    if angle == 0:
        return bbox
    x1, y1, x2, y2 = bbox
    if angle == 90:
        return [y1, w - x2, y2, w - x1]
    elif angle == 180:
        return [w - x2, h - y2, w - x1, h - y1]
    elif angle == 270:
        return [h - y2, x1, h - y1, x2]
    return bbox


from .tracking_helpers import bbox_iou
import json as _json


def _load_skeleton_connections(name: str):
    spec_path = Path(__file__).parent.parent / "coco_specs" / f"{name}_skeleton.json"
    if not spec_path.exists():
        return COCO_SKELETON
    with open(spec_path, encoding="utf-8") as f:
        data = _json.load(f)
    if "skeleton" in data:
        return [tuple(c[:2]) for c in data["skeleton"]]
    connections = data.get("connections", [])
    if isinstance(connections, dict):
        body = connections.get("body", [])
        return [(c["from"], c["to"]) for c in body]
    else:
        return [(c["from"], c["to"]) for c in connections]


_SKEL_CONNECTIONS_CACHE = {}

def _get_skeleton_connections_cached(name: str):
    if name not in _SKEL_CONNECTIONS_CACHE:
        _SKEL_CONNECTIONS_CACHE[name] = _load_skeleton_connections(name)
    return _SKEL_CONNECTIONS_CACHE[name]


def get_skeleton_connections(cfg: dict):
    name = cfg.get("skeleton_name", "human")
    if name == "human":
        return COCO_SKELETON
    return _get_skeleton_connections_cached(name)


_OFFSETS = {
    "coco": {
        "n_body":      17,
        "feet_start":  17,
        "face_start":  23,
        "lhand_start": 91,
        "rhand_start": 112,
        "n_hand":      21,
    },
    "openpose": {
        "n_body":      18,
        "feet_start":  18,
        "face_start":  24,
        "lhand_start": 92,
        "rhand_start": 113,
        "n_hand":      21,
    },
}

def _skeleton_format_for(skeleton_name: str) -> str:
    if skeleton_name == "openpose_fullbody":
        return "openpose"
    return "coco"


from .model import get_model, get_dwpose
from .process_input_helpers import save_labelme_json, _boxes_to_shapes, _masks_to_shapes, _keypoints_to_shapes
from .tracking.kalman_tracker import KalmanTracker
from .config import load_config

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _safe_txt(label: Optional[str], gid: Optional[Any]) -> str:
    if gid is not None and not (label or "").endswith(f"_{gid}"):
        return f"{label or ''} {gid}"
    return (label or "").strip()


def draw_shapes_on_frame(frame: np.ndarray, shapes: List[Dict[str, Any]], skeleton_connections=None) -> np.ndarray:
    joint_color = COCO_COLORS.get("joint",    (0, 180, 255))
    bone_color  = COCO_COLORS.get("bone",     (0, 120, 255))
    box_color   = COCO_COLORS.get("box",      (0, 200, 0))
    mask_color  = COCO_COLORS.get("mask",     (0, 0, 255))
    label_bg    = COCO_COLORS.get("label_bg", (0, 0, 0))
    label_txt   = COCO_COLORS.get("label_txt",(255, 255, 255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness_txt, pad = 0.55, 1, 4

    for s in shapes:
        gid        = s.get("group_id")
        label      = s.get("label", "") or ""
        shape_type = s.get("shape_type")
        txt        = _safe_txt(label, gid)

        if shape_type == "rectangle":
            try:
                x1, y1 = map(int, s["points"][0])
                x2, y2 = map(int, s["points"][1])
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                if txt:
                    (w, h), _ = cv2.getTextSize(txt, font, font_scale, thickness_txt)
                    tx, ty = x1, max(0, y1 - h - 2*pad)
                    cv2.rectangle(frame, (tx, ty), (tx + w + 2*pad, ty + h + 2*pad), label_bg, -1)
                    cv2.putText(frame, txt, (tx + pad, ty + h + pad//2), font, font_scale, label_txt, thickness_txt, cv2.LINE_AA)
            except Exception:
                continue

        elif shape_type == "polygon":
            try:
                pts = np.array(s["points"], np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, mask_color, 2)
                if txt:
                    cx, cy = int(np.mean(pts[:,0,0])), int(np.mean(pts[:,0,1]))
                    (w, h), _ = cv2.getTextSize(txt, font, font_scale, thickness_txt)
                    tx, ty = cx - w//2 - pad, max(0, cy - h//2 - pad)
                    cv2.rectangle(frame, (tx, ty), (tx + w + 2*pad, ty + h + 2*pad), label_bg, -1)
                    cv2.putText(frame, txt, (tx + pad, ty + h + pad//2), font, font_scale, label_txt, thickness_txt, cv2.LINE_AA)
            except Exception:
                continue

        elif shape_type in ["point", "skeleton"]:
            pts = s.get("points", [])
            if not pts:
                continue
            for p in pts:
                if p[1] == 0 and p[2] == 0:
                    continue
                try:
                    cv2.circle(frame, (int(p[1]), int(p[2])), 4, joint_color, -1)
                except Exception:
                    continue
            connections = skeleton_connections if skeleton_connections is not None else COCO_SKELETON
            for a, b in connections:
                if a < len(pts) and b < len(pts):
                    xa, ya = pts[a][1], pts[a][2]
                    xb, yb = pts[b][1], pts[b][2]
                    if (xa == 0 and ya == 0) or (xb == 0 and yb == 0):
                        continue
                    try:
                        if all(np.isfinite(v) for v in [xa, ya, xb, yb]):
                            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), bone_color, 2)
                    except Exception:
                        continue
            if txt:
                try:
                    x0, y0 = int(pts[0][1]), int(pts[0][2])
                    (w, h), _ = cv2.getTextSize(txt, font, font_scale, thickness_txt)
                    tx, ty = x0 - w//2 - pad, max(0, y0 - h - 2*pad)
                    cv2.rectangle(frame, (tx, ty), (tx + w + 2*pad, ty + h + 2*pad), label_bg, -1)
                    cv2.putText(frame, txt, (tx + pad, ty + h + pad//2), font, font_scale, label_txt, thickness_txt, cv2.LINE_AA)
                except Exception:
                    pass
    return frame


def detect_frame(
    frame: np.ndarray,
    mdl_boxes,
    mdl_seg,
    mdl_skeletons,
    allowed_classes: set,
    tracker=None,
    cfg: dict = None,
) -> Dict[str, List]:
    if cfg is None:
        cfg = load_config()

    raw_boxes = []
    r_boxes = mdl_boxes(frame, verbose=False)
    bb = r_boxes[0].boxes
    if bb is not None and len(bb) > 0:
        for b, c in zip(bb.xyxy.cpu().numpy(), bb.cls):
            lbl = mdl_boxes.names[int(c)]
            if allowed_classes and lbl not in allowed_classes:
                continue
            raw_boxes.append({"bbox": b.tolist(), "label": lbl})

    if tracker and raw_boxes:
        tracked_ids = tracker.update([b["bbox"] for b in raw_boxes])
    else:
        tracked_ids = list(range(len(raw_boxes)))

    for i, box in enumerate(raw_boxes):
        tid = tracked_ids[i] if i < len(tracked_ids) else i
        box["track_id"]   = tid
        box["label_full"] = f"{box['label']}_{tid}"

    def find_parent_box(x, y):
        for box in raw_boxes:
            bx1, by1, bx2, by2 = box["bbox"]
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                return box
        return None

    def in_box(x, y, bbox, margin_ratio=0.1):
        bx1, by1, bx2, by2 = bbox
        margin = margin_ratio * max(bx2 - bx1, by2 - by1)
        return (bx1 - margin <= x <= bx2 + margin) and (by1 - margin <= y <= by2 + margin)

    def dedup_boxes(boxes, iou_thresh):
        if iou_thresh is None or iou_thresh <= 0 or len(boxes) <= 1:
            return boxes
        kept = []
        for b in boxes:
            duplicate = False
            for k in kept:
                if bbox_iou(b["bbox"], k["bbox"]) > iou_thresh:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(b)
        return kept

    raw_masks = []
    if mdl_seg:
        r_seg   = mdl_seg(frame, verbose=False)
        mm      = getattr(r_seg[0], "masks", None)
        seg_cls = r_seg[0].boxes.cls.cpu().numpy() if r_seg[0].boxes is not None else []
        if mm is not None:
            for i, poly in enumerate(getattr(mm, "xy", [])):
                if i < len(seg_cls) and int(seg_cls[i]) != 0:
                    continue
                pts = [[float(x), float(y)] for x, y in
                       (poly.tolist() if hasattr(poly, "tolist") else poly)]
                if not pts:
                    continue
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                parent = find_parent_box(cx, cy)
                raw_masks.append({
                    "points":   pts,
                    "label":    parent["label_full"] if parent else "person",
                    "track_id": parent["track_id"]   if parent else None,
                })

    raw_skeletons = []
    backend       = cfg.get("pose_backend", "yolo")
    skeleton_name = cfg.get("skeleton_name", "human")

    if backend == "dwpose" or mdl_skeletons:
        min_conf = cfg.get("min_conf", 0.2)

        if backend == "dwpose":
            detector      = get_dwpose()
            dwpose_groups = cfg.get("dwpose_groups", {})
            nms_iou       = cfg.get("nms_iou", None)
            person_boxes  = [b for b in raw_boxes if b["label"] == "person"]
            person_boxes  = dedup_boxes(person_boxes, nms_iou)

            rotation = int(cfg.get("rotation", 0)) % 360
            h_orig, w_orig = frame.shape[:2]
            frame_rot = _rotate_frame(frame, rotation)
            h_rot, w_rot = frame_rot.shape[:2]

            if rotation != 0:
                boxes_rot = []
                for b in person_boxes:
                    bx = _rotate_bbox(b["bbox"], rotation, w_orig, h_orig)
                    bx = [min(bx[0],bx[2]), min(bx[1],bx[3]),
                          max(bx[0],bx[2]), max(bx[1],bx[3])]
                    boxes_rot.append(bx)
                boxes_np = np.array(boxes_rot, dtype=np.float32)
            else:
                boxes_np = np.array([b["bbox"] for b in person_boxes], dtype=np.float32)

            sk_fmt = _skeleton_format_for(skeleton_name)
            offsets = _OFFSETS[sk_fmt]

            pose   = detector.detect_raw(frame_rot, boxes_np, skeleton_format=sk_fmt)
            kp_all = pose["keypoints"]
            sc_all = pose["scores"]

            for person_idx in range(len(kp_all)):
                kpts   = []
                if rotation != 0:
                    bbox_p = boxes_rot[person_idx]
                else:
                    bbox_p = person_boxes[person_idx]["bbox"]
                n_total = kp_all.shape[1]

                if dwpose_groups.get("body", True):
                    n_body = offsets["n_body"]
                    for part_idx in range(n_body):
                        x    = float(kp_all[person_idx, part_idx, 0])
                        y    = float(kp_all[person_idx, part_idx, 1])
                        conf = float(sc_all[person_idx, part_idx])
                        if conf < min_conf or not in_box(x, y, bbox_p):
                            x, y, conf = 0.0, 0.0, 0.0
                        kpts.append([float(part_idx), x, y, conf])

                if dwpose_groups.get("feet", False):
                    feet_start = offsets["feet_start"]
                    for fi in range(6):
                        idx = feet_start + fi
                        if idx >= n_total:
                            kpts.append([float(idx), 0.0, 0.0, 0.0])
                            continue
                        x    = float(kp_all[person_idx, idx, 0])
                        y    = float(kp_all[person_idx, idx, 1])
                        conf = float(sc_all[person_idx, idx])
                        if conf < min_conf or not in_box(x, y, bbox_p):
                            x, y, conf = 0.0, 0.0, 0.0
                        kpts.append([float(idx), x, y, conf])

                if dwpose_groups.get("face", False):
                    face_start = offsets["face_start"]
                    for fi in range(68):
                        idx = face_start + fi
                        if idx >= n_total:
                            kpts.append([float(idx), 0.0, 0.0, 0.0])
                            continue
                        x    = float(kp_all[person_idx, idx, 0])
                        y    = float(kp_all[person_idx, idx, 1])
                        conf = float(sc_all[person_idx, idx])
                        if conf < min_conf or not in_box(x, y, bbox_p):
                            x, y, conf = 0.0, 0.0, 0.0
                        kpts.append([float(idx), x, y, conf])

                if dwpose_groups.get("hands", False):
                    for src_off in [offsets["lhand_start"], offsets["rhand_start"]]:
                        for ki in range(offsets["n_hand"]):
                            idx = src_off + ki
                            if idx >= n_total:
                                kpts.append([float(idx), 0.0, 0.0, 0.0])
                                continue
                            x    = float(kp_all[person_idx, idx, 0])
                            y    = float(kp_all[person_idx, idx, 1])
                            conf = float(sc_all[person_idx, idx])
                            if conf < min_conf or not in_box(x, y, bbox_p, margin_ratio=0.2):
                                x, y, conf = 0.0, 0.0, 0.0
                            kpts.append([float(idx), x, y, conf])

                visible = [(k[1], k[2]) for k in kpts if k[3] > 0 and k[1] > 0 and k[2] > 0]
                if not visible:
                    continue
                if person_idx < len(person_boxes):
                    parent = person_boxes[person_idx]
                    label, tid = parent["label_full"], parent["track_id"]
                else:
                    label, tid = "skeleton", None
                if rotation != 0:
                    kpts = _rotate_keypoints(kpts, rotation, w_orig, h_orig)
                raw_skeletons.append({"keypoints": kpts, "label": label, "track_id": tid})

        else:
            if mdl_skeletons:
                r_sk = mdl_skeletons(frame, verbose=False)
                kp   = getattr(r_sk[0], "keypoints", None)
                if kp is not None:
                    sk_array = kp.xy.cpu().numpy()
                    sk_conf  = kp.conf.cpu().numpy()
                    for inst, conf in zip(sk_array, sk_conf):
                        kpts = [
                            [float(i), float(x), float(y), float(c)]
                            for i, ((x, y), c) in enumerate(zip(inst, conf))
                        ]
                        if not kpts:
                            continue
                        visible = [(k[1], k[2]) for k in kpts if k[3] > min_conf]
                        if not visible:
                            continue
                        sx, sy = visible[0]
                        parent = find_parent_box(sx, sy)
                        raw_skeletons.append({
                            "keypoints": kpts,
                            "label":     parent["label_full"] if parent else "skeleton",
                            "track_id":  parent["track_id"]   if parent else None,
                        })

    return {
        "raw_boxes":     raw_boxes,
        "raw_masks":     raw_masks,
        "raw_skeletons": raw_skeletons,
        "skeleton_name": skeleton_name,
    }


def build_shapes(
    detection: Dict[str, List],
    requested_tasks: set,
) -> List[Dict]:
    raw_boxes     = detection["raw_boxes"]
    raw_masks     = detection["raw_masks"]
    raw_skeletons = detection["raw_skeletons"]
    shapes = []

    if "boxes" in requested_tasks:
        shapes += _boxes_to_shapes(
            [b["bbox"]       for b in raw_boxes],
            [b["label_full"] for b in raw_boxes],
        )

    if "segmentation" in requested_tasks:
        shapes += _masks_to_shapes(
            [m["points"] for m in raw_masks],
            labels=[m["label"] for m in raw_masks],
        )

    if "skeletons" in requested_tasks:
        skeleton_name = detection.get("skeleton_name", "human")
        for sk in raw_skeletons:
            shapes += _keypoints_to_shapes(
                sk["keypoints"],
                group_id=sk["track_id"],
                label=sk["label"],
                skeleton_name=skeleton_name,
            )

    return shapes


def predict(
    input_path: str,
    tasks=None,
    target_fps: float = None,
    use_tracker: bool = True,
    cfg: dict = None,
):
    if cfg is None:
        cfg = load_config()
    allowed_classes = set(cfg.get("classes") or [])
    path            = Path(input_path)

    tasks = tasks or []
    if "everything" in tasks:
        tasks = ["boxes", "segmentation", "skeletons"]
    requested_tasks = set(tasks)

    backend = cfg.get("pose_backend", "yolo")

    mdl_boxes     = get_model("boxes")
    mdl_seg       = get_model("segmentation") if "segmentation" in requested_tasks else None
    mdl_skeletons = (get_model("skeletons")
                     if "skeletons" in requested_tasks and backend == "yolo"
                     else None)

    out_dir = os.path.join(
        str(path.parent),
        cfg.get("output", {}).get("processed_dirname", "processed")
    )
    os.makedirs(out_dir, exist_ok=True)

    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return _predict_video(
            path, out_dir, cfg,
            mdl_boxes, mdl_seg, mdl_skeletons,
            allowed_classes, requested_tasks,
            target_fps, use_tracker,
        )
    elif path.is_dir():
        images  = sorted(f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        print(f"[DOSSIER] {len(images)} images trouvées dans {path}")
        results = [
            _predict_single_image(img_path, out_dir, idx,
                                  mdl_boxes, mdl_seg, mdl_skeletons,
                                  allowed_classes, requested_tasks, cfg)
            for idx, img_path in enumerate(images)
        ]
        print(f"[DOSSIER] {len(results)} images traitées.")
        return {"images_processed": len(results), "output_dir": out_dir}
    elif path.suffix.lower() in IMAGE_EXTENSIONS:
        return _predict_single_image(
            path, out_dir, 0,
            mdl_boxes, mdl_seg, mdl_skeletons,
            allowed_classes, requested_tasks, cfg,
        )
    else:
        raise ValueError(f"Input non reconnu : {input_path}")


def _predict_single_image(
    img_path: Path,
    out_dir: str,
    idx: int,
    mdl_boxes, mdl_seg, mdl_skeletons,
    allowed_classes: set,
    requested_tasks: set,
    cfg: dict,
) -> Dict:
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[IMAGE] Impossible de lire {img_path}, ignorée.")
        return {}

    h, w = frame.shape[:2]
    detection = detect_frame(frame, mdl_boxes, mdl_seg, mdl_skeletons,
                             allowed_classes, tracker=None, cfg=cfg)
    shapes = build_shapes(detection, requested_tasks)

    out_img  = os.path.join(out_dir, f"{img_path.stem}_annotated{img_path.suffix}")
    out_json = os.path.join(out_dir, f"{img_path.stem}_annotated.json")

    cv2.imwrite(out_img, frame)
    save_labelme_json(out_img, shapes, w, h, out_json)

    if cfg.get("output", {}).get("save_openpose_png", False):
        _save_openpose_png(frame, out_dir, img_path.stem, cfg)

    print(f"[IMAGE] {img_path.name} → {len(shapes)} shapes exportées")
    return {"image": out_img, "json": out_json, "shapes": len(shapes)}


def _predict_video(
    path: Path,
    out_dir: str,
    cfg: dict,
    mdl_boxes, mdl_seg, mdl_skeletons,
    allowed_classes: set,
    requested_tasks: set,
    target_fps: float,
    use_tracker: bool,
) -> Dict:
    print(f"[VIDEO] Chargement : {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir : {path}")

    orig_fps     = cap.get(cv2.CAP_PROP_FPS)
    orig_width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[VIDEO] {orig_fps:.1f}fps — {total_frames} frames — {orig_width}x{orig_height}")

    sample_interval = max(1, int(orig_fps / target_fps)) if target_fps and target_fps > 0 else 1
    print(f"[VIDEO] Export 1 frame toutes les {sample_interval} frames")

    out_video_path = os.path.join(out_dir, "annotated_video.mp4")
    out_writer = cv2.VideoWriter(
        out_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(target_fps or orig_fps),
        (orig_width, orig_height),
    )

    tracker = KalmanTracker(
        iou_thresh=0.3,
        max_missed_seconds=2.0,
        source_fps=orig_fps,
        sample_interval=1,
    ) if use_tracker else None

    frame_idx = export_idx = 0
    print("[VIDEO] Début traitement...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        detection = detect_frame(frame, mdl_boxes, mdl_seg, mdl_skeletons,
                                 allowed_classes, tracker=tracker, cfg=cfg)

        if frame_idx % sample_interval != 0:
            continue

        shapes    = build_shapes(detection, requested_tasks)
        img_path  = os.path.join(out_dir, f"frame_{export_idx:06d}.jpg")
        json_path = os.path.join(out_dir, f"frame_{export_idx:06d}.json")

        cv2.imwrite(img_path, frame)
        save_labelme_json(img_path, shapes, orig_width, orig_height, json_path)

        if cfg.get("output", {}).get("save_openpose_png", False):
            _save_openpose_png(frame, out_dir, f"frame_{export_idx:06d}", cfg)

        annotated = draw_shapes_on_frame(
            frame.copy(), shapes,
            skeleton_connections=get_skeleton_connections(cfg)
        )
        out_writer.write(annotated)
        if cfg.get("video", {}).get("show_live", True):
            cv2.imshow(cfg.get("video", {}).get("window_name", "YOLO11 Preview"), annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        export_idx += 1

    cv2.destroyAllWindows()
    cap.release()
    out_writer.release()
    print(f"[VIDEO] Fin — {export_idx} frames exportées.")

    return {
        "frames_exported": export_idx,
        "fps_used":        target_fps or orig_fps,
        "output_video":    out_video_path,
        "output_dir":      out_dir,
    }


def _save_openpose_png(frame: np.ndarray, out_dir: str, stem: str, cfg: dict):
    try:
        from .model import get_model
        detector = get_dwpose()
        mdl_boxes = get_model("boxes")
        allowed_classes = set(cfg.get("classes") or [])

        r_boxes = mdl_boxes(frame, verbose=False)
        bb = r_boxes[0].boxes
        person_boxes = []
        if bb is not None and len(bb) > 0:
            for b, c in zip(bb.xyxy.cpu().numpy(), bb.cls):
                if mdl_boxes.names[int(c)] == "person":
                    person_boxes.append(b.tolist())

        boxes_np = np.array(person_boxes, dtype=np.float32) if person_boxes else np.zeros((0, 4), dtype=np.float32)
        png = detector.render_png(frame, boxes_np)
        out_path = os.path.join(out_dir, f"{stem}_openpose.png")
        cv2.imwrite(out_path, png)
        print(f"[PNG] OpenPose sauvegardé : {out_path}")
    except Exception as e:
        print(f"[PNG] Erreur rendu OpenPose : {e}")


def reinfer_dwpose_from_labelme_boxes(
    image_path: str,
    json_path: str,
    cfg: dict = None,
) -> list:
    if cfg is None:
        cfg = load_config()

    import json as _json_mod

    seg_enabled = cfg.get("seg_enabled", False)

    with open(json_path, encoding="utf-8") as f:
        data = _json_mod.load(f)

    person_boxes = []
    non_skeleton_shapes = []
    for shape in data.get("shapes", []):
        if shape["shape_type"] == "rectangle":
            pts = shape["points"]
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            person_boxes.append({
                "bbox":       [x1, y1, x2, y2],
                "label":      shape.get("label", "person"),
                "label_full": shape.get("label", "person"),
                "track_id":   shape.get("group_id"),
            })
            non_skeleton_shapes.append(shape)
        elif shape["shape_type"] == "polygon" and seg_enabled:
            continue
        elif shape["shape_type"] != "skeleton":
            non_skeleton_shapes.append(shape)

    if not person_boxes:
        print("[REINFER] Aucune box rectangle trouvée dans le JSON.")
        return data.get("shapes", [])

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Impossible de lire l'image : {image_path}")

    h, w = frame.shape[:2]

    rotation = int(cfg.get("rotation", 0)) % 360
    frame_inf = _rotate_frame(frame, rotation)

    if rotation != 0:
        boxes_rot = []
        for b in person_boxes:
            bx = _rotate_bbox(b["bbox"], rotation, w, h)
            bx = [min(bx[0],bx[2]), min(bx[1],bx[3]),
                  max(bx[0],bx[2]), max(bx[1],bx[3])]
            boxes_rot.append(bx)
        boxes_np = np.array(boxes_rot, dtype=np.float32)
    else:
        boxes_np = np.array([b["bbox"] for b in person_boxes], dtype=np.float32)

    from .model import get_dwpose
    detector      = get_dwpose()
    skeleton_name = cfg.get("skeleton_name", "human")
    sk_fmt        = _skeleton_format_for(skeleton_name)
    offsets       = _OFFSETS[sk_fmt]
    dwpose_groups = cfg.get("dwpose_groups", {"body": True, "feet": False, "face": False, "hands": False})
    min_conf      = cfg.get("min_conf", 0.3)

    pose     = detector.detect_raw(frame_inf, boxes_np, skeleton_format=sk_fmt)
    kp_all   = pose["keypoints"]
    sc_all   = pose["scores"]

    def in_box(x, y, bbox, margin_ratio=0.1):
        bx1, by1, bx2, by2 = bbox
        margin = margin_ratio * max(bx2 - bx1, by2 - by1)
        return (bx1 - margin <= x <= bx2 + margin) and (by1 - margin <= y <= by2 + margin)

    new_skeletons = []
    for person_idx in range(len(kp_all)):
        kpts   = []
        if rotation != 0:
            bbox_p = boxes_np[person_idx].tolist()
        else:
            bbox_p = person_boxes[person_idx]["bbox"]
        n_total = kp_all.shape[1]

        if dwpose_groups.get("body", True):
            for part_idx in range(offsets["n_body"]):
                x    = float(kp_all[person_idx, part_idx, 0])
                y    = float(kp_all[person_idx, part_idx, 1])
                conf = float(sc_all[person_idx, part_idx])
                if conf < min_conf or not in_box(x, y, bbox_p):
                    x, y, conf = 0.0, 0.0, 0.0
                kpts.append([float(part_idx), x, y, conf])

        if dwpose_groups.get("feet", False):
            for fi in range(6):
                idx = offsets["feet_start"] + fi
                if idx >= n_total:
                    kpts.append([float(idx), 0.0, 0.0, 0.0])
                    continue
                x    = float(kp_all[person_idx, idx, 0])
                y    = float(kp_all[person_idx, idx, 1])
                conf = float(sc_all[person_idx, idx])
                if conf < min_conf or not in_box(x, y, bbox_p):
                    x, y, conf = 0.0, 0.0, 0.0
                kpts.append([float(idx), x, y, conf])

        if dwpose_groups.get("face", False):
            for fi in range(68):
                idx = offsets["face_start"] + fi
                if idx >= n_total:
                    kpts.append([float(idx), 0.0, 0.0, 0.0])
                    continue
                x    = float(kp_all[person_idx, idx, 0])
                y    = float(kp_all[person_idx, idx, 1])
                conf = float(sc_all[person_idx, idx])
                if conf < min_conf or not in_box(x, y, bbox_p):
                    x, y, conf = 0.0, 0.0, 0.0
                kpts.append([float(idx), x, y, conf])

        if dwpose_groups.get("hands", False):
            for src_off in [offsets["lhand_start"], offsets["rhand_start"]]:
                for ki in range(offsets["n_hand"]):
                    idx = src_off + ki
                    if idx >= n_total:
                        kpts.append([float(idx), 0.0, 0.0, 0.0])
                        continue
                    x    = float(kp_all[person_idx, idx, 0])
                    y    = float(kp_all[person_idx, idx, 1])
                    conf = float(sc_all[person_idx, idx])
                    if conf < min_conf or not in_box(x, y, bbox_p, margin_ratio=0.2):
                        x, y, conf = 0.0, 0.0, 0.0
                    kpts.append([float(idx), x, y, conf])

        visible = [(k[1], k[2]) for k in kpts if k[3] > 0 and k[1] > 0 and k[2] > 0]
        if not visible:
            continue

        if rotation != 0:
            kpts = _rotate_keypoints(kpts, rotation, w, h)

        parent = person_boxes[person_idx]
        new_skeletons.append({
            "label":        parent["label_full"],
            "group_id":     parent["track_id"],
            "description":  "",
            "shape_type":   "skeleton",
            "flags":        {},
            "mask":         None,
            "points":       kpts,
            "skeleton_name": skeleton_name,
        })

    new_masks = []
    if seg_enabled:
        from .model import get_model
        mdl_seg = get_model("segmentation")
        if mdl_seg is None:
            print("[REINFER] Segmentation demandée mais modèle non chargé.")
        else:
            r_seg   = mdl_seg(frame, verbose=False)
            mm      = getattr(r_seg[0], "masks", None)
            seg_cls = r_seg[0].boxes.cls.cpu().numpy() if r_seg[0].boxes is not None else []
            if mm is not None:
                for i, poly in enumerate(getattr(mm, "xy", [])):
                    if i < len(seg_cls) and int(seg_cls[i]) != 0:
                        continue
                    pts = [[float(x), float(y)] for x, y in
                           (poly.tolist() if hasattr(poly, "tolist") else poly)]
                    if not pts:
                        continue
                    cx = sum(p[0] for p in pts) / len(pts)
                    cy = sum(p[1] for p in pts) / len(pts)
                    parent = None
                    for pb in person_boxes:
                        bx1, by1, bx2, by2 = pb["bbox"]
                        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                            parent = pb
                            break
                    if parent is None:
                        continue
                    new_masks.append({
                        "label":        parent["label_full"],
                        "group_id":     parent["track_id"],
                        "description":  "",
                        "shape_type":   "polygon",
                        "flags":        {},
                        "mask":         None,
                        "points":       pts,
                        "skeleton_name": None,
                    })

    data["shapes"] = non_skeleton_shapes + new_skeletons + new_masks
    with open(json_path, "w", encoding="utf-8") as f:
        _json_mod.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[REINFER] {len(new_skeletons)} skeleton(s) générés depuis {len(person_boxes)} box(es)."
          + (f" {len(new_masks)} mask(s) régénéré(s)." if seg_enabled else ""))
    return data["shapes"]
