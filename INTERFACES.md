# LabelMe 5.9.1-Yolo-DWPose — Integration Interfaces

This document describes the integration points between LabelMe and the YOLO11/DWPose pipeline. It is intended for developers who want to:
- Replace YOLO11 or DWPose with a stronger or different model
- Build their own annotation pipeline on top of LabelMe
- Extend or customize the existing pipeline

---

## 1. LabelMe ↔ Pipeline hook points (`labelme/app.py`)

### 1.1 Menu actions

All pipeline actions are registered in `MainWindow._init_actions()` and added to the `Edit` menu:

```python
# labelme/app.py — _init_actions()
self.actions.yolo11_file    # Edit → YOLO11 — Fichier
self.actions.yolo11_dir     # Edit → YOLO11 — Dossier
self.actions.yolo11_settings  # Edit → Réglages YOLO11
self.actions.reinfer_dwpose   # Edit → Réinférer DWPose depuis les boxes
self.actions.export_openpose  # Edit → Exporter PNG OpenPose
self.actions.export_openpose_batch  # Edit → Exporter dataset OpenPose — dossier
```

To add your own action, follow the same pattern in `_init_actions()` and `_init_menus()`.

### 1.2 Shape loading — `_load_shape_dicts()`

When a JSON is loaded, this method converts raw dicts into `Shape` objects. The skeleton-specific block resolves `skeleton_name` and populates `skeleton_meta`:

```python
# labelme/app.py — _load_shape_dicts()
if shape_dict["shape_type"] == "skeleton":
    sk_name = shape_dict.get("skeleton_name")  # read from JSON first
    if sk_name:
        shape.skeleton_name = sk_name
    elif n == 17:
        shape.skeleton_name = "human"
    elif n == 134:
        shape.skeleton_name = "openpose_fullbody"
    else:
        shape.skeleton_name = "coco_fullbody"
    shape.other_data["skeleton_meta"] = shape_dict["points"]  # [id, x, y, conf]
```

To support a new skeleton format: add its `skeleton_name` resolution here and create `coco_specs/{name}_skeleton.json`.

### 1.3 Shape saving — `format_shape()`

```python
# labelme/app.py — format_shape()
if s.shape_type == "skeleton":
    data["skeleton_name"] = getattr(s, "skeleton_name", "human")
```

Any new metadata your pipeline produces must be serialized here.

### 1.4 Direct imports from pipeline into app.py

```python
from labelme.yolo11_settings  import Yolo11SettingsDialog, get_current_settings
from labelme.yolo11_reinfer   import reinfer_from_boxes
from labelme.openpose_exporter import export_dataset, export_dataset_batch, export_openpose_png
```

To replace or extend the pipeline, swap these imports for your own modules — the rest of `app.py` does not need to change.

---

## 2. Model anchor points (`yolo11/model.py`)

### 2.1 Model registry — `TASK_MODEL_MAP`

```python
# yolo11/model.py
TASK_MODEL_MAP = {
    "boxes":        "yolo11n.pt",
    "segmentation": "yolo11n-seg.pt",
    "skeletons":    "yolo11n-pose.pt",
}
```

Model files are stored in `yolo11/models/originals/`. To use a larger or custom YOLO model, replace the filename here and put your model file in the same folder. For example:

```python
TASK_MODEL_MAP = {
    "boxes":        "yolo11x.pt",       # larger model
    "segmentation": "yolo11x-seg.pt",
    "skeletons":    "yolo11x-pose.pt",
}
```

### 2.2 Model cache — `load_model()` / `get_model()`

```python
load_model(tasks: list)       # preload one or several models — ["boxes", "segmentation", "skeletons"]
get_model(task: str) -> model # retrieve a cached model by task name
```

To replace YOLO11 with a different detector (RT-DETR, YOLOv9, custom model):
1. Implement a wrapper exposing the same interface as an Ultralytics model (`.predict()` returning boxes/masks/keypoints)
2. Register it in `load_model()` and `get_model()`
3. No other file needs to change

### 2.3 DWPose model — `get_dwpose()`

```python
get_dwpose() -> DWposeDetectorRaw
```

The DWPose ONNX model file is `yolo11/models/originals/dw-ll_ucoco_384.onnx`. To use a different pose model:
1. Implement a class exposing the same interface as `DWposeDetectorRaw` (see section 3)
2. Swap the instantiation in `get_dwpose()`

---

## 3. DWPose detector interface (`yolo11/dwpose_wrapper.py`)

```python
class DWposeDetectorRaw:
    def detect_raw(
        self,
        frame_bgr: np.ndarray,   # full image, BGR
        boxes: np.ndarray,        # (N, 4) xyxy person boxes — already rotated if rotation active
        skeleton_format: str,     # "openpose" (134 pts) or "coco" (133 pts)
    ) -> dict:
        # Returns:
        # {
        #   "keypoints": np.ndarray (N, K, 2),  # x, y in pixels, in rotated frame
        #   "scores":    np.ndarray (N, K),
        # }
```

**To replace DWPose** with another whole-body pose estimator (MMPose, MediaPipe, ViTPose, etc.):
1. Implement a class with the same `detect_raw()` signature
2. Register it in `model.py → get_dwpose()`
3. If your model outputs a different keypoint count or order, update `_OFFSETS` in `inference.py` and create the matching `coco_specs/{name}_skeleton.json`

---

## 4. Skeleton format specification (`coco_specs/`)

Each skeleton format is described by a JSON file:

```
coco_specs/
  human_skeleton.json               # 17 pts — YOLO-pose output
  coco_fullbody_skeleton.json       # 133 pts — DWPose COCO-WholeBody
  openpose_fullbody_skeleton.json   # 134 pts — DWPose OpenPose
```

Structure:
```json
{
  "keypoints": ["nose", "neck", "..."],
  "connections": {
    "body":       [{"from": 0, "to": 1}, "..."],
    "feet":       [{"from": 18, "to": 19}, "..."],
    "left_hand":  ["..."],
    "right_hand": ["..."],
    "face":       ["..."]
  },
  "groups": {
    "body":  {"start": 0,  "end": 17},
    "feet":  {"start": 18, "end": 23},
    "face":  {"start": 24, "end": 91},
    "hands": {"start": 92, "end": 133}
  }
}
```

To add a new skeleton format, create `coco_specs/{name}_skeleton.json` and register `{name}` in:

| File | Where |
|------|-------|
| `inference.py` | `_skeleton_format_for()` and `_OFFSETS` |
| `main.py` | `_SKELETON_NAME` dict |
| `shape.py` | `_FEET_CANONICAL_IDS` (if the format includes feet keypoints) |
| `app.py` | `_load_shape_dicts()` fallback heuristic |

---

## 5. OpenPose export (`labelme/openpose_exporter.py`)

The PNG renderer is a single function:

```python
def render_openpose_png(shapes: list, width: int, height: int) -> np.ndarray:
    """
    Renders skeleton shapes + silhouette masks onto a black canvas.
    shapes : list of Shape-like objects
             required attributes: shape_type, points, label, group_id, skeleton_name
    Returns: BGR numpy array (black background, rainbow skeleton, grey silhouette)
    """
```

To customize the render (colors, line thickness, additional overlays):
- Color palette: `_RAINBOW` (body connections) and `_GROUP_PT_COLOR` (feet/hands/face)
- Silhouette: `_MASK_FILL_COLOR_BGR` and `_MASK_ALPHA`

---

## 6. Settings interface (`labelme/yolo11_settings.py`)

Pipeline settings are read/written via:

```python
get_current_settings() -> dict
```

Example output:
```json
{
    "tasks":    {"boxes": true, "seg": true, "skeletons": true},
    "mode":     "dwpose",
    "skeleton": "openpose",
    "groups":   {"body": true, "feet": true, "face": false, "hands": true},
    "min_conf": 0.3,
    "nms_iou":  0.5,
    "fps":      1.0,
    "rotation": 0
}
```

Settings are persisted in `yolo11/ui_settings.json`. A pipeline replacement can read this dict directly or use its own config.

---

## 7. Summary — files to touch per use case

| Goal | Files to modify |
|------|----------------|
| Use a larger YOLO model | `yolo11/model.py` (`TASK_MODEL_MAP`), `download_models.py` |
| Replace YOLO11 detector | `yolo11/model.py` (`load_model`, `get_model`) |
| Replace DWPose | `yolo11/dwpose_wrapper.py`, `yolo11/model.py` (`get_dwpose`) |
| Add a new skeleton format | `coco_specs/{name}_skeleton.json`, `inference.py`, `main.py`, `shape.py`, `app.py` |
| Add a LabelMe menu action | `labelme/app.py` (`_init_actions`, `_init_menus`) |
| Customize OpenPose PNG render | `labelme/openpose_exporter.py` |
| Change pipeline settings UI | `labelme/yolo11_settings.py` |
| Change JSON serialization | `labelme/app.py` (`format_shape`, `_load_shape_dicts`) |
