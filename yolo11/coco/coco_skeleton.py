# yolo11/coco/coco_skeleton.py
import json
from pathlib import Path

_spec_path = Path(__file__).parent.parent.parent / "coco_specs" / "human_skeleton.json"
with open(_spec_path, encoding="utf-8") as _f:
    _spec = json.load(_f)

COCO_SKELETON = [tuple(pair) for pair in _spec["skeleton"]]