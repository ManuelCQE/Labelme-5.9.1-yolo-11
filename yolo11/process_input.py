# yolo11/process_input.py
from pathlib import Path
from .config import load_config
from .inference import predict

_cfg = load_config()

def process_path(path, task=None, target_fps=None, cfg=None):
    if cfg is None:
        cfg = load_config()
    if task is None:
        task = cfg["tasks"]
    return predict(str(path), tasks=task, target_fps=target_fps, cfg=cfg)
