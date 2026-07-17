# yolo11/model.py
from pathlib import Path
from ultralytics import YOLO
from .config import load_config

_cfg = load_config()
_model_cache = {}

TASK_MODEL_MAP = {
    "boxes":        "yolo11n.pt",
    "segmentation": "yolo11n-seg.pt",
    "skeletons":    "yolo11n-pose.pt",
}

BASE_DIR            = Path(__file__).parent
ORIGINAL_MODELS_DIR = BASE_DIR / "models" / "originals"


def _get_model_path(task):
    model_name = TASK_MODEL_MAP.get(task)
    if model_name is None:
        raise ValueError(f"Tâche inconnue ou non-YOLO : {task}")
    model_path = ORIGINAL_MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle absent : {model_path}")
    return str(model_path)


def load_model(tasks=None):
    global _model_cache
    if tasks is None:
        tasks = _cfg.get("tasks", ["boxes"])

    loaded = {}
    for task in tasks:
        if task in _model_cache:
            loaded[task] = _model_cache[task]
            continue
        if task == "dwpose":
            loaded["dwpose"] = get_dwpose()
        else:
            model_path = _get_model_path(task)
            print(f"[MODEL] Chargement {task} : {model_path}")
            model = YOLO(model_path)
            _model_cache[task] = model
            loaded[task] = model

    return loaded


def get_model(task):
    global _model_cache
    if task not in _model_cache:
        load_model([task])
    return _model_cache[task]


def get_dwpose():
    """Retourne le détecteur DWPose (ONNX, un seul modèle)."""
    global _model_cache
    if "dwpose" not in _model_cache:
        from .dwpose_wrapper import DWposeDetectorRaw
        model_pose = ORIGINAL_MODELS_DIR / "dw-ll_ucoco_384.onnx"
        if not model_pose.exists():
            raise FileNotFoundError(
                f"Modèle DWPose absent : {model_pose}\n"
                "Télécharger depuis : "
                "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"
            )
        print(f"[MODEL] Chargement DWPose : {model_pose}")
        _model_cache["dwpose"] = DWposeDetectorRaw(str(model_pose))
        print("[MODEL] DWPose prêt.")
    return _model_cache["dwpose"]