import yaml
from pathlib import Path

DEFAULT_CONFIG = {
    "tasks": ["boxes"],
    "classes": [],
    "video": {
        "target_fps": None,
        "draw_annotations": True,
        "show_live": True,
        "window_name": "YOLO11 Preview"
    },
    "model": {
        "path": "yolo11_pretrained.pt",
        "fallback": "yolov11n.pt"
    },
    "output": {
        "processed_dirname": "processed"
    }
}

def load_config(config_path=None):
    """
    Charge le config.yaml si trouvé (par defaut cherche yolo11/config.yaml à côté du module).
    Retourne un dict complet (avec valeurs par defaut si manquantes).
    """
    if config_path is None:
        # default file relative to this script
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    cfg = DEFAULT_CONFIG.copy()
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        # simple shallow merge for top-level keys
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg
