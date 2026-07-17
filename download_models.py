"""
download_models.py
Télécharge les modèles nécessaires au pipeline YOLO11 + DWPose.
Appelé par install.bat lors de la première installation.
"""
import urllib.request
from pathlib import Path

BASE_DIR  = Path(__file__).parent
MODELS_DIR = BASE_DIR / "yolo11" / "models" / "originals"

MODELS = [
    {
        "name": "yolo11n.pt",
        "url":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    },
    {
        "name": "yolo11n-seg.pt",
        "url":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
    },
    {
        "name": "yolo11n-pose.pt",
        "url":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt",
    },
    {
        "name": "dw-ll_ucoco_384.onnx",
        "url":  "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
    },
    {
        "name": "yolox_l.onnx",
        "url":  "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
    },
]

def _progress(count, block_size, total_size):
    if total_size > 0:
        pct = min(int(count * block_size * 100 / total_size), 100)
        print(f"\r    {pct}%", end="", flush=True)

def download(name, url):
    dest = MODELS_DIR / name
    if dest.exists():
        print(f"[OK] {name} déjà présent, skip.")
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[...] Téléchargement {name}...")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print(f"\n[OK] {name} téléchargé.")
    except Exception as e:
        print(f"\n[ERREUR] {name} : {e}")

if __name__ == "__main__":
    print("=== Téléchargement des modèles ===")
    for m in MODELS:
        download(m["name"], m["url"])
    print("\n=== Terminé ===")
