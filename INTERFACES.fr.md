# LabelMe 5.9.1-Yolo-DWPose — Interfaces d'intégration

Ce document décrit les points d'intégration entre LabelMe et le pipeline YOLO11/DWPose. Il s'adresse aux développeurs qui souhaitent :
- Remplacer YOLO11 ou DWPose par un modèle plus puissant ou différent
- Construire leur propre pipeline d'annotation sur LabelMe
- Étendre ou personnaliser le pipeline existant

---

## 1. Points d'ancrage LabelMe ↔ Pipeline (`labelme/app.py`)

### 1.1 Actions de menu

Toutes les actions du pipeline sont enregistrées dans `MainWindow._init_actions()` et ajoutées au menu `Edit` :

```python
# labelme/app.py — _init_actions()
self.actions.yolo11_file          # Edit → YOLO11 — Fichier
self.actions.yolo11_dir           # Edit → YOLO11 — Dossier
self.actions.yolo11_settings      # Edit → Réglages YOLO11
self.actions.reinfer_dwpose       # Edit → Réinférer DWPose depuis les boxes
self.actions.export_openpose      # Edit → Exporter PNG OpenPose
self.actions.export_openpose_batch # Edit → Exporter dataset OpenPose — dossier
```

Pour ajouter votre propre action, suivez le même pattern dans `_init_actions()` et `_init_menus()`.

### 1.2 Chargement des shapes — `_load_shape_dicts()`

Lors du chargement d'un JSON, cette méthode convertit les dicts bruts en objets `Shape`. Le bloc skeleton résout `skeleton_name` et peuple `skeleton_meta` :

```python
# labelme/app.py — _load_shape_dicts()
if shape_dict["shape_type"] == "skeleton":
    sk_name = shape_dict.get("skeleton_name")  # lu depuis le JSON en priorité
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

Pour supporter un nouveau format skeleton : ajoutez sa résolution ici et créez `coco_specs/{name}_skeleton.json`.

### 1.3 Sauvegarde des shapes — `format_shape()`

```python
# labelme/app.py — format_shape()
if s.shape_type == "skeleton":
    data["skeleton_name"] = getattr(s, "skeleton_name", "human")
```

Toute nouvelle métadonnée produite par votre pipeline doit être sérialisée ici.

### 1.4 Imports directs du pipeline dans app.py

```python
from labelme.yolo11_settings   import Yolo11SettingsDialog, get_current_settings
from labelme.yolo11_reinfer    import reinfer_from_boxes
from labelme.openpose_exporter import export_dataset, export_dataset_batch, export_openpose_png
```

Pour remplacer ou étendre le pipeline, remplacez ces imports par vos propres modules — le reste de `app.py` n'a pas besoin de changer.

---

## 2. Points d'ancrage des modèles (`yolo11/model.py`)

### 2.1 Registre des modèles — `TASK_MODEL_MAP`

```python
# yolo11/model.py
TASK_MODEL_MAP = {
    "boxes":        "yolo11n.pt",
    "segmentation": "yolo11n-seg.pt",
    "skeletons":    "yolo11n-pose.pt",
}
```

Les fichiers modèles sont dans `yolo11/models/originals/`. Pour utiliser un modèle plus grand ou personnalisé, remplacez le nom de fichier ici et placez votre modèle dans le même dossier. Par exemple :

```python
TASK_MODEL_MAP = {
    "boxes":        "yolo11x.pt",       # modèle plus grand
    "segmentation": "yolo11x-seg.pt",
    "skeletons":    "yolo11x-pose.pt",
}
```

### 2.2 Cache des modèles — `load_model()` / `get_model()`

```python
load_model(tasks: list)       # précharge un ou plusieurs modèles — ["boxes", "segmentation", "skeletons"]
get_model(task: str) -> model # récupère un modèle en cache par nom de tâche
```

Pour remplacer YOLO11 par un autre détecteur (RT-DETR, YOLOv9, modèle custom) :
1. Implémentez un wrapper exposant la même interface qu'un modèle Ultralytics (`.predict()` retournant boxes/masks/keypoints)
2. Enregistrez-le dans `load_model()` et `get_model()`
3. Aucun autre fichier n'a besoin de changer

### 2.3 Modèle DWPose — `get_dwpose()`

```python
get_dwpose() -> DWposeDetectorRaw
```

Le fichier modèle DWPose ONNX est `yolo11/models/originals/dw-ll_ucoco_384.onnx`. Pour utiliser un modèle de pose différent :
1. Implémentez une classe exposant la même interface que `DWposeDetectorRaw` (voir section 3)
2. Remplacez l'instanciation dans `get_dwpose()`

---

## 3. Interface du détecteur DWPose (`yolo11/dwpose_wrapper.py`)

```python
class DWposeDetectorRaw:
    def detect_raw(
        self,
        frame_bgr: np.ndarray,   # image complète, BGR
        boxes: np.ndarray,        # (N, 4) xyxy boxes personnes — déjà tournées si rotation active
        skeleton_format: str,     # "openpose" (134 pts) ou "coco" (133 pts)
    ) -> dict:
        # Retourne :
        # {
        #   "keypoints": np.ndarray (N, K, 2),  # x, y en pixels, dans le repère tourné
        #   "scores":    np.ndarray (N, K),
        # }
```

**Pour remplacer DWPose** par un autre estimateur de pose corps entier (MMPose, MediaPipe, ViTPose, etc.) :
1. Implémentez une classe avec la même signature `detect_raw()`
2. Enregistrez-la dans `model.py → get_dwpose()`
3. Si votre modèle sort un nombre ou un ordre de keypoints différent, mettez à jour `_OFFSETS` dans `inference.py` et créez le `coco_specs/{name}_skeleton.json` correspondant

---

## 4. Spécification des formats skeleton (`coco_specs/`)

Chaque format skeleton est décrit par un fichier JSON :

```
coco_specs/
  human_skeleton.json               # 17 pts — sortie YOLO-pose
  coco_fullbody_skeleton.json       # 133 pts — DWPose COCO-WholeBody
  openpose_fullbody_skeleton.json   # 134 pts — DWPose OpenPose
```

Structure :
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

Pour ajouter un nouveau format skeleton, créez `coco_specs/{name}_skeleton.json` et enregistrez `{name}` dans :

| Fichier | Où |
|---------|-----|
| `inference.py` | `_skeleton_format_for()` et `_OFFSETS` |
| `main.py` | dict `_SKELETON_NAME` |
| `shape.py` | `_FEET_CANONICAL_IDS` (si le format inclut des keypoints pieds) |
| `app.py` | heuristique de repli dans `_load_shape_dicts()` |

---

## 5. Export OpenPose (`labelme/openpose_exporter.py`)

Le rendu PNG est une seule fonction :

```python
def render_openpose_png(shapes: list, width: int, height: int) -> np.ndarray:
    """
    Dessine les shapes skeleton + silhouettes mask sur un canvas noir.
    shapes : liste d'objets Shape-like
             attributs requis : shape_type, points, label, group_id, skeleton_name
    Retourne : tableau numpy BGR (fond noir, skeleton arc-en-ciel, silhouette grise)
    """
```

Pour personnaliser le rendu (couleurs, épaisseur des lignes, overlays supplémentaires) :
- Palette couleurs : `_RAINBOW` (connexions body) et `_GROUP_PT_COLOR` (pieds/mains/visage)
- Silhouette : `_MASK_FILL_COLOR_BGR` et `_MASK_ALPHA`

---

## 6. Interface des réglages (`labelme/yolo11_settings.py`)

Les réglages du pipeline sont lus/écrits via :

```python
get_current_settings() -> dict
```

Exemple de sortie :
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

Les réglages sont persistés dans `yolo11/ui_settings.json`. Un pipeline de remplacement peut lire ce dict directement ou utiliser sa propre configuration.

---

## 7. Résumé — fichiers à modifier selon le cas d'usage

| Objectif | Fichiers à modifier |
|----------|-------------------|
| Utiliser un modèle YOLO plus grand | `yolo11/model.py` (`TASK_MODEL_MAP`), `download_models.py` |
| Remplacer le détecteur YOLO11 | `yolo11/model.py` (`load_model`, `get_model`) |
| Remplacer DWPose | `yolo11/dwpose_wrapper.py`, `yolo11/model.py` (`get_dwpose`) |
| Ajouter un nouveau format skeleton | `coco_specs/{name}_skeleton.json`, `inference.py`, `main.py`, `shape.py`, `app.py` |
| Ajouter une action dans le menu LabelMe | `labelme/app.py` (`_init_actions`, `_init_menus`) |
| Personnaliser le rendu PNG OpenPose | `labelme/openpose_exporter.py` |
| Modifier l'UI des réglages | `labelme/yolo11_settings.py` |
| Modifier la sérialisation JSON | `labelme/app.py` (`format_shape`, `_load_shape_dicts`) |
