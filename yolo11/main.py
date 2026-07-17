# yolo11/main.py
import argparse
import json
from .config import load_config
from .model import load_model
from .process_input import process_path

# ---------------------------------------------------------------------------
# Résolution skeleton_name depuis (mode, skeleton)
#
#   mode=yolo                          → skeleton_name="human"       (toujours)
#   mode=dwpose + skeleton=openpose    → skeleton_name="openpose_fullbody"
#   mode=dwpose + skeleton=coco        → skeleton_name="coco_fullbody"
#
# --groups sans --mode → implique mode=dwpose silencieusement
# --mode yolo + --skeleton → ignoré avec warning
# --mode yolo + --groups  → erreur
# ---------------------------------------------------------------------------

DWPOSE_GROUPS  = ["body", "feet", "face", "hands"]
VALID_BACKENDS = ["yolo", "dwpose"]
VALID_SKELETONS = ["coco", "openpose"]   # uniquement pour DWPose

# skeleton_name utilisé dans cfg selon (mode, skeleton)
_SKELETON_NAME = {
    ("yolo",   None):       "human",
    ("yolo",   "coco"):     "human",          # --skeleton ignoré pour yolo
    ("yolo",   "openpose"): "human",          # idem
    ("dwpose", None):       "openpose_fullbody",   # défaut dwpose = openpose
    ("dwpose", "openpose"): "openpose_fullbody",
    ("dwpose", "coco"):     "coco_fullbody",
}


def run(path, tasks=None, target_fps=None, mode=None, skeleton=None,
        groups=None, min_conf=None, nms_iou=None, rotation=None):
    """
    Fonction réutilisable — appelable depuis LabelMe ou en script.

    Args:
        path       : chemin vers une vidéo, une image ou un dossier
        tasks      : liste de tâches ex: ["boxes", "skeletons"]
        target_fps : fps cible (None = valeur config.yaml)
        mode       : "yolo" ou "dwpose" (None = config.yaml ou inféré depuis groups)
        skeleton   : "coco" ou "openpose" — uniquement pour DWPose
                     (None = "openpose" si dwpose, ignoré si yolo)
        groups     : groupes DWPose ex: ["body", "hands"]
                     (ignoré si mode=yolo ; implique mode=dwpose si mode absent)
        min_conf   : seuil de confiance minimal pour garder un keypoint
        nms_iou    : seuil IoU pour dédupliquer les boxes person avant DWPose
    """
    cfg = load_config()

    if tasks is None:
        tasks = cfg.get("tasks", ["boxes"])
    if target_fps is None:
        target_fps = cfg.get("video", {}).get("target_fps", None)

    # --- Résolution du mode ---
    if mode is None:
        if groups is not None:
            # --groups sans --mode → dwpose implicite
            mode = "dwpose"
        else:
            mode = cfg.get("pose_backend", "yolo")

    if mode not in VALID_BACKENDS:
        raise ValueError(f"Mode inconnu : {mode!r}. Choix : {VALID_BACKENDS}")

    # --- Validation croisée ---
    if mode == "yolo" and groups is not None:
        raise ValueError("--groups n'est pas compatible avec --mode yolo (YOLO produit toujours 17pts COCO).")

    if mode == "yolo" and skeleton is not None:
        print(f"[WARNING] --skeleton {skeleton!r} ignoré : sans effet avec --mode yolo.")
        skeleton = None

    if skeleton is not None and skeleton not in VALID_SKELETONS:
        raise ValueError(f"Skeleton inconnu : {skeleton!r}. Choix : {VALID_SKELETONS}")

    # --- Résolution skeleton_name ---
    skeleton_name = _SKELETON_NAME.get((mode, skeleton), "openpose_fullbody")
    cfg["pose_backend"]  = mode
    cfg["skeleton_name"] = skeleton_name

    # --- Groupes DWPose ---
    if mode == "dwpose":
        if groups is not None:
            invalid = set(groups) - set(DWPOSE_GROUPS)
            if invalid:
                raise ValueError(f"Groupes inconnus : {invalid}. Choix : {DWPOSE_GROUPS}")
            cfg["dwpose_groups"] = {g: (g in groups) for g in DWPOSE_GROUPS}
        # sinon on garde les dwpose_groups du config.yaml

    # --- Conf / IoU ---
    if min_conf is not None:
        cfg["min_conf"] = min_conf
    if nms_iou is not None:
        cfg["nms_iou"] = nms_iou
    if rotation is not None:
        cfg["rotation"] = int(rotation) % 360

    load_model(tasks)

    # --- Log de démarrage ---
    print(f"[YOLO11] Traitement  : {path}")
    print(f"[YOLO11] Tâches      : {tasks}")
    print(f"[YOLO11] FPS cible   : {target_fps}")
    print(f"[YOLO11] Mode        : {mode}")
    print(f"[YOLO11] Skeleton    : {skeleton_name}")
    if mode == "dwpose":
        active = [g for g, v in cfg.get("dwpose_groups", {}).items() if v]
        print(f"[YOLO11] Groupes     : {active}")
    print(f"[YOLO11] Min conf    : {cfg.get('min_conf', 0.3)}")
    if cfg.get("rotation", 0):
        print(f"[YOLO11] Rotation    : {cfg['rotation']}°")
    if cfg.get("nms_iou") is not None:
        print(f"[YOLO11] NMS IoU     : {cfg.get('nms_iou')}")

    return process_path(path, task=tasks, target_fps=target_fps, cfg=cfg)


def main():
    """Point d'entrée CLI."""
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="yolo11 — pipeline annotation YOLO + DWPose",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
exemples :
  python -m yolo11.main video.mp4                              # yolo, COCO 17pts
  python -m yolo11.main video.mp4 --mode dwpose               # dwpose, openpose body
  python -m yolo11.main video.mp4 --groups body,hands         # dwpose implicite, openpose
  python -m yolo11.main video.mp4 --mode dwpose --skeleton coco --groups body,face
        """
    )

    parser.add_argument("path",
                        help="image / dossier / vidéo")

    parser.add_argument("--tasks",
                        default=",".join(cfg.get("tasks", ["boxes"])),
                        help="tâches séparées par virgule : boxes,segmentation,skeletons\n"
                             "(défaut : depuis config.yaml)")

    parser.add_argument("--fps",
                        type=float,
                        default=cfg.get("video", {}).get("target_fps", None),
                        metavar="N",
                        help="fps cible pour le sampling vidéo (défaut : depuis config.yaml)")

    parser.add_argument("--mode",
                        default=None,
                        choices=VALID_BACKENDS,
                        help="moteur pose : yolo (COCO 17pts) ou dwpose\n"
                             "(défaut : yolo, ou dwpose si --groups est présent)")

    parser.add_argument("--skeleton",
                        default=None,
                        choices=VALID_SKELETONS,
                        help="format skeleton DWPose : coco (133pts) ou openpose (134pts)\n"
                             "(défaut : openpose — ignoré si --mode yolo)")

    parser.add_argument("--groups",
                        default=None,
                        metavar="body,feet,face,hands",
                        help="groupes DWPose à activer, séparés par virgule\n"
                             "(implique --mode dwpose si absent ; ignoré si --mode yolo)")

    parser.add_argument("--min-conf",
                        type=float,
                        default=None,
                        metavar="F",
                        help="seuil de confiance minimal pour garder un keypoint\n"
                             "(défaut : depuis config.yaml, typiquement 0.3)")

    parser.add_argument("--nms-iou",
                        type=float,
                        default=None,
                        metavar="F",
                        help="seuil IoU NMS pour dédupliquer les boxes person avant DWPose\n"
                             "(défaut : depuis config.yaml)")

    parser.add_argument("--output",
                        type=str,
                        default=None,
                        metavar="FILE",
                        help="sauvegarder le résumé JSON dans ce fichier (optionnel)")

    args   = parser.parse_args()
    tasks  = [t.strip() for t in args.tasks.split(",") if t.strip()]
    groups = [g.strip() for g in args.groups.split(",") if g.strip()] if args.groups else None

    res = run(
        args.path,
        tasks=tasks,
        target_fps=args.fps,
        mode=args.mode,
        skeleton=args.skeleton,
        groups=groups,
        min_conf=args.min_conf,
        nms_iou=args.nms_iou,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"[YOLO11] Résultats sauvegardés dans {args.output}")
    else:
        print(res)


if __name__ == "__main__":
    main()