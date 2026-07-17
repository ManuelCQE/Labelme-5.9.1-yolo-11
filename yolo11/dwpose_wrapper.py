# yolo11/dwpose_wrapper.py
import cv2
import numpy as np
import onnxruntime as ort
from controlnet_dwpose.onnxpose import inference_pose


# Réindexation COCO-WholeBody -> OpenPose 18pts
# Insère le Neck (= milieu des épaules) à la position 17, puis réordonne 0-17.
_MMPOSE_IDX  = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
_OPENPOSE_IDX = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]


def _remap_coco_to_openpose(keypoints: np.ndarray, scores: np.ndarray):
    """
    Convertit la sortie brute COCO-WholeBody (ordre natif RTMPose)
    vers l'ordre OpenPose 18pts (avec Neck calculé = milieu des épaules).

    Args:
        keypoints : (N, K, 2)
        scores    : (N, K)

    Returns:
        keypoints : (N, K+1, 2) — Neck inséré en position 17
        scores    : (N, K+1)
    """
    n = keypoints.shape[0]
    if n == 0:
        return keypoints, scores

    keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)  # (N,K,3)

    # Neck = milieu des épaules (indices COCO 5=left_shoulder, 6=right_shoulder)
    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)  # (N,3)
    neck_visible = np.logical_and(
        keypoints_info[:, 5, 2] > 0.3,
        keypoints_info[:, 6, 2] > 0.3,
    ).astype(float)
    neck[:, 2] = neck_visible

    # Insère Neck à l'indice 17 → (N, K+1, 3)
    new_info = np.insert(keypoints_info, 17, neck, axis=1)

    # Réindexation body 0-17 : COCO-natif -> OpenPose
    # Copie obligatoire avant remapping pour éviter les effets de bord
    # (certaines positions sources sont aussi destinations)
    source = new_info.copy()
    new_info[:, _OPENPOSE_IDX] = source[:, _MMPOSE_IDX]

    keypoints_out = new_info[..., :2]
    scores_out    = new_info[..., 2]
    return keypoints_out, scores_out


class DWposeDetectorRaw:
    """
    Pose estimator DWPose (ONNX) — utilise les boxes YOLO existantes,
    pas de détecteur interne.
    Nécessite uniquement : dw-ll_ucoco_384.onnx

    Deux formats de sortie via le paramètre `skeleton_format` de detect_raw() :
      - "coco"     : sortie brute du modèle, ordre natif COCO-WholeBody.
                     133pts : body=0-16 (COCO-17 sans Neck), feet=17-22,
                     face=23-90, left_hand=91-111, right_hand=112-132.
                     Correspond à coco_specs/coco_fullbody_skeleton.json.

      - "openpose" : sortie remappée avec Neck inséré et réindexation OpenPose.
                     134pts : body=0-17 (OpenPose-18 avec Neck), feet=18-23,
                     face=24-91, left_hand=92-112, right_hand=113-133.
                     Correspond à coco_specs/openpose_fullbody_skeleton.json.
    """

    def __init__(self, model_pose: str, device: str = "cpu"):
        providers = (["CUDAExecutionProvider"] if device != "cpu"
                     else ["CPUExecutionProvider"])
        self.session_pose = ort.InferenceSession(model_pose, providers=providers)

    def detect_raw(self, frame_bgr: np.ndarray, boxes: np.ndarray,
                   skeleton_format: str = "openpose") -> dict:
        """
        Retourne les keypoints en coordonnées pixels.

        Args:
            frame_bgr       : frame OpenCV BGR
            boxes           : np.ndarray (N, 4) xyxy — boxes YOLO déjà détectées
            skeleton_format : "openpose" (défaut, 134pts) ou "coco" (133pts brut)

        Retourne :
            {
              "keypoints": np.ndarray (N, K, 2),  # x, y en pixels
              "scores":    np.ndarray (N, K),      # confiances
            }
            K = 133 si skeleton_format="coco"
            K = 134 si skeleton_format="openpose"
        """
        if skeleton_format not in ("coco", "openpose"):
            raise ValueError(
                f"skeleton_format inconnu : {skeleton_format!r} "
                "(attendu 'coco' ou 'openpose')"
            )

        if len(boxes) == 0:
            k = 134 if skeleton_format == "openpose" else 133
            return {
                "keypoints": np.zeros((0, k, 2)),
                "scores":    np.zeros((0, k)),
            }

        keypoints, scores = inference_pose(self.session_pose, boxes, frame_bgr)

        if skeleton_format == "openpose":
            keypoints, scores = _remap_coco_to_openpose(keypoints, scores)

        return {"keypoints": keypoints, "scores": scores}

    def render_png(self, frame_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Retourne le PNG OpenPose rendu (fond noir, style ComfyUI/ControlNet).
        Utilise draw_pose de controlnet_dwpose.

        Toujours en format "openpose" (Neck requis par draw_pose/limbSeq).
        Les boxes doivent être passées explicitement (np.ndarray (N,4) xyxy).

        Retourne : np.ndarray BGR, même taille que frame_bgr.
        """
        from controlnet_dwpose.util import draw_pose
        H, W = frame_bgr.shape[:2]

        result    = self.detect_raw(frame_bgr, boxes, skeleton_format="openpose")
        keypoints = result["keypoints"]
        scores    = result["scores"]

        if len(keypoints) == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)

        # candidate/subset attendus par draw_pose : body 0-17 (OpenPose 18pts avec Neck)
        candidate    = keypoints[:, :18].reshape(-1, 2)
        subset_score = scores[:, :18]
        subset = np.full_like(subset_score, -1, dtype=float)
        for i in range(len(subset_score)):
            for j in range(18):
                if subset_score[i, j] > 0.3:
                    subset[i, j] = 18 * i + j

        bodies = dict(candidate=candidate, subset=subset, score=subset_score)

        # Format openpose 134pts : feet=18-23, face=24-91, hands=92-133
        hands       = np.vstack([keypoints[:, 92:113],  keypoints[:, 113:134]])
        faces       = keypoints[:, 24:92]
        hands_score = np.vstack([scores[:, 92:113], scores[:, 113:134]])
        faces_score = scores[:, 24:92]

        pose = dict(
            bodies=bodies,
            hands=hands,       hands_score=hands_score,
            faces=faces,       faces_score=faces_score,
        )

        rendered = draw_pose(pose, H, W)
        rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
        return cv2.resize(rendered, (W, H), interpolation=cv2.INTER_LINEAR)