import copy

import numpy as np
import skimage.measure
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui

import labelme.utils

import json
from pathlib import Path

def _load_skeleton_spec(name="human"):
    spec_path = Path(__file__).parent.parent / "coco_specs" / f"{name}_skeleton.json"
    with open(spec_path, encoding="utf-8") as f:
        data = json.load(f)

    if "skeleton" in data:
        return data["skeleton"]

    # Format dwpose_skeleton.json : connections dict par groupe — ne prendre que body
    connections = data.get("connections", [])
    if isinstance(connections, dict):
        return [[c["from"], c["to"]] for c in connections.get("body", [])]
    else:
        return [[c["from"], c["to"]] for c in connections]

_SKELETON_CACHE = {}  # cache pour éviter de relire le fichier à chaque frame
_KEYPOINT_NAMES_CACHE = {}  # cache index → nom keypoint par skeleton

def _get_skeleton(name="human"):
    if name not in _SKELETON_CACHE:
        _SKELETON_CACHE[name] = _load_skeleton_spec(name)
    return _SKELETON_CACHE[name]

def _load_skeleton_keypoint_names(name="human"):
    """Retourne un dict {index: keypoint_name} pour un skeleton donné."""
    spec_path = Path(__file__).parent.parent / "coco_specs" / f"{name}_skeleton.json"
    try:
        with open(spec_path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    keypoints = data.get("keypoints", [])
    if isinstance(keypoints, list):
        return {kp["id"]: kp["name"] for kp in keypoints if "id" in kp and "name" in kp}
    return {}

def get_keypoint_name(skeleton_name, index):
    """Retourne le nom du keypoint à l'index donné, ou None si inconnu."""
    if skeleton_name not in _KEYPOINT_NAMES_CACHE:
        _KEYPOINT_NAMES_CACHE[skeleton_name] = _load_skeleton_keypoint_names(skeleton_name)
    names = _KEYPOINT_NAMES_CACHE[skeleton_name]
    return names.get(index)

BODY_KEYPOINT_NAMES = {
    17: ["nose","left_eye","right_eye","left_ear","right_ear",
         "left_shoulder","right_shoulder","left_elbow","right_elbow",
         "left_wrist","right_wrist","left_hip","right_hip",
         "left_knee","right_knee","left_ankle","right_ankle"],
    18: ["nose","neck","right_shoulder","right_elbow","right_wrist",
         "left_shoulder","left_elbow","left_wrist","right_hip",
         "right_knee","right_ankle","left_hip","left_knee","left_ankle",
         "right_eye","left_eye","right_ear","left_ear"],
}

# Plages d'id canoniques des points pieds par skeleton_name (cf. inference.py _OFFSETS).
# "human" (17pts YOLO) n'a jamais de pieds, donc absent volontairement.
_FEET_CANONICAL_IDS = {
    "openpose_fullbody": range(18, 24),
    "coco_fullbody":      range(17, 23),
}

# TODO(unknown):
# - [opt] Store paths instead of creating new ones at each paint.


class Shape:
    # Render handles as squares
    P_SQUARE = 0

    # Render handles as circles
    P_ROUND = 1

    # Flag for the handles we would move if dragging
    MOVE_VERTEX = 0

    # Flag for all other handles on the current shape
    NEAR_VERTEX = 1

    PEN_WIDTH = 2

    # The following class variables influence the drawing of all shape objects.
    line_color: QtGui.QColor = QtGui.QColor(0, 255, 0, 128)
    fill_color: QtGui.QColor = QtGui.QColor(0, 0, 0, 64)
    vertex_fill_color: QtGui.QColor = QtGui.QColor(0, 255, 0, 255)
    select_line_color: QtGui.QColor = QtGui.QColor(255, 255, 255, 255)
    select_fill_color: QtGui.QColor = QtGui.QColor(0, 255, 0, 64)
    hvertex_fill_color: QtGui.QColor = QtGui.QColor(255, 255, 255, 255)

    point_type = P_ROUND
    point_size = 8
    scale = 1.0

    _current_vertex_fill_color: QtGui.QColor

    def __init__(
        self,
        label=None,
        line_color=None,
        shape_type=None,
        flags=None,
        group_id=None,
        description=None,
        mask=None,
    ):
        self.label = label
        self.group_id = group_id
        self.points = []
        self.point_labels = []
        self.shape_type = shape_type
        self._shape_raw = None
        self._points_raw = []
        self._shape_type_raw = None
        self.fill = False
        self.selected = False
        self.flags = flags
        self.description = description
        self.other_data = {}
        self.mask = mask
        self.skeleton_name = "human"

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }

        self._closed = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color

    def _scale_point(self, point: QtCore.QPointF) -> QtCore.QPointF:
        return QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)

    def setShapeRefined(self, shape_type, points, point_labels, mask=None):
        self._shape_raw = (self.shape_type, self.points, self.point_labels)
        self.shape_type = shape_type
        self.points = points
        self.point_labels = point_labels
        self.mask = mask

    def restoreShapeRaw(self):
        if self._shape_raw is None:
            return
        self.shape_type, self.points, self.point_labels = self._shape_raw
        self._shape_raw = None

    @property
    def shape_type(self):
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value):
        if value is None:
            value = "polygon"
        if value not in [
            "polygon",
            "rectangle",
            "point",
            "line",
            "circle",
            "linestrip",
            "points",
            "mask",
            "skeleton",
        ]:
            raise ValueError(f"Unexpected shape_type: {value}")
        self._shape_type = value

    def close(self):
        self._closed = True

    def addPoint(self, point, label=1):
        if self.points and point == self.points[0]:
            self.close()
        else:
            self.points.append(point)
            self.point_labels.append(label)

    def canAddPoint(self):
        return self.shape_type in ["polygon", "linestrip"]

    def popPoint(self):
        if self.points:
            if self.point_labels:
                self.point_labels.pop()
            return self.points.pop()
        return None

    def insertPoint(self, i, point, label=1):
        self.points.insert(i, point)
        self.point_labels.insert(i, label)

    def canRemovePoint(self) -> bool:
        if not self.canAddPoint():
            return False

        if self.shape_type == "polygon" and len(self.points) <= 3:
            return False

        if self.shape_type == "linestrip" and len(self.points) <= 2:
            return False

        return True

    def removePoint(self, i: int):
        if not self.canRemovePoint():
            logger.warning(
                "Cannot remove point from: shape_type=%r, len(points)=%d",
                self.shape_type,
                len(self.points),
            )
            return

        self.points.pop(i)
        self.point_labels.pop(i)

    def isClosed(self):
        return self._closed

    def setOpen(self):
        self._closed = False

    def paint(self, painter):
        if self.mask is None and not self.points:
            return

        color = self.select_line_color if self.selected else self.line_color
        pen = QtGui.QPen(color)
        # Try using integer sizes for smoother drawing(?)
        pen.setWidth(self.PEN_WIDTH)
        painter.setPen(pen)

        if self.mask is not None:
            image_to_draw = np.zeros(self.mask.shape + (4,), dtype=np.uint8)
            fill_color = (
                self.select_fill_color.getRgb()
                if self.selected
                else self.fill_color.getRgb()
            )
            image_to_draw[self.mask] = fill_color
            qimage = QtGui.QImage.fromData(labelme.utils.img_arr_to_data(image_to_draw))
            qimage = qimage.scaled(
                qimage.size() * self.scale,
                QtCore.Qt.IgnoreAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )

            painter.drawImage(self._scale_point(point=self.points[0]), qimage)

            line_path = QtGui.QPainterPath()
            contours = skimage.measure.find_contours(np.pad(self.mask, pad_width=1))
            for contour in contours:
                contour += [self.points[0].y(), self.points[0].x()]
                line_path.moveTo(
                    self._scale_point(QtCore.QPointF(contour[0, 1], contour[0, 0]))
                )
                for point in contour[1:]:
                    line_path.lineTo(
                        self._scale_point(QtCore.QPointF(point[1], point[0]))
                    )
            painter.drawPath(line_path)

        if self.points:
            line_path = QtGui.QPainterPath()
            vrtx_path = QtGui.QPainterPath()
            negative_vrtx_path = QtGui.QPainterPath()

            if self.shape_type in ["rectangle", "mask"]:
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = QtCore.QRectF(
                        self._scale_point(self.points[0]),
                        self._scale_point(self.points[1]),
                    )
                    line_path.addRect(rectangle)
                if self.shape_type == "rectangle":
                    for i in range(len(self.points)):
                        self.drawVertex(vrtx_path, i)
            elif self.shape_type == "circle":
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    raidus = labelme.utils.distance(
                        self._scale_point(self.points[0] - self.points[1])
                    )
                    line_path.addEllipse(
                        self._scale_point(self.points[0]), raidus, raidus
                    )
                for i in range(len(self.points)):
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "linestrip":
                line_path.moveTo(self._scale_point(self.points[0]))
                for i, p in enumerate(self.points):
                    line_path.lineTo(self._scale_point(p))
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "points":
                assert len(self.points) == len(self.point_labels)
                for i, point_label in enumerate(self.point_labels):
                    if point_label == 1:
                        self.drawVertex(vrtx_path, i)
                    else:
                        self.drawVertex(negative_vrtx_path, i)
            elif self.shape_type == "skeleton":
                sk = _get_skeleton(getattr(self, "skeleton_name", "human"))

                # Construire id_canonique → position dans self.points
                meta = self.other_data.get("skeleton_meta", []) if self.other_data else []
                id_to_pos = {}
                if meta:
                    for pos, pt in enumerate(meta):
                        id_to_pos[int(pt[0])] = pos
                else:
                    # fallback : position == id (cas human 17pts YOLO)
                    for pos in range(len(self.points)):
                        id_to_pos[pos] = pos

                def is_zero(idx):
                    p = self.points[idx]
                    return p.x() == 0 and p.y() == 0

                for i in range(len(self.points)):
                    if is_zero(i):
                        continue
                    self.drawVertex(vrtx_path, i)
                for a, b in sk:
                    pa = id_to_pos.get(a)
                    pb = id_to_pos.get(b)
                    if pa is None or pb is None:
                        continue
                    if pa < len(self.points) and pb < len(self.points):
                        if is_zero(pa) or is_zero(pb):
                            continue
                        line_path.moveTo(self._scale_point(self.points[pa]))
                        line_path.lineTo(self._scale_point(self.points[pb]))
            else:
                line_path.moveTo(self._scale_point(self.points[0]))
                # Uncommenting the following line will draw 2 paths
                # for the 1st vertex, and make it non-filled, which
                # may be desirable.
                # self.drawVertex(vrtx_path, 0)

                for i, p in enumerate(self.points):
                    line_path.lineTo(self._scale_point(p))
                    self.drawVertex(vrtx_path, i)
                if self.isClosed():
                    line_path.lineTo(self._scale_point(self.points[0]))

            painter.drawPath(line_path)
            if vrtx_path.length() > 0:
                painter.drawPath(vrtx_path)
                painter.fillPath(vrtx_path, self._current_vertex_fill_color)
            if self.fill and self.shape_type not in [
                "line",
                "linestrip",
                "points",
                "mask",
            ]:
                color = self.select_fill_color if self.selected else self.fill_color
                painter.fillPath(line_path, color)

            pen.setColor(QtGui.QColor(255, 0, 0, 255))
            painter.setPen(pen)
            painter.drawPath(negative_vrtx_path)
            painter.fillPath(negative_vrtx_path, QtGui.QColor(255, 0, 0, 255))

    def drawVertex(self, path, i):
        d = self.point_size
        shape = self.point_type
        point = self._scale_point(self.points[i])
        if i == self._highlightIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size  # type: ignore[assignment]
        if self._highlightIndex is not None:
            self._current_vertex_fill_color = self.hvertex_fill_color
        else:
            self._current_vertex_fill_color = self.vertex_fill_color
        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    def missing_body_keypoints(self):
        """Retourne [(index, name)] des keypoints (0,0) ajoutables via clic droit.

        Body : comportement existant, via BODY_KEYPOINT_NAMES (indices fixes
        0..16 ou 0..17, toujours en tête du tableau).

        Feet : ajoutés en plus, identifiés via l'id canonique stocké dans
        other_data["skeleton_meta"] (les 4-tuples [id, x, y, conf] d'origine,
        conservés tels quels au chargement). On ne les propose QUE s'ils sont
        réellement présents dans ce skeleton — c'est-à-dire seulement si le
        groupe "feet" était coché à l'inférence, sinon ces indices n'existent
        pas du tout dans le tableau (compacté, pas de zéro-padding)."""
        if self.shape_type != "skeleton":
            return []
        n = len(self.points)
        names = BODY_KEYPOINT_NAMES.get(17 if n == 17 else 18)
        result = []
        if names:
            for i, name in enumerate(names):
                if i < len(self.points):
                    p = self.points[i]
                    if p.x() == 0 and p.y() == 0:
                        result.append((i, name))

        feet_ids = _FEET_CANONICAL_IDS.get(self.skeleton_name)
        meta = self.other_data.get("skeleton_meta") if self.other_data else None
        if feet_ids and meta:
            for i, entry in enumerate(meta):
                if i >= len(self.points) or not entry:
                    continue
                try:
                    canonical_id = int(entry[0])
                except (TypeError, ValueError, IndexError):
                    continue
                if canonical_id not in feet_ids:
                    continue
                p = self.points[i]
                if p.x() == 0 and p.y() == 0:
                    kp_name = get_keypoint_name(self.skeleton_name, canonical_id) \
                        or f"foot_point_{canonical_id}"
                    result.append((i, kp_name))

        return result

    def body_bbox_center(self):
        """Centre de la bbox des points valides (non (0,0))."""
        valid = [p for p in self.points if not (p.x() == 0 and p.y() == 0)]
        if not valid:
            return QtCore.QPointF(0, 0)
        xs = [p.x() for p in valid]
        ys = [p.y() for p in valid]
        return QtCore.QPointF((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)

    def nearestVertex(self, point, epsilon):
        min_distance = float("inf")
        min_i = None
        point = QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)
        for i, p in enumerate(self.points):
            p = QtCore.QPointF(p.x() * self.scale, p.y() * self.scale)
            dist = labelme.utils.distance(p - point)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                min_i = i
        return min_i

    def nearestEdge(self, point, epsilon):
        min_distance = float("inf")
        post_i = None
        point = QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)
        for i in range(len(self.points)):
            start = self.points[i - 1]
            end = self.points[i]
            start = QtCore.QPointF(start.x() * self.scale, start.y() * self.scale)
            end = QtCore.QPointF(end.x() * self.scale, end.y() * self.scale)
            line = [start, end]
            dist = labelme.utils.distancetoline(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                post_i = i
        return post_i

    def containsPoint(self, point) -> bool:
        if self.shape_type in ["line", "linestrip", "points"]:
            return False
        if self.mask is not None:
            y = np.clip(
                int(round(point.y() - self.points[0].y())),
                0,
                self.mask.shape[0] - 1,
            )
            x = np.clip(
                int(round(point.x() - self.points[0].x())),
                0,
                self.mask.shape[1] - 1,
            )
            return self.mask[y, x]
        return self.makePath().contains(point)

    def makePath(self):
        if self.shape_type in ["rectangle", "mask"]:
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                path.addRect(QtCore.QRectF(self.points[0], self.points[1]))
        elif self.shape_type == "circle":
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                raidus = labelme.utils.distance(self.points[0] - self.points[1])
                path.addEllipse(self.points[0], raidus, raidus)
        else:
            path = QtGui.QPainterPath(self.points[0])
            for p in self.points[1:]:
                path.lineTo(p)
        return path

    def boundingRect(self):
        return self.makePath().boundingRect()

    def moveBy(self, offset):
        self.points = [p + offset for p in self.points]

    def moveVertexBy(self, i, offset):
        self.points[i] = self.points[i] + offset

    def highlightVertex(self, i, action):
        """Highlight a vertex appropriately based on the current action

        Args:
            i (int): The vertex index
            action (int): The action
            (see Shape.NEAR_VERTEX and Shape.MOVE_VERTEX)
        """
        self._highlightIndex = i
        self._highlightMode = action

    def highlightClear(self):
        """Clear the highlighted point"""
        self._highlightIndex = None

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value