# yolo11/tracking/kalman_tracker.py
import numpy as np
from typing import List, Dict
from scipy.optimize import linear_sum_assignment

from .base import BaseTracker
from .utils import iou


class KalmanTrack:
    def __init__(self, box, track_id):
        cx, cy, w, h = self._box_to_state(box)

        # state: [cx, cy, vx, vy, w, h]
        self.x = np.array([cx, cy, 0, 0, w, h], dtype=float)

        self.P = np.eye(6) * 10.0
        self.id = track_id
        self.missed = 0

        self.H = np.zeros((4, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 4] = 1
        self.H[3, 5] = 1

        self.Q = np.eye(6) * 0.01
        self.R = np.eye(4) * 1.0

    def predict(self, dt: int = 1):
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q * dt

    def update(self, box):
        z = np.array(self._box_to_meas(box))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.missed = 0

    def get_box(self):
        cx, cy, _, _, w, h = self.x
        return [
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
        ]

    @staticmethod
    def _box_to_state(box):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return cx, cy, w, h

    @staticmethod
    def _box_to_meas(box):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return cx, cy, w, h


class KalmanTracker(BaseTracker):
    def __init__(self, iou_thresh=0.3, max_missed_seconds=2.0, source_fps=25.0, sample_interval=1):
        self.iou_thresh = iou_thresh
        self.max_missed = max(1, int(max_missed_seconds * source_fps / sample_interval))
        self.tracks: Dict[int, KalmanTrack] = {}
        self.next_id = 1

    def reset(self):
        self.tracks.clear()
        self.next_id = 1

    def update(self, detections: List[List[float]], dt: int = 1) -> List[int]:
        # 1) Predict
        for trk in self.tracks.values():
            trk.predict(dt=dt)

        if not detections:
            for trk in self.tracks.values():
                trk.missed += 1
            self._cleanup()
            return []

        dets = np.array(detections)
        track_ids = list(self.tracks.keys())
        track_boxes = np.array([self.tracks[t].get_box() for t in track_ids])

        # 2) IoU cost
        cost = np.zeros((len(track_boxes), len(dets)))
        for i, tb in enumerate(track_boxes):
            for j, db in enumerate(dets):
                cost[i, j] = 1 - iou(tb.tolist(), db.tolist())

        row, col = linear_sum_assignment(cost)

        assigned_dets = set()
        result_ids = [-1] * len(dets)

        # 3) Update matched
        for r, c in zip(row, col):
            if cost[r, c] < 1 - self.iou_thresh:
                tid = track_ids[r]
                self.tracks[tid].update(dets[c].tolist())
                result_ids[c] = tid
                assigned_dets.add(c)

        # 4) New tracks
        for i, box in enumerate(dets):
            if i not in assigned_dets:
                tid = self.next_id
                self.tracks[tid] = KalmanTrack(box.tolist(), tid)
                result_ids[i] = tid
                self.next_id += 1

        # 5) Missed tracks
        for tid, trk in self.tracks.items():
            if tid not in result_ids:
                trk.missed += 1

        self._cleanup()
        return result_ids

    def _cleanup(self):
        self.tracks = {
            tid: trk for tid, trk in self.tracks.items()
            if trk.missed <= self.max_missed
        }
