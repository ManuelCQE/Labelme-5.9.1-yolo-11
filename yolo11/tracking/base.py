# yolo11/tracking/base.py
from abc import ABC, abstractmethod
from typing import List


class BaseTracker(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, detections: List[list]) -> List[int]:
        """
        detections: list of [x1,y1,x2,y2]
        returns: list of track_ids aligned with detections
        """
        pass
