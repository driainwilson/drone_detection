import abc
import enum
from typing import Any

import numpy as np
from loguru import logger
from numpy import typing as npt
from omegaconf import DictConfig

from .track import Track

from ..detectors import Detection


class TrackerType(enum.Enum):
    MultiObject = "MultiObject"
    DeepSort = "DeepSort"
    YOLO = "YOLO"


class BaseTracker(abc.ABC):

    @abc.abstractmethod
    def update(self, detections: list[Detection], frame: npt.NDArray[np.uint8]) -> list[Track]:
        ...


from .yolo_tracker import TrackerYOLO
from .deep_sort import DeepSortTracker

TRACKER_FACTORY: dict[TrackerType, Any] = {
    TrackerType.DeepSort: DeepSortTracker,
    TrackerType.YOLO: TrackerYOLO
}


def create(cfg: DictConfig) -> DeepSortTracker:
    if "type" not in cfg:
        raise ValueError("Tracker config must have a type")

    if cfg.type not in TrackerType._value2member_map_:
        raise ValueError("Tracker unknown type %s" % cfg.type)

    tracker_type = TrackerType(cfg.type)
    if tracker_type not in TRACKER_FACTORY:
        raise ValueError("Tracker unknown type %s" % cfg.type)

    logger.debug("Creating Tracker %s" % cfg.type)
    return TRACKER_FACTORY[TrackerType(cfg.type)](**cfg.parameters)
