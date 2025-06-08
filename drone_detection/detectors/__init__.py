import abc
import enum
from abc import ABC
from typing import Any
import dataclasses
import numpy as np
from loguru import logger
from numpy import typing as npt
from omegaconf import DictConfig

from ..utils import xyxy_to_cxcywh, xyxy_to_xywh
__all__ = ["create"]


class DetectorType(enum.Enum):
    # MOG2 = "MOG2"
    YOLO = "YOLO"


@dataclasses.dataclass()
class Detection:
    bbox_xyxy: tuple[float | int, ...]
    data: npt.NDArray[np.uint8]
    confidence: float | None = None

    @property
    def bbox_xywh(self) -> tuple[float | int, ...]:
        return xyxy_to_xywh(self.bbox_xyxy)

    @property
    def bbox_cxcywh(self) -> tuple[float | int, ...]:
        return xyxy_to_cxcywh(self.bbox_xyxy)


class BaseDetector(ABC):

    @abc.abstractmethod
    def __init__(self, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def run(self, frame: npt.NDArray[np.int8]) -> list[Detection]:
        ...


from .dl_object_detector import *

DETECTOR_FACTORY: dict[DetectorType, Any] = {
    DetectorType.YOLO: DetectorYOLO
}


def create(cfg: DictConfig) -> BaseDetector:
    if "type" not in cfg:
        raise ValueError("detector config must have a type")

    if cfg.type not in DetectorType._value2member_map_:
        raise ValueError("Detector unknown type %s" % cfg.type)

    detector_type = DetectorType(cfg.type)
    if detector_type not in DETECTOR_FACTORY:
        raise ValueError("Detector unknown type %s" % cfg.type)

    logger.debug("Creating detector %s" % cfg.type)

    return DETECTOR_FACTORY[DetectorType(cfg.type)](**cfg.parameters)
