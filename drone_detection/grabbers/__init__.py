import abc
import enum
from abc import ABC
from typing import Any
import numpy as np
from numpy import typing as npt
from omegaconf import DictConfig
from loguru import logger

class GrabberType(enum.Enum):
    VIDEO = "VIDEO"
    CAMERA = "CAMERA"


class BaseGrabber(ABC):

    @abc.abstractmethod
    def __init__(self, parameters: dict[str, Any] | None = None):
        ...

    @abc.abstractmethod
    def grab(self) -> npt.NDArray[np.uint8]:
        ...


from .file_grabber import VideoGrabber

GRABBER_FACTORY: dict[GrabberType, Any] = {
    GrabberType.VIDEO: VideoGrabber
}

def create(cfg: DictConfig) -> BaseGrabber:
    if "type" not in cfg:
        raise ValueError("Grabber config must have a type")

    if cfg.type not in GrabberType._value2member_map_:
        raise ValueError("Grabber unknown type %s" % cfg.type)

    grabber_type = GrabberType(cfg.type)
    if grabber_type not in GRABBER_FACTORY:
        raise ValueError("Grabber unknown type %s" % cfg.type)

    logger.debug("Creating Grabber %s" % cfg.type)
    return GRABBER_FACTORY[GrabberType(cfg.type)](**cfg.parameters)


