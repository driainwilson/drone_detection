import abc
import enum
from abc import ABC
from typing import Any, Iterable
import numpy as np
from numpy import typing as npt
from omegaconf import DictConfig
from loguru import logger
from .video_writer import *

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

def create(cfg: DictConfig) -> Iterable[npt.NDArray[np.uint8]]:
    """
     Creates a grabber based on the provided configuration.

     Args:
         cfg: A DictConfig object containing the grabber configuration.  Must have a 'type' field.
              The parameters for the grabber are passed in the parameters field.

     Returns:
         An iterable of numpy arrays, each representing a frame from the grabber.

     Raises:
         ValueError: If the configuration does not contain a 'type' field,
                     if the specified grabber type is unknown, or if the grabber
                     type is not found in the grabber factory.
     """
    if "type" not in cfg:
        raise ValueError("Grabber config must have a type")

    if cfg.type not in GrabberType._value2member_map_:
        raise ValueError("Grabber unknown type %s" % cfg.type)

    grabber_type = GrabberType(cfg.type)
    if grabber_type not in GRABBER_FACTORY:
        raise ValueError("Grabber unknown type %s" % cfg.type)

    logger.debug("Creating Grabber %s" % cfg.type)
    return GRABBER_FACTORY[GrabberType(cfg.type)](**cfg.parameters)


