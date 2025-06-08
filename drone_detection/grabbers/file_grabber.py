from __future__ import annotations

import pathlib

import cv2
import numpy as np
from loguru import logger
from numpy import typing as npt
from omegaconf import ListConfig

from . import BaseGrabber

package_root = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PATH = package_root / "data/videos"


class VideoGrabber(BaseGrabber):

    def __init__(self, video_path: str | list[str], **kwargs) -> None:

        self.path = video_path

        if isinstance(self.path, ListConfig):
            self.path = list(self.path)
        elif pathlib.Path(self.path).is_dir():
            self.path = list(pathlib.Path(self.path).glob("*.mp4"))
        else:
            self.path = [self.path]

        self.path = [DEFAULT_PATH / path for path in self.path]

        self.index = 0
        self.cap = None
        logger.debug(f"Loaded: {self.path}")

    def __iter__(self) -> VideoGrabber:
        return self

    def grab(self):
        if self.cap is None:
            if self.index >= len(self.path):
                raise StopIteration
            if not pathlib.Path(self.path[self.index]).exists():
                raise FileNotFoundError(f"Video file not found: {self.path[self.index]}")
            self.cap = cv2.VideoCapture(self.path[self.index])
            logger.info(f"Loaded: {self.path[self.index]}")
        return self.cap.read()

    def __next__(self) -> npt.NDArray[np.uint8]:
        ret, frame = self.grab()
        if not ret:
            self.index += 1
            self.cap.release()
            self.cap = None
            ret, frame = self.grab()
        return frame

    def __del__(self) -> None:
        if self.cap is not None:
            self.cap.release()
