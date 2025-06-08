from __future__ import annotations

import pathlib

import cv2
import numpy as np
from loguru import logger
from numpy import typing as npt
from omegaconf import ListConfig

from . import BaseGrabber

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
# DEFAULT_PATH = package_root / "data/videos"


class VideoGrabber(BaseGrabber):
    """
    A grabber that reads frames from a video file or a list of video files.
    """

    def __init__(self, video_path: str | list[str], video_root_dir: str, **kwargs) -> None:
        """
           Initializes the VideoGrabber with the given video path(s).

           Args:
               video_path: A string or a list of strings representing the path(s) to the video file(s).
                           If a directory is provided, all .mp4 files in that directory will be used.
               **kwargs: Additional keyword arguments.
           """

        if not pathlib.Path(video_root_dir).is_absolute():
            video_root_dir = PACKAGE_ROOT / video_root_dir

        paths = video_path

        if isinstance(paths, ListConfig):
            paths = list(paths)
        elif pathlib.Path(paths).is_dir():
            if not pathlib.Path(paths).is_absolute():
                paths = video_root_dir / paths
            paths = list(pathlib.Path(paths).glob("*.mp4"))
        else:
            paths = [paths]

        self.paths = []
        for path in paths:
            if not pathlib.Path(path).is_absolute():
                # if that path is relative and does not exist in the video_root_dir - the user may have it relative to the running script, so ignore.
                if (video_root_dir / path).exists():
                    path = video_root_dir / path
            self.paths.append(path)

        self.index = 0
        self.cap = None
        logger.debug(f"Loaded: {self.paths}")

    def __iter__(self) -> VideoGrabber:
        return self

    def grab(self) -> tuple[bool, npt.NDArray[np.uint8]]:
        """
       Grabs a frame from the current video.

       Returns:
           A tuple containing a boolean indicating success and the frame as a numpy array.
           Returns None if there are no more frames or if the video file cannot be opened.

       Raises:
           StopIteration: If all video files have been processed.
           FileNotFoundError: If a video file is not found.
       """
        if self.cap is None:
            if self.index >= len(self.paths):
                raise StopIteration
            if not pathlib.Path(self.paths[self.index]).exists():
                raise FileNotFoundError(f"Video file not found: {self.paths[self.index]}")
            self.cap = cv2.VideoCapture(str(self.paths[self.index]))
            logger.info(f"Loaded: {self.paths[self.index]}")
        return self.cap.read()

    def __next__(self) -> npt.NDArray[np.uint8]:
        """
        Returns the next frame from the video.

        Returns:
            The next frame as a numpy array.

        Raises:
            StopIteration: If there are no more frames in the video(s).
        """
        ret, frame = self.grab()
        if not ret:
            self.index += 1
            self.cap.release()
            self.cap = None
            ret, frame = self.grab()
        return frame

    def __del__(self) -> None:
        """
        Releases the video capture object when the VideoGrabber is deleted.
        """
        if self.cap is not None:
            self.cap.release()
