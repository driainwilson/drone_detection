import pathlib
from typing import Any

import numpy as np
from loguru import logger
from numpy import typing as npt
from ultralytics import YOLO

from ..detectors import Detection

from . import BaseTracker, Track

__all__ = ["TrackerYOLO"]

package_root = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PATH = package_root / "data/weights"


class TrackerYOLO(BaseTracker):

    def __init__(self, model_path: str,
                 config_file: str | None = None,
                 track_kwargs: dict[str, Any] | None = None,
                 max_age: int = 5,
                 **kwargs) -> None:

        if config_file is None:
            config_file = "botsort.yml"

        self.config_file = config_file
        self.max_age = max_age
        model_path = DEFAULT_PATH / model_path
        if not pathlib.Path(model_path).exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")

        self.tracks = {}
        self.track_kwargs = track_kwargs
        if self.track_kwargs is None:
            self.track_kwargs = {}
        self.model = YOLO(model_path)

    def update(self, detections: list[Detection] | None, frame: npt.NDArray[np.uint8]) -> list[Track]:
        result = self.model.track(frame,
                                  verbose=False,
                                  persist=True,
                                  tracker=package_root / f"config/{self.config_file}")[0]

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        if not result.boxes.is_track:
            # no detections - just update tracks
            for track in self.tracks.values():
                track.update(None)

            return list(self.tracks.values())

        track_ids = result.boxes.id.int().cpu().tolist()
        for box, conf, track_id in zip(boxes, confs, track_ids):
            track = self.tracks.get(track_id)

            xmin, ymin, xmax, ymax = np.rint(box).astype(int)
            detection = Detection(bbox_xyxy=box,
                                  data=frame[ymin:ymax, xmin:xmax, :],
                                  confidence=conf)
            if track is None:
                logger.info(f"New Track: {track_id}")
                track = Track(track_id=track_id, detection=detection, **self.track_kwargs)
                self.tracks[track_id] = track

            track.update(detection=detection)

        # update all other tracks
        for track_id, track in self.tracks.items():
            if track_id not in track_ids:
                track.update()

        # remove all tracks that are too old
        for track_id in list(self.tracks.keys()):

            if self.tracks[track_id].time_since_last_seen > self.max_age:
                logger.info(f"Removing Track: {track_id}")
                del self.tracks[track_id]

        return list(self.tracks.values())
