from typing import Any

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from numpy import typing as npt
from loguru import logger

from ..detectors import Detection
from . import BaseTracker, Track

class DeepSortTracker(BaseTracker):

    def __init__(self, max_age: int = 5,
                 track_kwargs: dict[str, Any] | None = None,
                 **kwargs):

        self.tracks = {}
        self.max_age = max_age
        self.track_kwargs = track_kwargs
        if self.track_kwargs is None:
            self.track_kwargs = {}

        self.tracker = DeepSort(max_age=max_age,
                                n_init=2,
                                max_cosine_distance=0.9,
                                nms_max_overlap=1.0)

    def update(self, detections: list[Detection], frame: npt.NDArray[np.uint8]) -> list[Track]:
        detections_xywh = []

        for det in detections:
            detections_xywh.append((det.bbox_xywh, det.confidence, "drone"))  # hard coded as we only have one class

        ds_tracks = self.tracker.update_tracks(detections_xywh, frame=frame)


        tracked_ids = []
        for ds_track in ds_tracks:
            if not ds_track.is_confirmed():
                continue

            track_id = ds_track.track_id
            track = self.tracks.get(track_id)
            box_xyxy = ds_track.to_ltrb()
            xmin, ymin, xmax, ymax = np.rint(box_xyxy).astype(int)
            detection = Detection(bbox_xyxy=box_xyxy,
                                  data=frame[ymin:ymax, xmin:xmax, :])

            if track is None:
                logger.info(f"New Track: {track_id}")
                track = Track(track_id=track_id, detection=detection, **self.track_kwargs)
                self.tracks[track_id] = track
            tracked_ids.append(track_id)

            track.update(detection=detection)

        # update all other tracks
        for track_id, track in self.tracks.items():
            if track_id not in tracked_ids:
                track.update()

        # remove all tracks that are too old
        for track_id, track in self.tracks.items():
            if track.time_since_last_seen > self.max_age:
                logger.info(f"Removing Track: {track_id}")
                del self.tracks[track_id]

        return list(self.tracks.values())