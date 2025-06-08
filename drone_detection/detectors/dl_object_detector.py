import pathlib
from collections import defaultdict
from typing import Any

import numpy as np
from numpy import typing as npt
from ultralytics import YOLO

from . import BaseDetector, Detection

__all__ = ["DetectorYOLO"]

package_root = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PATH = package_root / "data/weights"

class DetectorYOLO(BaseDetector):

    def __init__(self, model_path: str, min_confidence:float=0.5, **kwargs) -> None:

        self.min_confidence = min_confidence
        model_path = DEFAULT_PATH / model_path
        if not pathlib.Path(model_path).exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")

        self.model = YOLO(model_path)

    def run(self, frame: npt.NDArray[np.uint8]) -> list[Detection]:

        results = self.model(frame, verbose=False)
        detections: list[Detection] = []
        for r in results:
            bboxes = r.boxes.cpu()

            for box, conf in zip(bboxes.xyxy.numpy(), bboxes.conf.numpy()):
                if conf < self.min_confidence:
                    continue
                xmin, ymin, xmax, ymax = np.rint(box).astype(int)

                detections.append(Detection(bbox_xyxy=box,
                                            data=frame[ymin:ymax, xmin:xmax, :],
                                            confidence=conf))
        return detections