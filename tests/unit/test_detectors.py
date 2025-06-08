import pathlib

import cv2
import pytest
from omegaconf import DictConfig

from drone_detection.detectors import create, DetectorType
from drone_detection.detectors.dl_object_detector import DetectorYOLO


def test_detector_factory():
    package_root = pathlib.Path(__file__).resolve().parents[2]
    model_path = package_root / "weights" / "yolov11s_640_best.pt"
    create(cfg=DictConfig({"type": DetectorType.YOLO.value,
                           "parameters": {"model_path": model_path}}))


def test_detector_yolo_run():
    package_root = pathlib.Path(__file__).resolve().parents[2]
    model_path = package_root / "weights" / "yolov11s_640_best.pt"
    detector = DetectorYOLO(model_path=str(model_path))

    frame = cv2.imread("../files/VS_P1200.jpg")

    # Run the detector
    detections = detector.run(frame)

    assert isinstance(detections, list)
    assert len(detections) == 1
    assert len(detections[0].bbox_xyxy) == 4
