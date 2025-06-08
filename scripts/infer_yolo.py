# Test inference on a video
import pathlib

import cv2
from drone_detection.multiobject_tracker import MultiObjectTracker
from omegaconf import DictConfig

from drone_detection import grabbers, detectors
from drone_detection.utils import draw_bbox

package_root = pathlib.Path(__file__).resolve().parents[1]

video = package_root / "data/videos/V_DRONE_052.mp4"
model_path = package_root / "data/weights/yolov11s_640_best.pt"
grabber = grabbers.create(DictConfig({"type": "VIDEO",
                                      "parameters": {
                                          "video_path": video}}))

detector = detectors.create(DictConfig({"type": "YOLO",
                                        "parameters": {
                                            "model_path": model_path}}))

tracker = MultiObjectTracker()

for image in grabber:
    detections = detector.model.track(image, persist=True, tracker=package_root/"config/botsort.yml")
    annotated_frame = detections[0].plot()
    print("poo")
    # for detection in detections:
    #     image = draw_bbox(image,
    #                       bbox=detection.bbox_xyxy,
    #                       confidence=detection.confidence,
    #                       thickness=1)

    # tracks = tracker.update(detections=detections)
    # for track_id, bbox in tracks:
    #     print(tracks)
    #     # image = draw_bbox(image, bbox=bbox, colour=(255,255,0), thickness=1)
    #     track = tracker.tracks[track_id]
    #
    #     state = track.current_state()
    #     image = draw_track(image, bbox=state["bbox"],
    #                        velocity=state["velocity_xy"],
    #                        direction_radians=state["direction_xy_radians"])
    cv2.imshow("image", annotated_frame)
    key = cv2.waitKey(0)
    if key == ord("q"):
        break
