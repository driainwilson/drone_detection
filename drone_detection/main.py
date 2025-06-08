"""

Entry point

Load Vid
Detect
Track & ID

"""

import cv2
import hydra
from omegaconf import DictConfig

from drone_detection import grabbers, detectors, trackers, classifiers
from drone_detection.utils import draw_track, draw_classification, draw_threat_score


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    detector = detectors.create(cfg.detector)
    grabber = grabbers.create(cfg.grabber)
    tracker = trackers.create(cfg.tracker)
    behaviour_classifier, threat_score_calculator = classifiers.create(cfg.classifier)

    min_track_length = cfg.classifier.min_track_length

    for image in grabber:
        detections = detector.run(image)
        tracks = tracker.update(detections=detections, frame=image)
        if tracks is None:
            continue

        for track in tracks:

            # is track too old
            if track.time_since_last_seen > cfg.tracker.age_threshold:
                continue

            # classify behaviour
            if len(track) > min_track_length:
                classifications = behaviour_classifier(state_history=list(track.state_history))
                threat_score = threat_score_calculator(state=track.state,
                                                       behavior_probs=classifications)
                image = draw_classification(image, classifications=classifications)
                image = draw_threat_score(image, score=threat_score)

            image = draw_track(image,
                               bbox_xyxy=track.bbox_xyxy,
                               track_id=track.track_id,
                               )
            # image = draw_state(image, state=track.state)

        cv2.imshow("image", image)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
