import cv2
import hydra
from omegaconf import DictConfig

from drone_detection import grabbers, detectors, trackers, classifiers
from drone_detection.utils import draw_track, draw_classification, draw_threat_scores



@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # create components from the config file
    detector = detectors.create(cfg.detector)
    grabber = grabbers.create(cfg.grabber)
    tracker = trackers.create(cfg.tracker)
    behaviour_classifier, threat_score_calculator = classifiers.create(cfg.classifier)

    min_track_length = cfg.classifier.min_track_length

    frame_delay = 1  # if set to 1, display will wait for a user input
    write_video = cfg.writer.enabled
    writer = None

    # loop through frames
    for image in grabber:
        detections = detector.run(image)

        tracks = tracker.update(detections=detections, frame=image)
        if tracks is None:
            continue

        threat_scores: dict[int, float] = {}
        for track in tracks:

            # is track too old
            if track.time_since_last_seen > cfg.tracker.age_threshold:
                continue

            # classify behaviour
            if len(track) > min_track_length:
                classifications = behaviour_classifier(state_history=list(track.state_history))
                threat_score = threat_score_calculator(state=track.state,
                                                       behavior_probs=classifications)
                threat_scores[track.track_id] = threat_score
                image = draw_classification(image,
                                            classifications=classifications,
                                            bbox_xyxy=track.bbox_xyxy)

            image = draw_track(image,
                               bbox_xyxy=track.bbox_xyxy,
                               track_id=track.track_id,
                               )

        image = draw_threat_scores(image, scores=threat_scores)

        if write_video:
            if writer is None:
                height, width = image.shape[:2]
                writer = grabbers.VideoWriter(filename=cfg.writer.filename,frame_size=(width, height))
            writer.add_frame(image)

        cv2.imshow("image", image)
        key = cv2.waitKey(frame_delay)
        if key == ord("q"):
            break
        if key == ord("p"):
            frame_delay = 0 if frame_delay == 1 else 1
    if write_video:
        writer.save()

if __name__ == "__main__":
    main()
