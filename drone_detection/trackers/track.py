import dataclasses
from collections import deque
from typing import Any

from ..detectors import Detection
from .kalman_filter import KalmanFilter


@dataclasses.dataclass()
class Track:
    track_id: int
    detection: Detection
    time_since_last_seen: int = 0
    history: list[Detection | None] = dataclasses.field(default_factory=list, repr=False)
    estimator: KalmanFilter | None = None
    estimator_Q: float = 10
    estimator_R: float = 10
    estimator_dt: float = 1 / 30.0
    state: dict[str, Any] = None
    state_history_max_length: int = 15
    state_history: deque[dict[str, Any]] | None = None

    def __post_init__(self):
        self.state_history = deque(maxlen=self.state_history_max_length)
        self.estimator = KalmanFilter(dt=self.estimator_dt, Q=self.estimator_Q, R=self.estimator_R)

    def update(self, detection: Detection | None = None):

        self.estimator.predict()
        if detection is None:
            self.time_since_last_seen += 1
            self.history.append(None)
        else:
            self.detection = detection
            self.time_since_last_seen = 0
            self.estimator.update(bbox_cxcywh=detection.bbox_cxcywh)

        self.state = self.estimator.get_state()
        self.state_history.append(self.state)

    @property
    def bbox_xyxy(self) -> tuple[float | int, ...]:
        return self.estimator.get_bbox_xyxy()

    @property
    def velocity_xy(self) -> tuple[float, float]:
        return self.state["vx"], self.state["vy"]

    @property
    def velocity_z(self) -> float:
        return self.state["vw"]

    @property
    def direction_xy_radians(self) -> float:
        return self.state["direction_xy_radians"]

    def __len__(self) -> int:
        if self.state_history is None:
            return 0
        return len([h for h in self.state_history if h is not None])
