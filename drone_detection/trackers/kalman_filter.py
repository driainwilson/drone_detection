import math
import numpy as np

from ..utils import cxcywh_to_xyxy


class KalmanFilter:
    def __init__(self,
                 dt: float = 1 / 30.0,
                 Q: float = 10,
                 R: float = 10):
        """
        Initializes the Kalman Filter with a new 8D state vector.
        dt: time step (1/FPS)
        Q: process noise covariance (uncertainty in the process model)
            Low: The filter will be very smooth but slow to react to real changes in direction or speed (high lag)
            High: The filter will react quickly to changes but will be less smooth and more influenced by measurement noise.
        R: measurement noise covariance (uncertainty in the measurement)
            LOW: The filter will follow the measurements very closely, making it less smooth.
            HIGH: The filter will ignore much of the measurement noise and rely more on its own predictions, making it smoother.
        """
        self.dt = dt

        # State vector [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.zeros((8, 1))  # MODIFIED: State vector is now 8D

        # State Transition Matrix F (8x8)
        # Defines the constant velocity model for the new state
        self.F = np.array([
            [1, 0, 0, 0, self.dt, 0, 0, 0],  # cx_new = cx + vx*dt
            [0, 1, 0, 0, 0, self.dt, 0, 0],  # cy_new = cy + vy*dt
            [0, 0, 1, 0, 0, 0, self.dt, 0],  # w_new  = w  + vw*dt
            [0, 0, 0, 1, 0, 0, 0, self.dt],  # h_new  = h  + vh*dt
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx_new = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy_new = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw_new = vw
            [0, 0, 0, 0, 0, 0, 0, 1]  # vh_new = vh
        ])

        # Measurement Matrix H (4x8)
        # Maps the state vector to the measurement vector [cx, cy, w, h]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])

        # State Covariance Matrix P (8x8)
        # Our uncertainty in the initial state estimate.
        self.P = np.eye(8) * 1000

        # Process Noise Covariance Q (8x8)
        # Uncertainty in the process model (e.g., drone accelerating)
        self.Q = np.eye(8) * Q

        # Measurement Noise Covariance R (4x4)
        # Uncertainty in our measurement (noise from the object detector)
        self.R = np.eye(4) * R

        self._is_initialized = False

    def predict(self):
        """Predicts the next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, bbox_cxcywh: tuple[float | int, ...] | None):
        """
        Updates the state estimate with a new measurement (bounding box).
        bbox: tuple (x, y, w, h)
        """
        if bbox_cxcywh is None:
            return self.get_state()

        cx, cy, w, h = bbox_cxcywh
        z = np.array([[cx], [cy], [w], [h]])
        if not self._is_initialized:
            self.x[:4] = z
            self._is_initialized = True
            return self.get_state()

        # Kalman Filter update steps
        y_residual = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_residual
        I = np.eye(8)
        self.P = (I - K @ self.H) @ self.P

        return self.get_state()

    def get_state(self):
        """Returns the current smoothed state as a dictionary."""
        cx, cy, w, h, vx, vy, vw, vh = self.x.flatten()

        speed_xy = math.sqrt(vx ** 2 + vy ** 2)
        speed_3d = math.sqrt(vx ** 2 + vy ** 2 + vw ** 2)

        direction = math.atan2(vy, vx)
        return {
            "cx": cx, "cy": cy, "w": w, "h": h,
            "vx": vx, "vy": vy, "vw": vw, "vh": vh,
            "speed_xy": speed_xy,
            "speed_3d": speed_3d,
            "area": w*h,
            "va":(w * vh) + (h * vw),
            "direction_xy_radians":direction,
            "is_initialized": self._is_initialized
        }


    def get_bbox_xyxy(self) -> tuple[float | int, ...]:
        cx, cy, w, h, _, _, _, _ = self.x.flatten()
        return cxcywh_to_xyxy((cx, cy, w, h))
