from typing import Any

import math
import numpy as np


def _gaussian_score(x: float, mu: float, sigma: float) -> float:
    """Returns a score from 0 to 1 based on a Gaussian distribution."""
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _sigmoid_score(x: float, threshold: float, steepness: float = 1.0) -> float:
    """Returns a score from 0 to 1 based on a logistic function."""
    return 1 / (1 + math.exp(-(x - threshold) * steepness))


def _circular_std_dev(rads: list[float]) -> float:
    """Calculates the standard deviation of angles, handling wraparound (e.g., 359deg and 1deg)."""
    # Convert angles to vectors on the unit circle
    x = np.cos(rads)
    y = np.sin(rads)
    # Average the vectors
    mean_x, mean_y = np.mean(x), np.mean(y)
    # The length of the average vector R gives a measure of variance
    R = np.sqrt(mean_x ** 2 + mean_y ** 2)
    # Standard deviation in radians
    std_dev_rad = np.sqrt(-2 * np.log(R))
    return np.rad2deg(std_dev_rad)


def hovering(features: dict[str, Any], threshold: float) -> float:
    # Hovering: Most likely at zero speed and zero z-velocity.
    return _gaussian_score(features["avg_speed"], 0, threshold) * _gaussian_score(abs(features["avg_vz"]), 0, threshold)


def attacking(features: dict[str, Any], threshold: float, steepness: float) -> float:
    # Attacking: Most likely with high z-velocity (approaching).
    return _sigmoid_score(features["avg_vz"], threshold=threshold, steepness=steepness)


def retreating(features: dict[str, Any], threshold: float, steepness: float) -> float:
    # Retreating: The opposite of attacking.
    return _sigmoid_score(-features["avg_vz"], threshold=threshold, steepness=steepness)


def travelling(features: dict[str, Any], threshold: float, steepness: float, direction_sigma: float) -> float:
    # Travelling: High speed AND stable direction.
    speed_score = _sigmoid_score(features["avg_speed"], threshold=threshold, steepness=steepness)
    stability_score = _gaussian_score(features["direction_std"], 0, direction_sigma)  # Low std dev is good
    return speed_score * stability_score


def evading(features: dict[str, Any], threshold: float, steepness: float, direction_sigma: float) -> float:
    # Evading: High speed AND unstable direction.
    speed_score = _sigmoid_score(features["avg_speed"], threshold=threshold, steepness=steepness)
    stability_score = _gaussian_score(features["direction_std"], 0, direction_sigma)  # Low std dev is good

    instability_score = 1 - stability_score
    return speed_score * instability_score


def create_features_from_state_history(state_history: list[dict[str, Any]]) -> dict[str, float]:
    avg_speed = np.mean([s['speed_3d'] for s in state_history])
    avg_vz = np.mean([s['vw'] for s in state_history])  # use vw as an estimate of z velocity
    direction_std = _circular_std_dev([s['direction_xy_radians'] for s in state_history])

    return {"avg_speed": avg_speed,
            "avg_vz": avg_vz,
            "direction_std": direction_std}
