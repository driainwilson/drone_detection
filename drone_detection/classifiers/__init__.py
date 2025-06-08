import enum
from functools import partial
from typing import Callable

from omegaconf import DictConfig

from .prob import *


class ClassifierType(enum.Enum):
    Hovering = "Hovering"
    Travelling = "Travelling"
    Evading = "Evading"
    Attacking = "Attacking"
    Retreating = "Retreating"


CLASSIFIER_FACTORY: dict[ClassifierType, Callable] = {
    ClassifierType.Hovering: hovering,
    ClassifierType.Travelling: travelling,
    ClassifierType.Evading: evading,
    ClassifierType.Attacking: attacking,
    ClassifierType.Retreating: retreating,
}


def _behaviour_classifier(classifiers: dict[str, Callable], state_history: list[dict[str, Any]]) -> dict[str, float]:
    features = create_features_from_state_history(state_history)

    scores = {}
    for name, classifier in classifiers.items():
        scores[name] = classifier(features)

    # --- Normalize scores to get a probability distribution ---
    total_score = sum(scores.values())
    if total_score < 1e-9:
        return {b: 0.0 for b in classifiers.keys()}

    return {b: s / total_score for b, s in scores.items()}


def _threat_score_calculator(state: dict[str, Any],
                            behavior_probs: dict[str, float],
                            attacking_weight: float,
                            proximity_weight: float,
                            approach_velocity_weight: float,
                            proximity_threshold: float,
                            approach_threshold: float) -> float:


    # 1. Threat from behavior (the probability of "Attacking")
    threat_from_behavior = behavior_probs.get("Attacking", 0.0)

    # 2. Threat from proximity (proxy is area)
    threat_from_proximity = 1 - math.exp(-state['area'] / proximity_threshold)

    # 3. Threat from approach velocity ( vw is our proxy for this)
    threat_from_approach = 1 - math.exp(-max(0, state['vw']) / approach_threshold)

    raw_threat = (threat_from_behavior * attacking_weight +
                  threat_from_proximity * proximity_weight +
                  threat_from_approach * approach_velocity_weight)

    return min(100, raw_threat * 100)


def create(cfg: DictConfig) -> tuple[Callable, Callable]:
    # create classifiers
    classifiers = {}
    for classifier in cfg.types:
        name = classifier.name
        if classifier.name not in ClassifierType.__members__:
            raise ValueError(f"Unknown classifier type: {classifier.name}")
        classifier_type = ClassifierType[classifier.name]
        params = dict(classifier)
        params.pop("name")

        classifier_func = CLASSIFIER_FACTORY[classifier_type]
        classifiers[name] = partial(classifier_func, **params)

    behaviour_classifier = partial(_behaviour_classifier, classifiers=classifiers)

    threat_score_calculator = partial(_threat_score_calculator, **cfg.threat_score)

    return behaviour_classifier, threat_score_calculator
