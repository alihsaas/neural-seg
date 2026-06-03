from dataclasses import asdict, dataclass

import cv2
import numpy as np

from neural_seg.config import BOOTSTRAP_SAMPLES


@dataclass(frozen=True)
class CaseMetrics:
    dice: float
    iou: float
    precision: float
    recall: float
    hd95_px: float
    empty_prediction: bool

    def row(self) -> dict:
        return asdict(self)


def dice(prediction: np.ndarray, truth: np.ndarray) -> float:
    total = np.count_nonzero(prediction) + np.count_nonzero(truth)
    if total == 0:
        return 1.0
    return 2.0 * np.count_nonzero(prediction & truth) / total


def iou(prediction: np.ndarray, truth: np.ndarray) -> float:
    union = np.count_nonzero(prediction | truth)
    if union == 0:
        return 1.0
    return np.count_nonzero(prediction & truth) / union


def precision(prediction: np.ndarray, truth: np.ndarray) -> float:
    predicted = np.count_nonzero(prediction)
    if predicted == 0:
        return 1.0 if not truth.any() else 0.0
    return np.count_nonzero(prediction & truth) / predicted


def recall(prediction: np.ndarray, truth: np.ndarray) -> float:
    actual = np.count_nonzero(truth)
    if actual == 0:
        return 1.0
    return np.count_nonzero(prediction & truth) / actual


def boundary(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(
        mask.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return mask & ~eroded.astype(bool)


def distance_to(edge: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def hd95(prediction: np.ndarray, truth: np.ndarray) -> float:
    if not prediction.any() and not truth.any():
        return 0.0
    if not prediction.any() or not truth.any():
        return float(np.hypot(*prediction.shape))
    predicted_edge = boundary(prediction)
    true_edge = boundary(truth)
    to_truth = distance_to(true_edge)[predicted_edge]
    to_prediction = distance_to(predicted_edge)[true_edge]
    return float(max(np.percentile(to_truth, 95), np.percentile(to_prediction, 95)))


def evaluate_case(prediction: np.ndarray, truth: np.ndarray) -> CaseMetrics:
    if prediction.shape != truth.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} and {truth.shape}")
    prediction = prediction.astype(bool)
    truth = truth.astype(bool)
    return CaseMetrics(
        dice=dice(prediction, truth),
        iou=iou(prediction, truth),
        precision=precision(prediction, truth),
        recall=recall(prediction, truth),
        hd95_px=hd95(prediction, truth),
        empty_prediction=not prediction.any(),
    )


def bootstrap_interval(
    values: np.ndarray, samples: int = BOOTSTRAP_SAMPLES, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, values.size, size=(samples, values.size))
    means = values[picks].mean(axis=1)
    tail = 100.0 * (1.0 - level) / 2.0
    return float(np.percentile(means, tail)), float(np.percentile(means, 100.0 - tail))


def paired_difference_interval(
    first: np.ndarray, second: np.ndarray, samples: int = BOOTSTRAP_SAMPLES, seed: int = 0
) -> tuple[float, float, float]:
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    low, high = bootstrap_interval(difference, samples, seed)
    return float(difference.mean()), low, high
