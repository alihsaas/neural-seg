from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from neural_seg.metrics import dice
from neural_seg.postprocess import clean_mask

CHANNELS = ("lab_l", "lab_a", "lab_b", "hsv_s", "hsv_v")
THRESHOLD_GRID = tuple(round(value, 2) for value in np.arange(0.1, 0.91, 0.05))


def channel_image(image: np.ndarray, name: str) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    if name.startswith("lab_"):
        lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
        return lab[..., "lab".index(name[-1])]
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    return hsv[..., {"hsv_s": 1, "hsv_v": 2}[name]]


@dataclass(frozen=True)
class OtsuRule:
    channel: str
    bright: bool

    @property
    def name(self) -> str:
        return f"{self.channel}_{'bright' if self.bright else 'dark'}"

    def score(self, image: np.ndarray) -> np.ndarray:
        values = channel_image(image, self.channel)
        _, above = cv2.threshold(values, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = above.astype(bool) if self.bright else ~above.astype(bool)
        return clean_mask(mask).astype(np.float32)


def otsu_candidates() -> list[OtsuRule]:
    return [OtsuRule(channel, bright) for channel in CHANNELS for bright in (True, False)]


def fit_otsu_rule(images: np.ndarray, masks: np.ndarray) -> tuple[OtsuRule, list[dict]]:
    table = []
    for rule in otsu_candidates():
        scores = [
            dice(rule.score(image) >= 0.5, mask) for image, mask in zip(images, masks, strict=True)
        ]
        table.append({"rule": rule.name, "train_dice": float(np.mean(scores))})
    best = max(range(len(table)), key=lambda index: table[index]["train_dice"])
    return otsu_candidates()[best], table


def local_statistics(channel: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    mean = cv2.GaussianBlur(channel, (0, 0), sigma)
    square = cv2.GaussianBlur(channel * channel, (0, 0), sigma)
    return mean, np.sqrt(np.maximum(square - mean * mean, 0.0))


def pixel_features(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    saturation = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32) / 255.0
    lightness = lab[..., 0]
    features = [lab[..., 0], lab[..., 1], lab[..., 2], saturation]
    for channel in (lightness, lab[..., 1]):
        for sigma in (4.0, 12.0):
            features.extend(local_statistics(channel, sigma))
    gradient_x = cv2.Sobel(lightness, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(lightness, cv2.CV_32F, 0, 1, ksize=3)
    features.append(np.hypot(gradient_x, gradient_y))
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
    rows = rows / (height - 1) - 0.5
    columns = columns / (width - 1) - 0.5
    features.extend([columns, rows, np.hypot(columns, rows)])
    return np.stack(features, axis=-1)


FEATURE_COUNT = 16


def sample_pixels(
    images: np.ndarray, masks: np.ndarray, per_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs, labels = [], []
    for image, mask in zip(images, masks, strict=True):
        features = pixel_features(image).reshape(-1, FEATURE_COUNT)
        flat = mask.reshape(-1)
        for value in (True, False):
            where = np.flatnonzero(flat == value)
            if where.size == 0:
                continue
            chosen = rng.choice(where, size=min(per_class, where.size), replace=False)
            inputs.append(features[chosen])
            labels.append(np.full(chosen.size, value))
    return np.concatenate(inputs), np.concatenate(labels)


@dataclass
class PixelLogistic:
    model: Pipeline
    threshold: float = 0.5

    def probability(self, image: np.ndarray) -> np.ndarray:
        features = pixel_features(image).reshape(-1, FEATURE_COUNT)
        return self.model.predict_proba(features)[:, 1].reshape(image.shape[:2]).astype(np.float32)

    def score(self, image: np.ndarray) -> np.ndarray:
        return clean_mask(self.probability(image) >= self.threshold).astype(np.float32)


def fit_pixel_logistic(
    train_images: np.ndarray,
    train_masks: np.ndarray,
    validation_images: np.ndarray,
    validation_masks: np.ndarray,
    per_class: int = 200,
    seed: int = 0,
) -> tuple[PixelLogistic, list[dict]]:
    inputs, labels = sample_pixels(train_images, train_masks, per_class, seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(inputs, labels)
    segmenter = PixelLogistic(model)
    probabilities = [segmenter.probability(image) for image in validation_images]
    table = []
    for threshold in THRESHOLD_GRID:
        scores = [
            dice(clean_mask(probability >= threshold), mask)
            for probability, mask in zip(probabilities, validation_masks, strict=True)
        ]
        table.append({"threshold": threshold, "validation_dice": float(np.mean(scores))})
    segmenter.threshold = max(table, key=lambda row: row["validation_dice"])["threshold"]
    return segmenter, table
