import cv2
import numpy as np
import pytest


def synthetic_case(rng: np.random.Generator, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    image = np.empty((size, size, 3), np.uint8)
    image[:] = (150, 90, 80)
    image = np.clip(image + rng.normal(0, 6, image.shape), 0, 255).astype(np.uint8)
    mask = np.zeros((size, size), np.uint8)
    center = (
        int(rng.integers(size // 3, 2 * size // 3)),
        int(rng.integers(size // 3, 2 * size // 3)),
    )
    radius = int(rng.integers(size // 8, size // 5))
    cv2.circle(mask, center, radius, 1, -1)
    image[mask.astype(bool)] = np.clip(image[mask.astype(bool)] + np.array([70, 10, 10]), 0, 255)
    return image, mask.astype(bool)


def synthetic_set(
    rng: np.random.Generator, count: int, size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    pairs = [synthetic_case(rng, size) for _ in range(count)]
    return np.stack([pair[0] for pair in pairs]), np.stack([pair[1] for pair in pairs])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)
