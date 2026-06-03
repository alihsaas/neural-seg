import json
from pathlib import Path

import cv2
import numpy as np

from neural_seg.config import (
    DUPLICATE_HAMMING_DISTANCE,
    HASH_CROP_MARGIN,
    HASH_SIZE,
    SPLIT_SEED,
    TEST_COUNT,
    VALIDATION_COUNT,
)


def central_crop(image: np.ndarray, margin: float = HASH_CROP_MARGIN) -> np.ndarray:
    height, width = image.shape[:2]
    top, left = int(margin * height), int(margin * width)
    return image[top : height - top, left : width - left]


def difference_hash(image: np.ndarray, size: int = HASH_SIZE) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    small = cv2.resize(central_crop(gray), (size + 1, size), interpolation=cv2.INTER_AREA)
    small = small.astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << index for index, bit in enumerate(bits) if bit))


def hamming(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def near_duplicate_groups(
    hashes: list[int], max_distance: int = DUPLICATE_HAMMING_DISTANCE
) -> list[int]:
    parent = list(range(len(hashes)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first in range(len(hashes)):
        for second in range(first + 1, len(hashes)):
            if hamming(hashes[first], hashes[second]) <= max_distance:
                parent[root(second)] = root(first)
    roots = [root(index) for index in range(len(hashes))]
    labels = {value: label for label, value in enumerate(dict.fromkeys(roots))}
    return [labels[value] for value in roots]


def grouped_split(
    ids: list[str],
    groups: list[int],
    validation_count: int = VALIDATION_COUNT,
    test_count: int = TEST_COUNT,
    seed: int = SPLIT_SEED,
) -> dict[str, list[str]]:
    members: dict[int, list[str]] = {}
    for case_id, group in zip(ids, groups, strict=True):
        members.setdefault(group, []).append(case_id)
    order = np.random.default_rng(seed).permutation(sorted(members))
    split: dict[str, list[str]] = {"test": [], "validation": [], "train": []}
    for group in order:
        if len(split["test"]) < test_count:
            split["test"].extend(members[group])
        elif len(split["validation"]) < validation_count:
            split["validation"].extend(members[group])
        else:
            split["train"].extend(members[group])
    return {name: sorted(values) for name, values in split.items()}


def save_split(split: dict[str, list[str]], groups: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for group in groups.values():
        sizes[group] = sizes.get(group, 0) + 1
    payload = {
        "seed": SPLIT_SEED,
        "duplicate_hamming_distance": DUPLICATE_HAMMING_DISTANCE,
        "counts": {name: len(values) for name, values in split.items()},
        "duplicate_groups": sum(1 for size in sizes.values() if size > 1),
        "images_in_duplicate_groups": sum(size for size in sizes.values() if size > 1),
        **split,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="ascii")


def load_split(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="ascii"))
    return {name: payload[name] for name in ("train", "validation", "test")}
