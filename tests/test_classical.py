import numpy as np
from conftest import synthetic_set

from neural_seg.classical import FEATURE_COUNT, fit_otsu_rule, fit_pixel_logistic, pixel_features
from neural_seg.evaluation import finalize
from neural_seg.metrics import dice
from neural_seg.postprocess import clean_mask, fill_holes, largest_component


def test_fill_holes_and_largest_component() -> None:
    ring = np.zeros((20, 20), bool)
    ring[4:16, 4:16] = True
    ring[8:12, 8:12] = False
    assert fill_holes(ring)[10, 10]
    two = np.zeros((20, 20), bool)
    two[1:3, 1:3] = True
    two[10:18, 10:18] = True
    kept = largest_component(two)
    assert kept[12, 12] and not kept[1, 1]
    assert not largest_component(np.zeros((5, 5), bool)).any()


def test_clean_mask_removes_speckle() -> None:
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    mask[2, 2] = True
    cleaned = clean_mask(mask)
    assert not cleaned[2, 2]
    assert cleaned[20, 20]


def test_feature_shape(rng: np.random.Generator) -> None:
    images, _ = synthetic_set(rng, 1, 32)
    assert pixel_features(images[0]).shape == (32, 32, FEATURE_COUNT)


def test_otsu_rule_finds_red_disc(rng: np.random.Generator) -> None:
    images, masks = synthetic_set(rng, 8)
    rule, table = fit_otsu_rule(images, masks)
    assert len(table) == 10
    scores = [
        dice(rule.score(image) >= 0.5, mask) for image, mask in zip(images, masks, strict=True)
    ]
    assert np.mean(scores) > 0.8


def test_pixel_logistic_learns_red_disc(rng: np.random.Generator) -> None:
    train_images, train_masks = synthetic_set(rng, 12)
    validation_images, validation_masks = synthetic_set(rng, 4)
    segmenter, table = fit_pixel_logistic(
        train_images, train_masks, validation_images, validation_masks, per_class=100
    )
    assert segmenter.threshold in [row["threshold"] for row in table]
    scores = [
        dice(segmenter.score(image) >= 0.5, mask)
        for image, mask in zip(validation_images, validation_masks, strict=True)
    ]
    assert np.mean(scores) > 0.8


def test_finalize_restores_original_shape() -> None:
    score = np.zeros((64, 64), np.float32)
    score[16:48, 16:48] = 1.0
    mask = finalize(score, (100, 150), clean=True)
    assert mask.shape == (100, 150)
    assert mask[50, 75] and not mask[5, 5]
