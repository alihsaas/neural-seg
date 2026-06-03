import numpy as np
import pytest

from neural_seg.metrics import (
    bootstrap_interval,
    dice,
    evaluate_case,
    hd95,
    iou,
    paired_difference_interval,
)


def square(size: int, top: int, left: int, side: int) -> np.ndarray:
    mask = np.zeros((size, size), bool)
    mask[top : top + side, left : left + side] = True
    return mask


def test_dice_and_iou_on_known_overlap() -> None:
    first = square(20, 0, 0, 10)
    second = square(20, 0, 5, 10)
    assert dice(first, second) == pytest.approx(2 * 50 / 200)
    assert iou(first, second) == pytest.approx(50 / 150)


def test_identical_masks_are_perfect() -> None:
    mask = square(32, 4, 6, 12)
    result = evaluate_case(mask, mask)
    assert result.dice == 1.0
    assert result.iou == 1.0
    assert result.hd95_px == 0.0
    assert not result.empty_prediction


def test_hd95_of_shifted_square_is_the_shift() -> None:
    truth = square(64, 10, 10, 20)
    prediction = square(64, 10, 13, 20)
    assert hd95(prediction, truth) == pytest.approx(3.0, abs=0.05)


def test_empty_prediction_scores_zero_and_diagonal() -> None:
    truth = square(30, 5, 5, 10)
    result = evaluate_case(np.zeros_like(truth), truth)
    assert result.dice == 0.0
    assert result.recall == 0.0
    assert result.empty_prediction
    assert result.hd95_px == pytest.approx(np.hypot(30, 30))


def test_both_empty_is_perfect() -> None:
    empty = np.zeros((8, 8), bool)
    assert dice(empty, empty) == 1.0
    assert hd95(empty, empty) == 0.0


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate_case(np.zeros((4, 4), bool), np.zeros((4, 5), bool))


def test_bootstrap_interval_contains_mean(rng: np.random.Generator) -> None:
    values = rng.uniform(0.6, 0.9, size=200)
    low, high = bootstrap_interval(values)
    assert low < values.mean() < high
    assert high - low < 0.05


def test_paired_difference_sign(rng: np.random.Generator) -> None:
    base = rng.uniform(0.4, 0.8, size=100)
    mean, low, high = paired_difference_interval(base + 0.1, base)
    assert mean == pytest.approx(0.1)
    assert low > 0
