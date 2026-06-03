import csv
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from neural_seg.config import DATASET_DIR
from neural_seg.datasets import load_mask, probability_to_original
from neural_seg.metrics import bootstrap_interval, evaluate_case
from neural_seg.postprocess import clean_mask


def finalize(
    score: np.ndarray, shape: tuple[int, int], threshold: float = 0.5, clean: bool = False
) -> np.ndarray:
    mask = probability_to_original(score, shape) >= threshold
    if not clean:
        return mask
    kernel = max(3, int(round(5 * min(shape) / score.shape[0])) | 1)
    return clean_mask(mask, kernel)


def evaluate_scores(
    method: str,
    ids: list[str],
    scores: np.ndarray,
    clean: bool,
    dataset_dir: Path = DATASET_DIR,
    threshold: float = 0.5,
) -> list[dict]:
    rows = []
    for case_id, score in zip(ids, scores, strict=True):
        truth = load_mask(case_id, dataset_dir)
        prediction = finalize(score, truth.shape, threshold, clean)
        rows.append(
            {"method": method, "case_id": case_id, **evaluate_case(prediction, truth).row()}
        )
    return rows


def score_all(segment: Callable[[np.ndarray], np.ndarray], images: np.ndarray) -> np.ndarray:
    return np.stack([segment(image) for image in images])


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for method in dict.fromkeys(row["method"] for row in rows):
        chosen = [row for row in rows if row["method"] == method]
        dice_values = np.array([row["dice"] for row in chosen])
        low, high = bootstrap_interval(dice_values)
        summary.append(
            {
                "method": method,
                "images": len(chosen),
                "mean_dice": float(dice_values.mean()),
                "dice_ci_low": low,
                "dice_ci_high": high,
                "mean_iou": float(np.mean([row["iou"] for row in chosen])),
                "mean_precision": float(np.mean([row["precision"] for row in chosen])),
                "mean_recall": float(np.mean([row["recall"] for row in chosen])),
                "median_hd95_px": float(np.median([row["hd95_px"] for row in chosen])),
                "empty_predictions": int(sum(row["empty_prediction"] for row in chosen)),
            }
        )
    return summary


def format_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: format_value(value) for key, value in row.items()} for row in rows)


def write_json(values: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2), encoding="ascii")


def print_summary(summary: list[dict]) -> None:
    print(f"{'method':<24}{'Dice':>8}{'95% CI':>18}{'IoU':>8}{'HD95 px':>10}{'empty':>7}")
    for row in summary:
        interval = f"{row['dice_ci_low']:.3f}-{row['dice_ci_high']:.3f}"
        print(
            f"{row['method']:<24}{row['mean_dice']:>8.3f}{interval:>18}{row['mean_iou']:>8.3f}"
            f"{row['median_hd95_px']:>10.1f}{row['empty_predictions']:>7d}"
        )
