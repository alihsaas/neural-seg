import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_seg.classical import fit_pixel_logistic
from neural_seg.config import CHECKPOINT_DIR, RESULTS_DIR, SPLIT_PATH
from neural_seg.datasets import load_image, load_mask, load_work_set
from neural_seg.evaluation import finalize
from neural_seg.metrics import dice
from neural_seg.neural import load_checkpoint, predict
from neural_seg.splits import load_split

SEED_SHOWN = 0
RANKS = (
    ("best", 1.0),
    ("upper quartile", 0.75),
    ("median", 0.5),
    ("lower quartile", 0.25),
    ("worst", 0.0),
)
TRUTH_COLOUR = (255, 212, 0)
PREDICTION_COLOUR = (42, 127, 255)
DISPLAY_WIDTH = 480


def saved_dice(path, method: str) -> dict[str, float]:
    with path.open(encoding="ascii") as handle:
        return {
            row["case_id"]: float(row["dice"])
            for row in csv.DictReader(handle)
            if row["method"] == method
        }


def pick_cases(scores: dict[str, float]) -> list[tuple[str, str]]:
    ordered = sorted(scores, key=lambda case_id: (scores[case_id], case_id))
    last = len(ordered) - 1
    return [(label, ordered[int(round(position * last))]) for label, position in RANKS]


def outline(image: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> None:
    thickness = max(2, int(round(min(image.shape[:2]) / 160)))
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, colour, thickness)


def panel(image: np.ndarray, truth: np.ndarray, prediction: np.ndarray | None) -> np.ndarray:
    canvas = image.copy()
    if prediction is not None:
        tint = canvas.copy()
        tint[prediction] = PREDICTION_COLOUR
        canvas = cv2.addWeighted(tint, 0.45, canvas, 0.55, 0)
        outline(canvas, prediction, PREDICTION_COLOUR)
    outline(canvas, truth, TRUTH_COLOUR)
    height = int(round(canvas.shape[0] * DISPLAY_WIDTH / canvas.shape[1]))
    return cv2.resize(canvas, (DISPLAY_WIDTH, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_split(SPLIT_PATH)
    unet_name = f"unet_seed{SEED_SHOWN}"
    unet_saved = saved_dice(RESULTS_DIR / "learned" / "test_cases.csv", unet_name)
    logistic_saved = saved_dice(RESULTS_DIR / "classical" / "test_cases.csv", "pixel_logistic")
    cases = pick_cases(unet_saved)

    train_images, train_masks = load_work_set(split["train"])
    validation_images, validation_masks = load_work_set(split["validation"])
    logistic, _ = fit_pixel_logistic(train_images, train_masks, validation_images, validation_masks)
    del train_images, train_masks, validation_images, validation_masks
    model, payload = load_checkpoint(CHECKPOINT_DIR / f"{unet_name}.pt", device)
    normalisation = payload["normalisation"]
    test_images, _ = load_work_set(split["test"])
    unet_scores = predict(model, test_images, normalisation["mean"], normalisation["std"], device)
    position = {case_id: index for index, case_id in enumerate(split["test"])}

    figure, axes = plt.subplots(
        len(cases), 3, figsize=(10.5, 2.6 * len(cases)), layout="constrained"
    )
    for row, (label, case_id) in enumerate(cases):
        image = load_image(case_id)
        truth = load_mask(case_id)
        index = position[case_id]
        unet_mask = finalize(unet_scores[index], truth.shape)
        logistic_mask = finalize(logistic.score(test_images[index]), truth.shape, clean=True)
        unet_dice, logistic_dice = dice(unet_mask, truth), dice(logistic_mask, truth)
        for measured, saved in (
            (unet_dice, unet_saved[case_id]),
            (logistic_dice, logistic_saved[case_id]),
        ):
            if abs(measured - saved) > 1e-4:
                raise SystemExit(f"{case_id}: Dice {measured:.4f} differs from saved {saved:.4f}")
        panels = (
            (panel(image, truth, None), f"{label} U-Net case\ntrue outline"),
            (panel(image, truth, logistic_mask), f"pixel logistic\nDice {logistic_dice:.3f}"),
            (panel(image, truth, unet_mask), f"U-Net seed {SEED_SHOWN}\nDice {unet_dice:.3f}"),
        )
        for column, (picture, title) in enumerate(panels):
            axes[row, column].imshow(picture)
            axes[row, column].set_title(title, fontsize=9)
            axes[row, column].axis("off")
        print(f"{label:15s} {case_id}  logistic {logistic_dice:.4f}  U-Net {unet_dice:.4f}")

    figure.suptitle("Test images: true outline in yellow, prediction in blue", fontsize=11)
    output = RESULTS_DIR / "figures" / "test_examples.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=110)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
