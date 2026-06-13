import argparse
from dataclasses import asdict

import numpy as np
import torch

from neural_seg.config import CHECKPOINT_DIR, RESULTS_DIR, SEEDS, SPLIT_PATH
from neural_seg.datasets import load_work_set
from neural_seg.evaluation import evaluate_scores, print_summary, summarize, write_csv, write_json
from neural_seg.neural import (
    RunSettings,
    load_checkpoint,
    parameter_count,
    predict,
    save_checkpoint,
    train_model,
)
from neural_seg.splits import load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=RunSettings.epochs)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--reuse", action="store_true")
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    split = load_split(SPLIT_PATH)
    train_images, train_masks = load_work_set(split["train"])
    validation_images, validation_masks = load_work_set(split["validation"])
    test_images, _ = load_work_set(split["test"])
    output = RESULTS_DIR / "learned"

    rows = []
    runs = []
    for seed in arguments.seeds:
        settings = RunSettings(name=f"unet_seed{seed}", seed=seed, epochs=arguments.epochs)
        path = CHECKPOINT_DIR / f"{settings.name}.pt"
        if arguments.reuse and path.exists():
            model, payload = load_checkpoint(path, device)
            if payload["settings"] != asdict(settings):
                raise SystemExit(f"{path} was trained with different settings")
            normalisation = payload["normalisation"]
            history = payload["history"]
            best_epoch = payload["best_epoch"]
            print(f"{settings.name}: reused {path.name}")
        else:
            model, normalisation, history, best_epoch = train_model(
                settings, train_images, train_masks, validation_images, validation_masks, device
            )
            save_checkpoint(path, model, settings, normalisation, history, best_epoch)
        write_csv(history, output / f"{settings.name}_history.csv")
        probabilities = predict(
            model, test_images, normalisation["mean"], normalisation["std"], device
        )
        rows += evaluate_scores(settings.name, split["test"], probabilities, clean=False)
        runs.append(
            {
                "name": settings.name,
                "best_epoch": best_epoch,
                "best_validation_dice": max(entry["validation_dice"] for entry in history),
                "parameters": parameter_count(model),
            }
        )

    for row_index in range(len(split["test"])):
        seed_rows = [
            rows[row_index + offset * len(split["test"])] for offset in range(len(arguments.seeds))
        ]
        rows.append(
            {
                "method": "unet_seed_mean",
                "case_id": seed_rows[0]["case_id"],
                **{
                    key: float(np.mean([row[key] for row in seed_rows]))
                    for key in ("dice", "iou", "precision", "recall", "hd95_px")
                },
                "empty_prediction": any(row["empty_prediction"] for row in seed_rows),
            }
        )
    write_csv(rows, output / "test_cases.csv")
    summary = summarize(rows)
    write_csv(summary, output / "test_summary.csv")
    write_json(
        {"device": str(device), "torch": torch.__version__, "runs": runs}, output / "runs.json"
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
