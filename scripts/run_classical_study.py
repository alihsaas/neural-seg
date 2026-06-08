from neural_seg.classical import fit_otsu_rule, fit_pixel_logistic
from neural_seg.config import RESULTS_DIR, SPLIT_PATH
from neural_seg.datasets import load_work_set
from neural_seg.evaluation import evaluate_scores, print_summary, score_all, summarize, write_csv
from neural_seg.splits import load_split


def main() -> None:
    split = load_split(SPLIT_PATH)
    train_images, train_masks = load_work_set(split["train"])
    validation_images, validation_masks = load_work_set(split["validation"])
    test_images, _ = load_work_set(split["test"])
    output = RESULTS_DIR / "classical"

    rule, rule_table = fit_otsu_rule(train_images, train_masks)
    write_csv(rule_table, output / "otsu_rule_selection.csv")
    print(f"Otsu rule chosen on the training images: {rule.name}")

    logistic, threshold_table = fit_pixel_logistic(
        train_images, train_masks, validation_images, validation_masks
    )
    write_csv(threshold_table, output / "logistic_threshold_selection.csv")
    print(f"Pixel logistic threshold chosen on validation: {logistic.threshold}")

    rows = evaluate_scores(
        "otsu_" + rule.name, split["test"], score_all(rule.score, test_images), clean=True
    )
    rows += evaluate_scores(
        "pixel_logistic", split["test"], score_all(logistic.score, test_images), clean=True
    )
    write_csv(rows, output / "test_cases.csv")
    summary = summarize(rows)
    write_csv(summary, output / "test_summary.csv")
    print_summary(summary)


if __name__ == "__main__":
    main()
