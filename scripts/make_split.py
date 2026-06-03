from neural_seg.config import SPLIT_PATH
from neural_seg.datasets import case_ids, load_image
from neural_seg.splits import difference_hash, grouped_split, near_duplicate_groups, save_split


def main() -> None:
    ids = case_ids()
    hashes = [difference_hash(load_image(case_id)) for case_id in ids]
    groups = near_duplicate_groups(hashes)
    split = grouped_split(ids, groups)
    save_split(split, dict(zip(ids, groups, strict=True)), SPLIT_PATH)
    duplicates = len(ids) - len(set(groups))
    print(
        f"{len(ids)} images, {len(set(groups))} groups ({duplicates} images merged as near-duplicates)"
    )
    print({name: len(values) for name, values in split.items()}, "->", SPLIT_PATH)


if __name__ == "__main__":
    main()
