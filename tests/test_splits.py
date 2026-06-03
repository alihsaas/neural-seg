import numpy as np

from neural_seg.splits import difference_hash, grouped_split, hamming, near_duplicate_groups


def test_near_duplicate_has_small_hash_distance(rng: np.random.Generator) -> None:
    image = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    noisy = np.clip(image.astype(int) + rng.integers(-3, 4, image.shape), 0, 255).astype(np.uint8)
    other = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    assert hamming(difference_hash(image), difference_hash(noisy)) <= 40
    assert hamming(difference_hash(image), difference_hash(other)) > 40


def test_hash_ignores_the_frame_border(rng: np.random.Generator) -> None:
    first = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    second = first.copy()
    second[:15] = 0
    second[:, :15] = 0
    second[-15:, :30] = 255
    assert hamming(difference_hash(first), difference_hash(second)) == 0


def test_groups_are_transitive() -> None:
    assert near_duplicate_groups([0b0000, 0b0001, 0b0011, 0xFFFF], max_distance=1) == [0, 0, 0, 1]


def test_grouped_split_keeps_groups_together_and_is_disjoint() -> None:
    ids = [f"case{index:03d}" for index in range(300)]
    groups = [index // 3 for index in range(300)]
    split = grouped_split(ids, groups, validation_count=30, test_count=30, seed=1)
    sets = {name: set(values) for name, values in split.items()}
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["train"].isdisjoint(sets["validation"])
    assert sets["validation"].isdisjoint(sets["test"])
    assert sum(len(values) for values in sets.values()) == 300
    group_of = dict(zip(ids, groups, strict=True))
    for name, values in sets.items():
        for other, other_values in sets.items():
            if name != other:
                assert {group_of[i] for i in values}.isdisjoint({group_of[i] for i in other_values})


def test_grouped_split_is_reproducible() -> None:
    ids = [f"c{index}" for index in range(50)]
    groups = list(range(50))
    assert grouped_split(ids, groups, 10, 10, 3) == grouped_split(ids, groups, 10, 10, 3)
