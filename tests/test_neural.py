import numpy as np
import torch
from conftest import synthetic_set

from neural_seg.neural import (
    RunSettings,
    UNet,
    augment,
    load_checkpoint,
    parameter_count,
    predict,
    save_checkpoint,
    soft_dice_loss,
    train_model,
)


def test_unet_keeps_spatial_size() -> None:
    model = UNet(width=8)
    output = model(torch.zeros(2, 3, 64, 48))
    assert output.shape == (2, 64, 48)


def test_default_unet_size() -> None:
    assert 7_000_000 < parameter_count(UNet()) < 9_000_000


def test_soft_dice_loss_is_zero_for_confident_match() -> None:
    labels = torch.zeros(1, 8, 8)
    labels[0, 2:6, 2:6] = 1
    logits = (labels * 2 - 1) * 50
    assert float(soft_dice_loss(logits, labels)) < 1e-3


def test_augment_moves_image_and_mask_together() -> None:
    generator = torch.Generator().manual_seed(3)
    labels = torch.zeros(1, 16, 16)
    labels[0, 1:4, 2:9] = 1
    inputs = labels.unsqueeze(1).repeat(1, 3, 1, 1)
    for _ in range(8):
        moved_inputs, moved_labels = augment(inputs, labels, generator)
        foreground = moved_inputs[:, 0] > moved_inputs[:, 0].mean()
        assert torch.equal(foreground, moved_labels.bool())


def test_short_training_learns_synthetic_discs(tmp_path, rng: np.random.Generator) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_images, train_masks = synthetic_set(rng, 32)
    validation_images, validation_masks = synthetic_set(rng, 8)
    settings = RunSettings(
        name="smoke", seed=0, width=8, epochs=15, batch_size=8, learning_rate=3e-3
    )
    model, normalisation, history, best_epoch = train_model(
        settings,
        train_images,
        train_masks,
        validation_images,
        validation_masks,
        device,
        log=lambda _: None,
    )
    assert history[best_epoch - 1]["validation_dice"] > 0.7
    path = tmp_path / "smoke.pt"
    save_checkpoint(path, model, settings, normalisation, history, best_epoch)
    reloaded, payload = load_checkpoint(path, device)
    first = predict(model, validation_images, normalisation["mean"], normalisation["std"], device)
    second = predict(
        reloaded,
        validation_images,
        payload["normalisation"]["mean"],
        payload["normalisation"]["std"],
        device,
    )
    assert np.allclose(first, second, atol=1e-5)
