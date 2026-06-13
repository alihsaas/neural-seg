from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class RunSettings:
    name: str
    seed: int
    width: int = 32
    epochs: int = 60
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


def double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, width: int = 32, levels: int = 4):
        super().__init__()
        channels = [width * 2**level for level in range(levels + 1)]
        self.encoders = nn.ModuleList(
            [double_conv(3, channels[0])]
            + [double_conv(channels[level], channels[level + 1]) for level in range(levels)]
        )
        self.decoders = nn.ModuleList(
            [
                double_conv(channels[level + 1] + channels[level], channels[level])
                for level in reversed(range(levels))
            ]
        )
        self.head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        skips = []
        for index, encoder in enumerate(self.encoders):
            values = encoder(values if index == 0 else functional.max_pool2d(values, 2))
            skips.append(values)
        values = skips.pop()
        for decoder in self.decoders:
            skip = skips.pop()
            values = functional.interpolate(
                values, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            values = decoder(torch.cat([values, skip], dim=1))
        return self.head(values).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def seed_everything(seed: int, device: torch.device) -> torch.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def channel_statistics(images: np.ndarray) -> tuple[list[float], list[float]]:
    levels = np.arange(256, dtype=np.float64) / 255.0
    means, stds = [], []
    for channel in range(3):
        counts = np.bincount(images[..., channel].ravel(), minlength=256).astype(np.float64)
        weights = counts / counts.sum()
        mean = float(weights @ levels)
        means.append(mean)
        stds.append(float(np.sqrt(weights @ (levels - mean) ** 2)))
    return means, stds


def to_tensor(
    images: np.ndarray, mean: list[float], std: list[float], device: torch.device
) -> torch.Tensor:
    values = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 255.0
    shift = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    scale = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (values - shift) / scale


def soft_dice_loss(logits: torch.Tensor, labels: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    overlap = (probability * labels).sum(dim=(1, 2))
    total = probability.sum(dim=(1, 2)) + labels.sum(dim=(1, 2))
    return (1.0 - (2.0 * overlap + smooth) / (total + smooth)).mean()


def segmentation_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return functional.binary_cross_entropy_with_logits(logits, labels) + soft_dice_loss(
        logits, labels
    )


def augment(
    inputs: torch.Tensor, labels: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    device = inputs.device
    draws = torch.rand(4, generator=generator, device=device).tolist()
    if draws[0] < 0.5:
        inputs, labels = inputs.flip(-1), labels.flip(-1)
    if draws[1] < 0.5:
        inputs, labels = inputs.flip(-2), labels.flip(-2)
    turns = int(draws[2] * 4)
    inputs, labels = (
        torch.rot90(inputs, turns, dims=(-2, -1)),
        torch.rot90(labels, turns, dims=(-2, -1)),
    )
    gain = 1.0 + 0.2 * (
        torch.rand(inputs.shape[0], 1, 1, 1, generator=generator, device=device) - 0.5
    )
    offset = 0.2 * (torch.rand(inputs.shape[0], 1, 1, 1, generator=generator, device=device) - 0.5)
    return inputs * gain + offset, labels


@torch.no_grad()
def predict(
    model: nn.Module,
    images: np.ndarray,
    mean: list[float],
    std: list[float],
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, images.shape[0], batch_size):
        batch = to_tensor(images[start : start + batch_size], mean, std, device)
        outputs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def mean_dice_at(probabilities: np.ndarray, masks: np.ndarray, threshold: float = 0.5) -> float:
    predicted = probabilities >= threshold
    overlap = (predicted & masks).sum(axis=(1, 2))
    total = predicted.sum(axis=(1, 2)) + masks.sum(axis=(1, 2))
    scores = np.where(total == 0, 1.0, 2.0 * overlap / np.maximum(total, 1))
    return float(scores.mean())


def train_model(
    settings: RunSettings,
    train_images: np.ndarray,
    train_masks: np.ndarray,
    validation_images: np.ndarray,
    validation_masks: np.ndarray,
    device: torch.device,
    log=print,
):
    generator = seed_everything(settings.seed, device)
    mean, std = channel_statistics(train_images)
    model = UNet(settings.width).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs)
    labels_all = torch.from_numpy(train_masks.astype(np.float32))
    history = []
    best_state, best_epoch, best_dice = None, -1, -1.0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        order = (
            torch.randperm(train_images.shape[0], generator=generator, device=device).cpu().numpy()
        )
        running, batches = 0.0, 0
        for start in range(0, order.size, settings.batch_size):
            batch = np.sort(order[start : start + settings.batch_size])
            if batch.size < 2:
                continue
            inputs = to_tensor(train_images[batch], mean, std, device)
            labels = labels_all[batch].to(device)
            inputs, labels = augment(inputs, labels, generator)
            loss = segmentation_loss(model(inputs), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            batches += 1
        scheduler.step()
        validation_dice = mean_dice_at(
            predict(model, validation_images, mean, std, device), validation_masks
        )
        history.append(
            {"epoch": epoch, "train_loss": running / batches, "validation_dice": validation_dice}
        )
        log(
            f"{settings.name} epoch {epoch:3d}  loss {running / batches:.4f}  validation Dice {validation_dice:.4f}"
        )
        if validation_dice > best_dice:
            best_dice, best_epoch = validation_dice, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"mean": mean, "std": std}, history, best_epoch


def save_checkpoint(
    path: Path,
    model: nn.Module,
    settings: RunSettings,
    normalisation: dict,
    history: list[dict],
    best_epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "settings": asdict(settings),
            "normalisation": normalisation,
            "history": history,
            "best_epoch": best_epoch,
            "state": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = UNet(payload["settings"]["width"]).to(device)
    model.load_state_dict(payload["state"])
    model.eval()
    return model, payload
