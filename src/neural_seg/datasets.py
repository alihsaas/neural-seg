import hashlib
import json
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import truststore

from neural_seg.config import (
    ARCHIVE_NAME,
    DATA_DIR,
    DATASET_DIR,
    DATASET_URL,
    MASK_THRESHOLD,
    WORK_SIZE,
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_data(data_dir: Path = DATA_DIR, url: str = DATASET_URL) -> Path:
    dataset_dir = data_dir / DATASET_DIR.name
    if (dataset_dir / "images").is_dir() and (dataset_dir / "masks").is_dir():
        print(f"Dataset already present: {dataset_dir}")
        return dataset_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / ARCHIVE_NAME
    if not archive.exists():
        print(f"Downloading {url}")
        partial = archive.with_suffix(".part")
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        with (
            urllib.request.urlopen(url, context=context) as response,
            partial.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
        partial.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            target = (data_dir / member).resolve()
            if not target.is_relative_to(data_dir.resolve()):
                raise ValueError(f"Unsafe path in archive: {member}")
        bundle.extractall(data_dir)
    record = {
        "url": url,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_of(archive),
        "images": len(case_ids(dataset_dir)),
    }
    (data_dir / "download.json").write_text(json.dumps(record, indent=2), encoding="ascii")
    print(f"Extracted {record['images']} images to {dataset_dir}")
    return dataset_dir


def case_ids(dataset_dir: Path = DATASET_DIR) -> list[str]:
    images = {path.stem for path in (dataset_dir / "images").glob("*.jpg")}
    masks = {path.stem for path in (dataset_dir / "masks").glob("*.jpg")}
    if images != masks:
        raise ValueError(f"{len(images ^ masks)} images and masks do not pair up")
    return sorted(images)


def load_image(case_id: str, dataset_dir: Path = DATASET_DIR) -> np.ndarray:
    bgr = cv2.imread(str(dataset_dir / "images" / f"{case_id}.jpg"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(case_id)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(case_id: str, dataset_dir: Path = DATASET_DIR) -> np.ndarray:
    gray = cv2.imread(str(dataset_dir / "masks" / f"{case_id}.jpg"), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(case_id)
    return gray >= MASK_THRESHOLD


def to_work_size(image: np.ndarray, size: int = WORK_SIZE) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def mask_to_work_size(mask: np.ndarray, size: int = WORK_SIZE) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return resized >= 0.5


def probability_to_original(probability: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return cv2.resize(
        probability.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    )


def load_work_set(
    ids: list[str], dataset_dir: Path = DATASET_DIR, size: int = WORK_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    images = np.empty((len(ids), size, size, 3), dtype=np.uint8)
    masks = np.empty((len(ids), size, size), dtype=bool)
    for index, case_id in enumerate(ids):
        images[index] = to_work_size(load_image(case_id, dataset_dir), size)
        masks[index] = mask_to_work_size(load_mask(case_id, dataset_dir), size)
    return images, masks
