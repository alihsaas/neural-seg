from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "Kvasir-SEG"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
SPLIT_PATH = RESULTS_DIR / "split.json"

DATASET_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
ARCHIVE_NAME = "kvasir-seg.zip"

WORK_SIZE = 256
MASK_THRESHOLD = 128

SPLIT_SEED = 2026
VALIDATION_COUNT = 100
TEST_COUNT = 100
HASH_SIZE = 16
HASH_CROP_MARGIN = 0.2
DUPLICATE_HAMMING_DISTANCE = 40

SEEDS = (0, 1, 2)
BOOTSTRAP_SAMPLES = 2000
