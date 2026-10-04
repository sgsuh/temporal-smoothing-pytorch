import os

import numpy as np
import pytest
from PIL import Image

ROAD, SKY, CAR = (128, 64, 128), (128, 128, 128), (64, 0, 128)
LANE = (128, 0, 192)  # merged into Road in camvid-11


def write_png(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


@pytest.fixture
def camvid_root(tmp_path):
    """Tiny CamVid-like dataset: 8 x 12 images, test labels at frames 3 and 7 of two sequences."""
    rng = np.random.default_rng(0)
    label = np.zeros((8, 12, 3), dtype=np.uint8)
    label[:4] = SKY
    label[4:, :6] = ROAD
    label[4:, 6:] = LANE
    label[0, 0] = (1, 2, 3)  # unknown color -> void
    splits = {
        "train": ["0001TP_000030", "0006R0_f00060"],
        "val": ["0016E5_00090"],
        "test": ["0001TP_000003", "Seq05VD_f00007"],
    }
    for split, names in splits.items():
        for name in names:
            write_png(tmp_path / split / f"{name}.png", rng.integers(0, 255, (8, 12, 3)))
            write_png(tmp_path / f"{split}_labels" / f"{name}_L.png", label)
    for seq, fmt in (("0001TP", "{:06d}"), ("Seq05VD", "f{:05d}")):
        for i in range(10):
            write_png(tmp_path / "seq" / f"{seq}_{fmt.format(i)}.png", np.full((8, 12, 3), i * 10))
    return str(tmp_path)
