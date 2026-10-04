import os

import numpy as np
import pytest
import torch
from PIL import Image

from temporal_smoothing.data import (
    CamVid,
    Frame,
    SequenceWindowDataset,
    camvid_sequence,
    class_names,
    color_map,
    colors_to_index,
    median_frequency_weights,
    memorized_class_weights,
    num_classes,
    parse_camvid_name,
    random_crop_flip,
    resize_nearest,
)

CAMVID_ROOT = os.environ.get("CAMVID_ROOT", "")
needs_camvid = pytest.mark.skipif(not os.path.isdir(CAMVID_ROOT), reason="CamVid not available")

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
    for split, names in {"train": ["0001TP_000030", "0006R0_f00060"], "test": ["0001TP_000003", "Seq05VD_f00007"]}.items():
        for name in names:
            write_png(tmp_path / split / f"{name}.png", rng.integers(0, 255, (8, 12, 3)))
            write_png(tmp_path / f"{split}_labels" / f"{name}_L.png", label)
    for seq, fmt in (("0001TP", "{:06d}"), ("Seq05VD", "f{:05d}")):
        for i in range(10):
            write_png(tmp_path / "seq" / f"{seq}_{fmt.format(i)}.png", np.full((8, 12, 3), i * 10))
    return str(tmp_path)


# Labels


def test_parse_camvid_name():
    assert parse_camvid_name("0001TP_006690.png") == ("0001TP", 6690)
    assert parse_camvid_name("/a/b/Seq05VD_f05100_L.png") == ("Seq05VD", 5100)
    with pytest.raises(ValueError):
        parse_camvid_name("foo.png")


def test_camvid_11_mapping():
    colors = color_map("camvid-11")
    assert colors[ROAD] == colors[LANE] == 3
    assert colors[SKY] == 0 and colors[CAR] == 8
    assert (0, 0, 0) not in colors
    assert class_names("camvid") == [
        "Sky", "Building", "Pole", "Road", "Pavement", "Tree",
        "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist",
    ]
    assert num_classes("camvid-11") == len(memorized_class_weights("camvid-11")) == 11


def test_camvid_31_mapping():
    colors = color_map("camvid-31")
    assert len(colors) == num_classes("camvid-31") == len(memorized_class_weights("camvid-31")) == 31
    assert sorted(colors.values()) == list(range(31))
    assert colors[ROAD] != colors[LANE]
    with pytest.raises(ValueError):
        color_map("cityscapes")


def test_colors_to_index():
    rgb = torch.tensor([ROAD, LANE, CAR, (0, 0, 0), (1, 2, 3)], dtype=torch.uint8).t().reshape(3, 1, 5)
    assert colors_to_index(rgb, color_map("camvid-11")).tolist() == [[3, 3, 8, -1, -1]]


# Transforms


def test_resize_nearest_samples_pixel_centers():
    # TF2 nearest resize samples floor((i + 0.5) * scale): indices 1 and 3 when halving 4 pixels.
    x = torch.arange(16, dtype=torch.uint8).reshape(1, 4, 4)
    assert resize_nearest(x, (2, 2)).tolist() == [[[5, 7], [13, 15]]]
    assert resize_nearest(x, None) is x


def test_random_crop_flip_keeps_alignment():
    gen = torch.Generator().manual_seed(0)
    label = torch.arange(6 * 8).reshape(6, 8)
    image = label.float().expand(3, -1, -1)
    for _ in range(20):
        img, lab = random_crop_flip(image, label, (4, 5), flip=True, generator=gen)
        assert img.shape == (3, 4, 5) and lab.shape == (4, 5)
        assert torch.equal(img[0].long(), lab)
    with pytest.raises(ValueError):
        random_crop_flip(image, label, (7, 5))


# CamVid


def test_camvid_dataset(camvid_root):
    ds = CamVid(camvid_root, "train", size=(4, 6))
    assert len(ds) == 2
    image, label = ds[0]
    assert image.shape == (3, 4, 6) and image.dtype == torch.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert label.shape == (4, 6) and label.dtype == torch.int64
    assert set(label.unique().tolist()) <= {-1, 0, 3}

    full = CamVid(camvid_root, "train", size=None)[0][1]
    assert full[0, 0] == -1 and full[0, 1] == 0 and full[-1, -1] == 3


def test_camvid_missing_label(camvid_root):
    os.remove(os.path.join(camvid_root, "train_labels", "0001TP_000030_L.png"))
    with pytest.raises(FileNotFoundError):
        CamVid(camvid_root, "train")


def test_camvid_sequence(camvid_root):
    ds = camvid_sequence(camvid_root, size=None, past=5, future=1)
    assert len(ds) == 2
    frames, label = ds[0]  # 0001TP frame 3: clamped at the start of the sequence
    assert frames.shape == (7, 3, 8, 12) and label.shape == (8, 12)
    values = (frames[:, 0, 0, 0] * 255).round().long().tolist()
    assert values == [0, 0, 0, 10, 20, 30, 40]
    frames, _ = ds[1]  # Seq05VD frame 7
    assert (frames[:, 0, 0, 0] * 255).round().long().tolist() == [20, 30, 40, 50, 60, 70, 80]


def test_camvid_sequence_requires_frames(camvid_root):
    with pytest.raises(FileNotFoundError):
        camvid_sequence(camvid_root, seq_dir=os.path.join(camvid_root, "nope"))


# SequenceWindowDataset


def window_dataset(labels, boundary, past=2, future=1, missing_ok=False):
    frames = [Frame(seq, i, f"{seq}_{i:02d}") for seq in ("a", "b") for i in range(5)]
    return SequenceWindowDataset(
        frames,
        [Frame(seq, i, f"label:{seq}_{i:02d}") for seq, i in labels],
        load_frame=lambda path: torch.tensor([ord(path[0]) * 100 + int(path[-2:])]),
        load_label=lambda path: path,
        past=past,
        future=future,
        boundary=boundary,
    )


def window_ids(ds, i):
    frames, label = ds[i]
    return [f"{chr(v // 100)}{v % 100}" for v in frames.flatten().tolist()], label


def test_window_inside_sequence():
    for boundary in ("clamp", "skip", "legacy"):
        ds = window_dataset([("a", 2)], boundary)
        assert window_ids(ds, 0) == (["a0", "a1", "a2", "a3"], "label:a_02")


def test_window_clamp():
    ds = window_dataset([("a", 0), ("b", 4)], "clamp")
    assert window_ids(ds, 0)[0] == ["a0", "a0", "a0", "a1"]
    assert window_ids(ds, 1)[0] == ["b2", "b3", "b4", "b4"]


def test_window_skip():
    ds = window_dataset([("a", 0), ("a", 2), ("b", 4)], "skip")
    assert len(ds) == 1


def test_window_legacy_crosses_sequences():
    ds = window_dataset([("a", 0), ("b", 0), ("b", 4)], "legacy")
    assert len(ds) == 1  # a0 has no past frames, b4 is the last frame of all
    assert window_ids(ds, 0)[0] == ["a3", "a4", "b0", "b1"]


def test_window_missing_labeled_frame_warns():
    with pytest.warns(UserWarning, match="1 labeled frames"):
        ds = window_dataset([("a", 2), ("c", 0)], "clamp")
    assert len(ds) == 1


def test_window_invalid_arguments():
    with pytest.raises(ValueError):
        window_dataset([], "nope")
    with pytest.raises(ValueError):
        window_dataset([], "clamp", past=-1)


# Class weights


def test_median_frequency_weights():
    labels = [torch.tensor([0, 0, 0, 0, 1, 1, 2, -1, 9])]
    weights = median_frequency_weights(labels, 4)
    freq = np.array([4, 2, 1, 0]) / 7
    expected = np.where(freq > 0, np.median(freq) / np.where(freq > 0, freq, 1), 0)
    np.testing.assert_allclose(weights.numpy(), expected, rtol=1e-6)


# Real CamVid (mounted at CAMVID_ROOT inside the container)


@needs_camvid
def test_real_camvid_splits():
    sizes = {split: len(CamVid(CAMVID_ROOT, split)) for split in ("train", "val", "test")}
    assert sizes == {"train": 369, "val": 100, "test": 232}
    image, label = CamVid(CAMVID_ROOT, "test")[0]
    assert image.shape == (3, 360, 480) and label.shape == (360, 480)
    assert label.min() >= -1 and label.max() < 11


@needs_camvid
def test_real_camvid_class_weights_rank_matches_memorized():
    ds = CamVid(CAMVID_ROOT, "train")
    weights = median_frequency_weights((ds[i][1] for i in range(len(ds))), 11)
    memorized = memorized_class_weights("camvid-11")
    assert torch.equal(weights.argsort(), memorized.argsort())
    assert weights[8] == pytest.approx(1.0)  # Car is the median class
