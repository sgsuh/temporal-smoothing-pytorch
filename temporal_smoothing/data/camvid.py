"""CamVid dataset.

Expected layout (the common 701-image release, e.g. SegNet / Kaggle)::

    root/
      train/  train_labels/  val/  val_labels/  test/  test_labels/
      seq/    # optional: consecutive 30 Hz video frames for temporal smoothing

Images are named ``{sequence}_{frame}.png`` (e.g. ``0001TP_006690.png``,
``Seq05VD_f05100.png``) and labels ``{sequence}_{frame}_L.png`` with RGB color codes.
Frames in ``seq/`` follow the image naming with the same frame numbering.
"""

import os
import re
from functools import partial
from typing import List, Optional, Tuple

from torch import Tensor
from torch.utils.data import Dataset

from .labels import color_map
from .sequence import Frame, SequenceWindowDataset
from .transforms import load_image, load_label, random_crop_flip

__all__ = ["CamVid", "camvid_sequence", "parse_camvid_name", "CAMVID_SIZE"]

CAMVID_SIZE = (360, 480)  # half of the original 720 x 960
SPLITS = ("train", "val", "test")

_NAME = re.compile(r"^([0-9A-Za-z]+)_([0-9A-Za-z]+?)(_L)?\.png$")


def parse_camvid_name(filename: str) -> Tuple[str, int]:
    """``"0001TP_006690.png"`` -> ``("0001TP", 6690)``; ``"Seq05VD_f05100_L.png"`` -> ``("Seq05VD", 5100)``."""
    match = _NAME.match(os.path.basename(filename))
    if match is None:
        raise ValueError(f"not a CamVid file name: {filename}")
    sequence, frame, _ = match.groups()
    digits = re.sub(r"\D", "", frame)
    if not digits:
        raise ValueError(f"no frame number in CamVid file name: {filename}")
    return sequence, int(digits)


def _list_png(directory: str) -> List[str]:
    return sorted(
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".png") and not f.startswith(".")
    )


def _label_path(root: str, split: str, image_path: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(root, f"{split}_labels", f"{stem}_L.png")


class CamVid(Dataset):
    """Labeled CamVid images: ``(image [3, H, W] float in [0, 1], label [H, W] int64)``.

    Args:
        root: Dataset root.
        split: ``"train"``, ``"val"`` or ``"test"``.
        name: Label set, ``"camvid-11"`` (default) or ``"camvid-31"``.
        size: Resize target ``(H, W)``; None keeps 720 x 960.
        crop_size: Random crop size for augmentation (None disables cropping).
        flip: Random horizontal flip for augmentation.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        name: str = "camvid-11",
        size: Optional[Tuple[int, int]] = CAMVID_SIZE,
        crop_size: Optional[Tuple[int, int]] = None,
        flip: bool = False,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split}")
        self.root, self.split, self.name = root, split, name
        self.size, self.crop_size, self.flip = size, crop_size, flip
        self.colors = color_map(name)
        self.images = _list_png(os.path.join(root, split))
        self.labels = [_label_path(root, split, path) for path in self.images]
        missing = [path for path in self.labels if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} labels are missing, e.g. {missing[0]}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor]:
        image = load_image(self.images[i], self.size)
        label = load_label(self.labels[i], self.colors, self.size)
        if self.crop_size is not None or self.flip:
            image, label = random_crop_flip(image, label, self.crop_size, self.flip)
        return image, label


def camvid_sequence(
    root: str,
    seq_dir: Optional[str] = None,
    split: str = "test",
    name: str = "camvid-11",
    size: Optional[Tuple[int, int]] = CAMVID_SIZE,
    past: int = 5,
    future: int = 0,
    boundary: str = "clamp",
) -> SequenceWindowDataset:
    """Frame windows ``(frames [T, 3, H, W], label [H, W])`` around the labeled frames of a split.

    Args:
        seq_dir: Directory of consecutive video frames (default ``root/seq``).
    """
    seq_dir = seq_dir or os.path.join(root, "seq")
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"sequence frames not found: {seq_dir}")
    frames = [Frame(*parse_camvid_name(path), path) for path in _list_png(seq_dir)]
    labels = [
        Frame(*parse_camvid_name(path), path) for path in _list_png(os.path.join(root, f"{split}_labels"))
    ]
    return SequenceWindowDataset(
        frames,
        labels,
        load_frame=partial(load_image, size=size),
        load_label=partial(load_label, colors=color_map(name), size=size),
        past=past,
        future=future,
        boundary=boundary,
    )
