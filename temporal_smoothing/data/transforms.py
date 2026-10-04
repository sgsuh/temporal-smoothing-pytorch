"""Image / label loading and joint augmentation.

Images and color labels are resized with nearest neighbor sampling at pixel centers
(``mode="nearest-exact"``), which matches ``tf.image.resize(..., NEAREST_NEIGHBOR)`` in TF 2.
Images are scaled to ``[0, 1]`` without normalization, as in the official code.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

__all__ = ["read_rgb", "resize_nearest", "load_image", "load_label", "colors_to_index", "random_crop_flip"]

Size = Tuple[int, int]


def read_rgb(path: str) -> Tensor:
    """Read an image file as a ``uint8`` tensor ``[3, H, W]``."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"))
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def resize_nearest(x: Tensor, size: Optional[Size]) -> Tensor:
    """Nearest-neighbor resize of a ``[C, H, W]`` tensor (no-op if ``size`` is None or equal)."""
    if size is None or tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x.unsqueeze(0), size=tuple(size), mode="nearest-exact").squeeze(0)


def colors_to_index(rgb: Tensor, colors: Dict[Tuple[int, int, int], int]) -> Tensor:
    """Map a ``uint8`` color label ``[3, H, W]`` to class indices ``[H, W]`` (unknown -> -1)."""
    codes = (rgb[0].long() << 16) | (rgb[1].long() << 8) | rgb[2].long()
    keys = torch.tensor([(r << 16) | (g << 8) | b for r, g, b in colors], dtype=torch.long)
    values = torch.tensor(list(colors.values()), dtype=torch.long)
    keys, order = keys.sort()
    values = values[order]
    pos = torch.searchsorted(keys, codes).clamp(max=len(keys) - 1)
    return torch.where(keys[pos] == codes, values[pos], torch.full_like(codes, -1))


def load_image(path: str, size: Optional[Size] = None) -> Tensor:
    """Load an RGB image as ``float32`` ``[3, H, W]`` in ``[0, 1]``."""
    return resize_nearest(read_rgb(path), size).float() / 255.0


def load_label(path: str, colors: Dict[Tuple[int, int, int], int], size: Optional[Size] = None) -> Tensor:
    """Load a color-coded label as class indices ``[H, W]`` (``int64``, void = -1)."""
    return colors_to_index(resize_nearest(read_rgb(path), size), colors)


def random_crop_flip(
    image: Tensor,
    label: Tensor,
    crop_size: Optional[Size] = None,
    flip: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    """Apply the same random crop and horizontal flip to an image and its label."""
    if crop_size is not None:
        h, w = image.shape[-2:]
        ch, cw = crop_size
        if ch > h or cw > w:
            raise ValueError(f"crop size {tuple(crop_size)} exceeds image size {(h, w)}")
        top = int(torch.randint(0, h - ch + 1, (1,), generator=generator))
        left = int(torch.randint(0, w - cw + 1, (1,), generator=generator))
        image = image[..., top : top + ch, left : left + cw]
        label = label[..., top : top + ch, left : left + cw]
    if flip and bool(torch.rand((), generator=generator) < 0.5):
        image, label = image.flip(-1), label.flip(-1)
    return image, label
