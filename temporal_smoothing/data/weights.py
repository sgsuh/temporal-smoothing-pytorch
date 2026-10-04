"""Class weights for imbalanced segmentation datasets."""

from typing import Iterable

import torch
from torch import Tensor

__all__ = ["median_frequency_weights"]


def median_frequency_weights(labels: Iterable[Tensor], num_classes: int) -> Tensor:
    """Median frequency balancing: ``median(freq) / freq_c`` (0 for absent classes).

    Args:
        labels: Label tensors with class indices; values outside ``[0, num_classes)`` are ignored.
    """
    count = torch.zeros(num_classes, dtype=torch.float64)
    for label in labels:
        label = label.flatten().long()
        label = label[(label >= 0) & (label < num_classes)]
        count += torch.bincount(label, minlength=num_classes).double()
    freq = count / count.sum().clamp(min=1)
    median = freq.quantile(0.5)  # same as np.median in the official code
    weights = torch.where(freq > 0, median / freq.where(freq > 0, torch.ones_like(freq)), 0.0)
    return weights.float()
