"""Accuracy and uncertainty metrics for dense classification, following the official code.

For a cutoff ``c``, a pixel is *certain* if its confidence (max probability) is above ``c``
and *uncertain* otherwise. The metrics reported for a cutoff, e.g. ``c = 0.9`` -> ``_90``, are:

- ``acc_90``: pixel accuracy of certain pixels, ``p(accurate | confident)``
- ``iou_90``: mean IoU of certain pixels over classes present in their ground truth
- ``unc_90``: ``p(unconfident | inaccurate)``
- ``freq_90``: ``p(confident)``

``nll``, ``acc``, ``iou``, ``class_iou`` and ``ece`` are computed over all valid pixels.
"""

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .calibration import expected_calibration_error

__all__ = ["SegmentationMeter", "edge_mask", "class_iou", "mean_iou", "accuracy"]

# Keras clips probabilities to [eps, 1 - eps] (and renormalizes, which changes NLL by < K * eps).
NLL_EPS = 1e-7


def _div(num: Tensor, den: Tensor) -> Tensor:
    """Elementwise ``num / den`` with zero where ``den == 0``."""
    num, den = num.double(), den.double()
    return torch.where(den != 0, num / torch.where(den != 0, den, torch.ones_like(den)), 0.0)


def class_iou(cm: Tensor) -> Tensor:
    """Per-class IoU of a confusion matrix ``cm[target, prediction]``."""
    tp = cm.diagonal()
    return _div(tp, cm.sum(dim=1) + cm.sum(dim=0) - tp)


def mean_iou(cm: Tensor) -> float:
    """Mean IoU over classes that appear in the ground truth."""
    present = cm.sum(dim=1) > 0
    if not present.any():
        return 0.0
    return class_iou(cm)[present].mean().item()


def accuracy(cm: Tensor) -> float:
    return _div(cm.diagonal().sum(), cm.sum()).item()


def edge_mask(images: Tensor, threshold: float) -> Tensor:
    """Pixels whose Sobel gradient magnitude exceeds ``threshold`` (``tf.image.sobel_edges``).

    Args:
        images: RGB images ``[B, 3, H, W]`` in ``[0, 1]``.

    Returns:
        Boolean mask ``[B, H, W]``.
    """
    rgb_weights = images.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    gray = (images * rgb_weights).sum(dim=1, keepdim=True)
    sobel_y = images.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    kernels = torch.stack([sobel_y, sobel_y.t()]).unsqueeze(1)
    grads = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kernels)
    return grads.pow(2).sum(dim=1).sqrt() > threshold


class SegmentationMeter:
    """Accumulates NLL, confusion matrices and calibration bins over a dataset.

    Args:
        num_classes: Number of classes. Targets outside ``[0, num_classes)`` are ignored.
        cutoffs: Confidence cutoffs for the certain/uncertain metrics.
        n_bins: Number of equal-width confidence bins for ECE and reliability diagrams.
        legacy: Reproduce the official code's inconsistent bin boundaries, where accuracy
            uses ``(lo, hi]`` but mean confidence uses ``[lo, hi)``. By default both use
            ``(lo, hi]`` (the first bin also includes 0).
    """

    def __init__(
        self,
        num_classes: int,
        cutoffs: Sequence[float] = (0.9,),
        n_bins: int = 10,
        legacy: bool = False,
    ) -> None:
        self.num_classes = num_classes
        self.cutoffs = tuple(cutoffs)
        self.n_bins = n_bins
        self.legacy = legacy
        self.reset()

    def reset(self) -> None:
        k, n_cut = self.num_classes, len(self.cutoffs)
        self.nll_sum = torch.zeros((), dtype=torch.float64)
        # [cutoff, uncertain(0) / certain(1), target, prediction]
        self.cm = torch.zeros(n_cut, 2, k, k, dtype=torch.long)
        self.cm_all = torch.zeros(k, k, dtype=torch.long)
        self.bin_count = torch.zeros(self.n_bins, dtype=torch.long)
        self.bin_correct = torch.zeros(self.n_bins, dtype=torch.long)
        self.bin_conf_sum = torch.zeros(self.n_bins, dtype=torch.float64)
        self.bin_conf_count = torch.zeros(self.n_bins, dtype=torch.long)

    def _to(self, device: torch.device) -> None:
        for name in ("nll_sum", "cm", "cm_all", "bin_count", "bin_correct", "bin_conf_sum", "bin_conf_count"):
            setattr(self, name, getattr(self, name).to(device))

    @torch.no_grad()
    def update(self, probs: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> None:
        """Add a batch.

        Args:
            probs: Class probabilities ``[B, num_classes, ...]``.
            target: Class indices ``[B, ...]``; negative values denote void pixels.
            mask: Optional boolean mask ``[B, ...]`` of pixels to evaluate.
        """
        k = self.num_classes
        if probs.shape[1] != k or probs.shape[:1] + probs.shape[2:] != target.shape:
            raise ValueError(
                f"incompatible shapes: probs {tuple(probs.shape)}, target {tuple(target.shape)}"
            )
        self._to(probs.device)

        valid = (target >= 0) & (target < k)
        if mask is not None:
            valid &= mask.to(device=valid.device, dtype=torch.bool)
        probs = probs.movedim(1, -1)[valid]
        target = target[valid].long()
        if target.numel() == 0:
            return

        conf, pred = probs.max(dim=1)
        p_true = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        self.nll_sum += -p_true.double().clamp(NLL_EPS, 1 - NLL_EPS).log().sum()

        pair = target * k + pred
        self.cm_all += torch.bincount(pair, minlength=k * k).view(k, k)
        for i, cutoff in enumerate(self.cutoffs):
            certain = (conf > cutoff).long()
            self.cm[i] += torch.bincount(certain * k * k + pair, minlength=2 * k * k).view(2, k, k)

        edges = torch.linspace(0, 1, self.n_bins + 1, device=conf.device, dtype=conf.dtype)[1:-1]
        correct = (pred == target).double()
        bins = torch.bucketize(conf, edges, right=False)  # (lo, hi]
        self.bin_count += torch.bincount(bins, minlength=self.n_bins)
        self.bin_correct += torch.bincount(bins, weights=correct, minlength=self.n_bins).long()
        if self.legacy:
            keep = conf < 1.0  # [lo, hi) excludes confidence 1
            conf, bins = conf[keep], torch.bucketize(conf[keep], edges, right=True)
        self.bin_conf_sum += torch.bincount(bins, weights=conf.double(), minlength=self.n_bins)
        self.bin_conf_count += torch.bincount(bins, minlength=self.n_bins)

    @property
    def count(self) -> int:
        return int(self.cm_all.sum())

    def bins(self) -> Dict[str, Tensor]:
        """Per-bin pixel count, accuracy and mean confidence (CPU, float64)."""
        return {
            "edges": torch.linspace(0, 1, self.n_bins + 1, dtype=torch.float64),
            "count": self.bin_count.cpu().double(),
            "acc": _div(self.bin_correct, self.bin_count).cpu(),
            "conf": _div(self.bin_conf_sum, self.bin_conf_count).cpu(),
        }

    def compute(self) -> Dict[str, object]:
        cm_all = self.cm_all.cpu()
        bins = self.bins()
        results: Dict[str, object] = {
            "nll": (self.nll_sum.cpu() / max(self.count, 1)).item(),
            "acc": accuracy(cm_all),
            "iou": mean_iou(cm_all),
            "class_iou": class_iou(cm_all).tolist(),
            "ece": expected_calibration_error(bins["count"], bins["acc"], bins["conf"]),
        }
        for cutoff, cm in zip(self.cutoffs, self.cm.cpu()):
            uncertain, certain = cm
            inaccurate_certain = certain.sum() - certain.diagonal().sum()
            inaccurate_uncertain = uncertain.sum() - uncertain.diagonal().sum()
            inaccurate = inaccurate_certain + inaccurate_uncertain
            suffix = f"_{round(cutoff * 100)}"
            results["acc" + suffix] = accuracy(certain)
            results["iou" + suffix] = mean_iou(certain)
            results["unc" + suffix] = _div(inaccurate_uncertain, inaccurate).item() if inaccurate else float("nan")
            results["freq" + suffix] = _div(certain.sum(), cm.sum()).item()
        return results
