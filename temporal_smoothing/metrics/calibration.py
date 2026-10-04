"""Expected calibration error and calibration plots."""

from typing import Mapping, Optional, Sequence, Union

import torch
from torch import Tensor

__all__ = ["expected_calibration_error", "plot_calibration", "confidence_histogram", "reliability_diagram"]

ArrayLike = Union[Tensor, Sequence[float]]


def expected_calibration_error(count: ArrayLike, acc: ArrayLike, conf: ArrayLike) -> float:
    """``sum_b (n_b / n) |acc_b - conf_b|`` over confidence bins."""
    count, acc, conf = (torch.as_tensor(v, dtype=torch.float64) for v in (count, acc, conf))
    total = count.sum()
    if total == 0:
        return 0.0
    return ((count / total) * (acc - conf).abs()).sum().item()


def confidence_histogram(ax, edges: ArrayLike, count: ArrayLike, color: str = "tab:green") -> None:
    edges, count = torch.as_tensor(edges).double(), torch.as_tensor(count).double()
    freq = (count / count.sum().clamp(min=1)).numpy()
    edges = edges.numpy()
    ax.bar(edges[:-1] * 100, freq * 100, width=(edges[1:] - edges[:-1]) * 100, align="edge",
           color=color, edgecolor="black", alpha=0.8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Frequency (%)")


def reliability_diagram(
    ax,
    edges: ArrayLike,
    accs: Union[ArrayLike, Sequence[ArrayLike]],
    colors: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    mode: str = "bar",
) -> None:
    """Accuracy per confidence bin; several series can be overlaid with ``mode="line"``."""
    edges = torch.as_tensor(edges).double()
    accs = torch.as_tensor(accs, dtype=torch.float64) if not isinstance(accs, Tensor) else accs.double()
    accs = accs.unsqueeze(0) if accs.dim() == 1 else accs
    colors = list(colors or ["tab:red"]) + [None] * len(accs)
    labels = list(labels or []) + [None] * len(accs)
    edges, accs = edges.numpy(), accs.numpy()
    centers = (edges[:-1] + edges[1:]) / 2

    ax.plot([0, 100], [0, 100], linestyle=(0, (1, 1)), color="black")
    for acc, color, label in zip(accs, colors, labels):
        if mode == "bar":
            ax.bar(edges[:-1] * 100, acc * 100, width=(edges[1:] - edges[:-1]) * 100, align="edge",
                   color=color, edgecolor="black", alpha=0.8, label=label)
        elif mode == "line":
            ax.plot(centers * 100, acc * 100, color=color, marker="o", alpha=0.8, label=label)
        else:
            raise ValueError(f"invalid mode: {mode}")
    if any(labels):
        ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Accuracy (%)")


def plot_calibration(bins: Mapping[str, ArrayLike]):
    """Confidence histogram and reliability diagram from ``SegmentationMeter.bins()``."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 4))
    ax_hist, ax_rel = fig.subplots(1, 2)
    confidence_histogram(ax_hist, bins["edges"], bins["count"])
    reliability_diagram(ax_rel, bins["edges"], bins["acc"])
    fig.tight_layout()
    return fig
