"""PyTorch implementation of VQ-BNN temporal smoothing for data streams."""

from .smoothing import (
    EMASmoother,
    StreamSmoother,
    exp_decay_weights,
    smooth_categorical,
    smooth_gaussian,
)

__version__ = "0.1.0"

__all__ = [
    "EMASmoother",
    "StreamSmoother",
    "exp_decay_weights",
    "smooth_categorical",
    "smooth_gaussian",
]
