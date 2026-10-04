"""PyTorch implementation of VQ-BNN temporal smoothing for data streams."""

from .predictors import (
    predict_dnn,
    predict_ensemble,
    predict_ensemble_smoothing,
    predict_mc,
    predict_temp_scaling,
    predict_vq,
)
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
    "predict_dnn",
    "predict_ensemble",
    "predict_ensemble_smoothing",
    "predict_mc",
    "predict_temp_scaling",
    "predict_vq",
    "smooth_categorical",
    "smooth_gaussian",
]
