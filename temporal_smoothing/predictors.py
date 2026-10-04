"""Predictive distributions of DNN, BNN, VQ-DNN/VQ-BNN and ensembles for classification.

Every predictor maps inputs to class probabilities ``[B, num_classes, ...]``. Models return
logits with classes along dim 1. Sequence inputs are ``[B, T, C, ...]`` ordered from the
oldest to the newest frame, with ``T = past + future + 1`` and the current frame at index
``past``.

Models should be in eval mode: MC dropout layers stay stochastic, while batch norm uses
running statistics. Only then is batching frames or samples equivalent to separate forward
passes, as in the official implementation.
"""

import warnings
from typing import Callable, List, Optional, Sequence

import torch
from torch import Tensor

from .smoothing import exp_decay_weights, smooth_categorical

__all__ = [
    "predict_dnn",
    "predict_temp_scaling",
    "predict_mc",
    "predict_vq",
    "predict_ensemble",
    "predict_ensemble_smoothing",
    "sample_model_indices",
]

Model = Callable[[Tensor], Tensor]


def _check_eval(model: Model) -> None:
    if getattr(model, "training", False):
        warnings.warn(
            "model is in training mode; call model.eval() before prediction "
            "(MC dropout stays active in eval mode)",
            stacklevel=4,  # _check_eval <- predictor <- no_grad wrapper <- caller
        )


def _forward(model: Model, x: Tensor, chunk_size: Optional[int] = None, temp: float = 1.0) -> Tensor:
    """Softmax of ``model(x) / temp``, optionally evaluated in chunks along the batch."""
    chunks = x.split(chunk_size) if chunk_size else (x,)
    return torch.cat([(model(chunk) / temp).softmax(dim=1) for chunk in chunks])


def _check_window(xs: Tensor, past: int, future: int) -> None:
    if xs.dim() < 3 or xs.shape[1] != past + future + 1:
        raise ValueError(
            f"expected inputs of shape [B, T={past + future + 1}, ...] for past={past}, "
            f"future={future}, got {tuple(xs.shape)}"
        )


@torch.no_grad()
def predict_dnn(model: Model, x: Tensor) -> Tensor:
    """Single forward pass: ``p(y | x_0, w_0)``."""
    _check_eval(model)
    return _forward(model, x)


@torch.no_grad()
def predict_temp_scaling(model: Model, x: Tensor, temp: float = 1.0) -> Tensor:
    """Single forward pass with temperature-scaled logits."""
    _check_eval(model)
    return _forward(model, x, temp=temp)


@torch.no_grad()
def predict_mc(model: Model, x: Tensor, n_samples: int = 30) -> Tensor:
    """BNN predictive distribution by MC estimation: ``1/N sum_i p(y | x_0, w_i)``."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    _check_eval(model)
    probs = _forward(model, x)
    for _ in range(n_samples - 1):
        probs += _forward(model, x)
    return probs / n_samples


@torch.no_grad()
def predict_vq(
    model: Model,
    xs: Tensor,
    past: int = 5,
    future: int = 0,
    tau: float = 1.25,
    chunk_size: Optional[int] = None,
) -> Tensor:
    """VQ-BNN (or VQ-DNN) predictive distribution by temporal smoothing.

    Each frame of the window gets one forward pass (one weight sample for a BNN) and the
    resulting probabilities are combined with exponentially decaying importances.

    Args:
        xs: Frame windows ``[B, T, C, ...]``.
        chunk_size: Maximum number of frames per forward pass (all ``B * T`` at once if None).
    """
    _check_window(xs, past, future)
    _check_eval(model)
    batch_size, num_frames = xs.shape[:2]
    probs = _forward(model, xs.flatten(0, 1), chunk_size)
    probs = probs.unflatten(0, (batch_size, num_frames))
    weights = exp_decay_weights(past, future, tau, device=probs.device, dtype=probs.dtype)
    return smooth_categorical(probs, weights, dim=1)


@torch.no_grad()
def predict_ensemble(models: Sequence[Model], x: Tensor) -> Tensor:
    """Deep ensemble: average of the members' probabilities."""
    if not models:
        raise ValueError("models must not be empty")
    for model in models:
        _check_eval(model)
    return torch.stack([_forward(model, x) for model in models]).mean(dim=0)


def sample_model_indices(
    num_models: int,
    num_frames: int,
    generator: Optional[torch.Generator] = None,
    legacy: bool = False,
) -> List[int]:
    """Draw one ensemble member per frame, uniformly.

    With ``legacy=True`` the official code's sampling ``models[randint(0, n) - 1]`` is
    reproduced, under which the last member is drawn twice as often as the others.
    """
    if legacy:
        indices = torch.randint(0, num_models + 1, (num_frames,), generator=generator) - 1
        indices = indices % num_models
    else:
        indices = torch.randint(0, num_models, (num_frames,), generator=generator)
    return indices.tolist()


@torch.no_grad()
def predict_ensemble_smoothing(
    models: Sequence[Model],
    xs: Tensor,
    past: int = 5,
    future: int = 0,
    tau: float = 1.25,
    generator: Optional[torch.Generator] = None,
    legacy: bool = False,
) -> Tensor:
    """Temporal smoothing of an ensemble: each frame is predicted by one random member.

    As in the official code, the member is drawn per frame and shared across the batch.
    """
    if not models:
        raise ValueError("models must not be empty")
    _check_window(xs, past, future)
    for model in models:
        _check_eval(model)
    indices = sample_model_indices(len(models), xs.shape[1], generator, legacy)
    probs = torch.stack([_forward(models[i], xs[:, t]) for t, i in enumerate(indices)], dim=1)
    weights = exp_decay_weights(past, future, tau, device=probs.device, dtype=probs.dtype)
    return smooth_categorical(probs, weights, dim=1)
