"""Temporal smoothing of predictions for data streams (VQ-BNN inference).

The VQ-BNN predictive distribution for the current frame ``x_0`` is a weighted mixture of
the predictions for recent (and optionally future) frames::

    p(y | x_0, D) ~= sum_{t=J}^{-K} pi_t * p(y | x_t, w_t),
    pi_t = exp(-|t| / tau) / sum_s exp(-|s| / tau)

where ``K`` is the number of past frames, ``J`` the number of future frames and ``tau`` the
decay timescale. The official TensorFlow code parameterizes the decay as ``l = 1 / tau``.

For classification the mixture components are softmax probabilities (not logits). For
regression with Gaussian outputs the mixture is collapsed by moment matching.
"""

from collections import deque
from typing import Deque, Optional, Tuple, Union

import torch
from torch import Tensor

__all__ = [
    "exp_decay_weights",
    "smooth_categorical",
    "smooth_gaussian",
    "StreamSmoother",
    "EMASmoother",
]


def exp_decay_weights(
    past: int,
    future: int = 0,
    tau: float = 1.25,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Normalized exponentially decaying importances of frames in a window.

    Args:
        past: Number of past frames ``K``.
        future: Number of future frames ``J``.
        tau: Decay timescale. Larger values weight distant frames more.

    Returns:
        Tensor of shape ``[past + future + 1]`` ordered from the oldest to the newest frame.
        Index ``past`` is the current frame. The weights sum to one.
    """
    if past < 0 or future < 0:
        raise ValueError(f"past and future must be non-negative, got past={past}, future={future}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")

    offsets = torch.arange(-past, future + 1, device=device, dtype=torch.float64)
    weights = torch.exp(-offsets.abs() / tau)
    weights = weights / weights.sum()
    return weights.to(dtype=dtype or torch.get_default_dtype())


def _check_weights(x: Tensor, weights: Tensor, dim: int) -> Tensor:
    if weights.dim() != 1:
        raise ValueError(f"weights must be 1-D, got shape {tuple(weights.shape)}")
    if x.shape[dim] != weights.numel():
        raise ValueError(
            f"size of dim {dim} ({x.shape[dim]}) does not match number of weights ({weights.numel()})"
        )
    shape = [1] * x.dim()
    shape[dim] = -1
    return weights.to(device=x.device, dtype=x.dtype).view(shape)


def smooth_categorical(probs: Tensor, weights: Tensor, dim: int = 1) -> Tensor:
    """Weighted mixture of categorical distributions along the time dimension.

    Args:
        probs: Probabilities (e.g. softmax outputs) with time along ``dim``,
            e.g. ``[B, T, num_classes, H, W]``.
        weights: Frame importances of shape ``[T]``, e.g. from :func:`exp_decay_weights`.
        dim: Time dimension of ``probs``.

    Returns:
        ``probs`` reduced over ``dim``.
    """
    w = _check_weights(probs, weights, dim)
    return (probs * w).sum(dim=dim)


def smooth_gaussian(
    mean: Tensor,
    var: Union[Tensor, float],
    weights: Tensor,
    dim: int = 1,
) -> Tuple[Tensor, Tensor]:
    """Moment-matched Gaussian of a weighted mixture of Gaussians along the time dimension.

    The mixture mean is ``E[mu]`` and its variance is ``E[sigma^2] + Var[mu]``.

    Args:
        mean: Component means with time along ``dim``.
        var: Component variances broadcastable to ``mean``, or a constant variance shared
            by all components (e.g. a fixed aleatoric variance).
        weights: Frame importances of shape ``[T]``.
        dim: Time dimension of ``mean``.

    Returns:
        ``(mean, var)`` of the mixture, both reduced over ``dim``.
    """
    w = _check_weights(mean, weights, dim)
    mixture_mean = (mean * w).sum(dim=dim)
    spread = ((mean - mixture_mean.unsqueeze(dim)) ** 2 * w).sum(dim=dim)
    if isinstance(var, Tensor):
        expected_var = (var.expand_as(mean) * w).sum(dim=dim)
    else:
        expected_var = var
    return mixture_mean, spread + expected_var


class StreamSmoother:
    """Causal temporal smoothing over a fixed window of cached predictions.

    Each new frame needs a single forward pass: its prediction is pushed into a ring buffer
    of the latest ``past + 1`` predictions, which are combined with exponentially decaying
    weights. Until the buffer is full, weights are renormalized over the cached frames.

    Predictions may carry a leading batch dimension, in which case every batch element is
    treated as an independent stream.

    Args:
        past: Number of past frames ``K`` kept in addition to the current one.
        tau: Decay timescale.
    """

    def __init__(self, past: int = 5, tau: float = 1.25) -> None:
        exp_decay_weights(past, 0, tau)  # validate arguments
        self.past = past
        self.tau = tau
        self._buffer: Deque[Tensor] = deque(maxlen=past + 1)

    def __len__(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()

    @torch.no_grad()
    def update(self, pred: Tensor) -> Tensor:
        """Push the prediction for the newest frame and return the smoothed prediction."""
        if self._buffer and self._buffer[-1].shape != pred.shape:
            raise ValueError(
                f"prediction shape {tuple(pred.shape)} differs from buffered shape "
                f"{tuple(self._buffer[-1].shape)}; call reset() when the stream changes"
            )
        self._buffer.append(pred)
        weights = exp_decay_weights(len(self._buffer) - 1, 0, self.tau)
        return smooth_categorical(torch.stack(tuple(self._buffer), dim=0), weights, dim=0)


class EMASmoother:
    """Recursive exponential moving average of predictions.

    ``q_0 = alpha * p_0 + (1 - alpha) * q_{-1}``, which is temporal smoothing with an infinite
    window when ``alpha = 1 - exp(-1 / tau)``. The average is bias-corrected, i.e. normalized
    by the total weight of the frames seen so far, so that it matches :class:`StreamSmoother`
    with an unbounded window.

    Exactly one of ``alpha`` and ``tau`` must be given.
    """

    def __init__(self, alpha: Optional[float] = None, tau: Optional[float] = None) -> None:
        if (alpha is None) == (tau is None):
            raise ValueError("exactly one of alpha and tau must be given")
        if tau is not None:
            if tau <= 0:
                raise ValueError(f"tau must be positive, got {tau}")
            alpha = 1.0 - float(torch.exp(torch.tensor(-1.0 / tau, dtype=torch.float64)))
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._state: Optional[Tensor] = None
        self._norm = 0.0

    def reset(self) -> None:
        self._state = None
        self._norm = 0.0

    @torch.no_grad()
    def update(self, pred: Tensor) -> Tensor:
        """Push the prediction for the newest frame and return the smoothed prediction."""
        if self._state is None:
            self._state = self.alpha * pred
        else:
            if self._state.shape != pred.shape:
                raise ValueError(
                    f"prediction shape {tuple(pred.shape)} differs from state shape "
                    f"{tuple(self._state.shape)}; call reset() when the stream changes"
                )
            self._state = self.alpha * pred + (1.0 - self.alpha) * self._state
        self._norm = self.alpha + (1.0 - self.alpha) * self._norm
        return self._state / self._norm
