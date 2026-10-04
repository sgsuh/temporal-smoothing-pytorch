"""Evaluation of DNN, BNN, VQ-DNN/VQ-BNN and ensemble predictions on frame windows.

All methods are evaluated on the same frame windows ``[B, T, 3, H, W]`` around the labeled
test frames: single-frame methods use the current frame ``frames[:, past]`` (as the official
notebook does), temporal smoothing uses the whole window.
"""

import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..metrics import SegmentationMeter, edge_mask
from ..predictors import (
    predict_dnn,
    predict_ensemble,
    predict_ensemble_smoothing,
    predict_mc,
    predict_temp_scaling,
    predict_vq,
)
from .config import EvalConfig

__all__ = ["METHODS", "make_predictor", "evaluate"]

Predictor = Callable[[Tensor], Tensor]

SINGLE_MODEL_METHODS = ("dnn", "temp", "mc", "vq")
ENSEMBLE_METHODS = ("ensemble", "ensemble_vq")
METHODS = SINGLE_MODEL_METHODS + ENSEMBLE_METHODS


def make_predictor(method: str, models: Sequence[nn.Module], config: EvalConfig) -> Predictor:
    """Map frame windows ``[B, T, 3, H, W]`` to class probabilities for the current frame."""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}; expected one of {METHODS}")
    if method in SINGLE_MODEL_METHODS and len(models) != 1:
        raise ValueError(f"method {method} needs exactly one model, got {len(models)}")
    model, past = models[0], config.past
    if method == "dnn":
        return lambda xs: predict_dnn(model, xs[:, past])
    if method == "temp":
        return lambda xs: predict_temp_scaling(model, xs[:, past], config.temp)
    if method == "mc":
        return lambda xs: predict_mc(model, xs[:, past], config.mc_samples)
    if method == "vq":
        return lambda xs: predict_vq(model, xs, config.past, config.future, config.tau, config.chunk_size)
    if method == "ensemble":
        return lambda xs: predict_ensemble(models, xs[:, past])
    generator = torch.Generator().manual_seed(config.seed)
    return lambda xs: predict_ensemble_smoothing(
        models, xs, config.past, config.future, config.tau, generator=generator, legacy=config.legacy
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(
    predictor: Predictor,
    loader: DataLoader,
    num_classes: int,
    config: EvalConfig,
    device: torch.device,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Run ``predictor`` over frame windows and return metrics, calibration bins and timing.

    Timing covers the predictor only (inputs already on the device) and is reported in ms per
    batch and frames (labeled images) per second.
    """
    meter = SegmentationMeter(num_classes, cutoffs=config.cutoffs, legacy=config.legacy)
    times: List[float] = []
    images = 0
    for step, (frames, labels) in enumerate(loader):
        frames, labels = frames.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        _sync(device)
        tic = time.perf_counter()
        probs = predictor(frames)
        _sync(device)
        times.append(time.perf_counter() - tic)
        images += frames.shape[0]

        mask = edge_mask(frames[:, config.past], config.edge) if config.edge is not None else None
        meter.update(probs, labels, mask)
        if log is not None and (step + 1) % 20 == 0:
            log(f"  {step + 1}/{len(loader)} batches")

    results = meter.compute()
    # Skip the first batch (CUDA warm-up) when timing if possible.
    timed = times[1:] if len(times) > 1 else times
    results.update(
        {
            "time_ms": float(np.mean(timed) * 1e3),
            "time_ms_std": float(np.std(timed) * 1e3),
            "throughput": float(images / sum(times)) if sum(times) > 0 else 0.0,
            "bins": {k: v.tolist() for k, v in meter.bins().items()},
        }
    )
    return results
