"""Training loop for segmentation models, following the official training setup."""

import json
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..data import CamVid, median_frequency_weights, memorized_class_weights, num_classes
from ..metrics import SegmentationMeter
from ..nn import SegNet, UNet
from ..predictors import predict_mc
from .config import Config, ModelConfig, to_dict

__all__ = ["build_model", "segmentation_loss", "train_one_epoch", "fit", "seed_everything", "save_checkpoint", "load_checkpoint"]

ARCHS = {"unet": UNet, "segnet": SegNet}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(config: ModelConfig, n_classes: int) -> nn.Module:
    try:
        arch = ARCHS[config.arch.lower()]
    except KeyError:
        raise ValueError(f"unknown architecture: {config.arch}; expected one of {sorted(ARCHS)}") from None
    return arch(n_classes, rate=config.rate, tf_compat=config.tf_compat)


def segmentation_loss(
    logits: Tensor,
    target: Tensor,
    class_weights: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Dict[str, Tensor]:
    """Class-weighted cross entropy over valid pixels (``target >= 0``).

    As in the official code, each pixel's NLL is multiplied by the weight of its true class.
    ``"mean"`` divides by the number of valid pixels (not by the sum of weights as PyTorch's
    weighted ``cross_entropy`` does); ``"sum"`` matches the official gradient exactly.

    Returns:
        ``{"loss": weighted loss to optimize, "nll": unweighted mean NLL (for logging)}``.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
    nll = F.cross_entropy(logits, target, ignore_index=-1, reduction="none")
    valid = target >= 0
    count = valid.sum().clamp(min=1)
    weighted = nll if class_weights is None else nll * class_weights.to(nll)[target.clamp(min=0)]
    weighted = weighted[valid].sum()
    return {
        "loss": weighted / count if reduction == "mean" else weighted,
        "nll": nll[valid].sum().detach() / count,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: Optional[Tensor] = None,
    loss_reduction: str = "mean",
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "nll": 0.0}
    steps = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=scaler is not None):
            logits = model(images)
        losses = segmentation_loss(logits.float(), labels, class_weights, loss_reduction)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"non-finite loss: {losses['loss'].item()}")
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            optimizer.step()
        totals["loss"] += losses["loss"].item()
        totals["nll"] += losses["nll"].item()
        steps += 1
    return {k: v / max(steps, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int, n_samples: int) -> Dict[str, float]:
    model.eval()
    meter = SegmentationMeter(n_classes, cutoffs=(0.9,))
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        meter.update(predict_mc(model, images, n_samples), labels)
    return {k: v for k, v in meter.compute().items() if not isinstance(v, list)}


def save_checkpoint(path: str, model: nn.Module, optimizer=None, epoch: int = 0, config: Optional[Config] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {"model": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if config is not None:
        state["config"] = to_dict(config)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, map_location="cpu") -> dict:
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state


def _class_weights(config: Config, dataset: CamVid, n_classes: int) -> Optional[Tensor]:
    mode = config.train.class_weights
    if mode == "none":
        return None
    if mode == "memorized":
        return memorized_class_weights(config.data.name)
    if mode == "computed":
        return median_frequency_weights((dataset[i][1] for i in range(len(dataset))), n_classes)
    raise ValueError(f"unknown class_weights: {mode}")


def fit(config: Config, resume: bool = False, device: Optional[torch.device] = None, log=print) -> nn.Module:
    """Train a model on CamVid and write checkpoints / metrics to ``config.output_dir``."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config.train.seed)
    out = config.output_dir
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(to_dict(config), f, indent=2)

    data = config.data
    n_classes = num_classes(data.name)
    train_set = CamVid(data.root, "train", data.name, data.size, data.crop_size, data.flip)
    val_set = CamVid(data.root, "val", data.name, data.size)
    loader_kwargs = dict(num_workers=data.num_workers, pin_memory=device.type == "cuda",
                         persistent_workers=data.num_workers > 0)
    train_loader = DataLoader(train_set, config.train.batch_size, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_set, 1, shuffle=False, **loader_kwargs)

    model = build_model(config.model, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr, betas=tuple(config.train.betas))
    scaler = torch.amp.GradScaler(device.type) if config.train.amp else None
    class_weights = _class_weights(config, train_set, n_classes)
    class_weights = class_weights.to(device) if class_weights is not None else None

    start_epoch = 0
    last_path = os.path.join(out, "last.pt")
    if resume and os.path.exists(last_path):
        start_epoch = load_checkpoint(last_path, model, optimizer, map_location=device)["epoch"]
        log(f"resumed from {last_path} at epoch {start_epoch}")

    with open(os.path.join(out, "metrics.jsonl"), "a") as metrics_file:
        for epoch in range(start_epoch, config.train.epochs):
            tic = time.time()
            stats = train_one_epoch(model, train_loader, optimizer, device, class_weights,
                                    config.train.loss_reduction, scaler)
            record = {"epoch": epoch + 1, "time": time.time() - tic, **{f"train_{k}": v for k, v in stats.items()}}
            if config.train.eval_every and (epoch + 1) % config.train.eval_every == 0:
                val = validate(model, val_loader, device, n_classes, config.train.eval_samples)
                record.update({f"val_{k}": v for k, v in val.items()})
            save_checkpoint(last_path, model, optimizer, epoch + 1, config)
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))

    save_checkpoint(os.path.join(out, "model.pt"), model, epoch=config.train.epochs, config=config)
    return model
