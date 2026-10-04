"""Experiment configuration: nested dataclasses loaded from YAML with ``key.path=value`` overrides."""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

__all__ = ["DataConfig", "ModelConfig", "TrainConfig", "EvalConfig", "Config", "load_config", "to_dict"]


@dataclass
class DataConfig:
    root: str = "/data/CamVid"
    name: str = "camvid-11"
    size: Optional[Tuple[int, int]] = (360, 480)
    crop_size: Optional[Tuple[int, int]] = None
    flip: bool = False
    num_workers: int = 4


@dataclass
class ModelConfig:
    arch: str = "unet"  # "unet" or "segnet"
    rate: float = 0.5  # MC dropout rate; 0 gives the deterministic network
    tf_compat: bool = True


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 3
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    class_weights: str = "memorized"  # "memorized", "computed" or "none"
    loss_reduction: str = "mean"  # "mean" over valid pixels or "sum" (official code)
    eval_every: int = 5
    eval_samples: int = 5  # MC samples for validation
    amp: bool = False
    seed: int = 0


@dataclass
class EvalConfig:
    methods: List[str] = field(default_factory=lambda: ["dnn", "mc", "vq"])
    batch_size: int = 3
    mc_samples: int = 30
    past: int = 5
    future: int = 0
    tau: float = 1.25
    temp: float = 1.0
    chunk_size: Optional[int] = None
    cutoffs: Tuple[float, ...] = (0.7, 0.9)
    edge: Optional[float] = None
    boundary: str = "clamp"
    legacy: bool = False
    seed: int = 0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "runs/default"


def _build(cls, values: Dict[str, Any]):
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in values.items():
        if key not in fields:
            raise KeyError(f"unknown config key for {cls.__name__}: {key}")
        default = fields[key].default_factory() if fields[key].default_factory is not dataclasses.MISSING else fields[key].default
        if dataclasses.is_dataclass(default):
            value = _build(type(default), value or {})
        elif isinstance(value, list) and isinstance(default, tuple):
            value = tuple(value)
        kwargs[key] = value
    return cls(**kwargs)


def _set(tree: Dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    for key in parents:
        tree = tree.setdefault(key, {})
    tree[leaf] = value


def load_config(path: Optional[str] = None, overrides: Sequence[str] = ()) -> Config:
    """Load a YAML config and apply overrides such as ``train.epochs=10`` (values parsed as YAML)."""
    tree: Dict[str, Any] = {}
    if path:
        with open(path) as f:
            tree = yaml.safe_load(f) or {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be key=value, got {override}")
        key, value = override.split("=", 1)
        _set(tree, key.strip(), yaml.safe_load(value))
    return _build(Config, tree)


def to_dict(config: Config) -> Dict[str, Any]:
    def convert(value):
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    return convert(dataclasses.asdict(config))
