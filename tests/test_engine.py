import json
import os
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from temporal_smoothing import predict_dnn, predict_vq
from temporal_smoothing.engine import (
    Config,
    EvalConfig,
    ModelConfig,
    build_model,
    evaluate,
    load_checkpoint,
    load_config,
    make_predictor,
    save_checkpoint,
    segmentation_loss,
    to_dict,
    train_one_epoch,
)
from temporal_smoothing.nn import SegNet, UNet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = 3


def toy_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, K, 1))


# Config


def test_load_config(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("output_dir: runs/x\nmodel: {arch: segnet}\ntrain: {betas: [0.5, 0.9]}\n")
    config = load_config(str(path), ["train.epochs=7", "eval.methods=[dnn]", "data.size=null"])
    assert config.output_dir == "runs/x"
    assert config.model.arch == "segnet" and config.model.rate == 0.5
    assert config.train.betas == (0.5, 0.9) and config.train.epochs == 7
    assert config.eval.methods == ["dnn"] and config.data.size is None
    assert to_dict(config)["train"]["betas"] == [0.5, 0.9]


def test_load_config_rejects_unknown_keys():
    with pytest.raises(KeyError):
        load_config(None, ["train.epoch=1"])
    with pytest.raises(ValueError):
        load_config(None, ["train.epochs"])


@pytest.mark.parametrize("name", ["camvid_unet_bnn", "camvid_unet_dnn", "camvid_segnet_bnn", "camvid_segnet_dnn"])
def test_shipped_configs_load(name):
    config = load_config(os.path.join(REPO, "configs", f"{name}.yaml"))
    assert config.model.arch in name and (config.model.rate > 0) == ("bnn" in name)


def test_build_model():
    assert isinstance(build_model(ModelConfig("unet", 0.5), 11), UNet)
    assert isinstance(build_model(ModelConfig("segnet", 0.0), 11), SegNet)
    with pytest.raises(ValueError):
        build_model(ModelConfig("resnet"), 11)


# Loss


def test_segmentation_loss():
    torch.manual_seed(0)
    logits = torch.randn(2, K, 4, 5)
    target = torch.randint(-1, K, (2, 4, 5))
    weights = torch.tensor([0.5, 1.0, 2.0])
    valid = target >= 0

    nll = F.cross_entropy(logits, target, ignore_index=-1, reduction="none")
    weighted = (nll * weights[target.clamp(min=0)])[valid]
    mean = segmentation_loss(logits, target, weights, "mean")
    total = segmentation_loss(logits, target, weights, "sum")
    torch.testing.assert_close(mean["loss"], weighted.sum() / valid.sum())
    torch.testing.assert_close(total["loss"], weighted.sum())
    torch.testing.assert_close(mean["nll"], nll[valid].mean())
    # Not PyTorch's weighted mean, which divides by the sum of the weights.
    assert not torch.isclose(mean["loss"], F.cross_entropy(logits, target, weights, ignore_index=-1))

    unweighted = segmentation_loss(logits, target, None)
    torch.testing.assert_close(unweighted["loss"], F.cross_entropy(logits, target, ignore_index=-1))
    with pytest.raises(ValueError):
        segmentation_loss(logits, target, reduction="none")


def test_train_one_epoch_reduces_loss():
    torch.manual_seed(0)
    images = torch.rand(6, 3, 8, 8)
    labels = images.argmax(dim=1)  # learnable by the toy model
    loader = DataLoader(TensorDataset(images, labels), batch_size=3)
    model = toy_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = train_one_epoch(model, loader, optimizer, torch.device("cpu"))
    for _ in range(30):
        last = train_one_epoch(model, loader, optimizer, torch.device("cpu"))
    assert last["loss"] < first["loss"] * 0.5
    assert model.training


def test_checkpoint_roundtrip(tmp_path):
    model, path = toy_model(), str(tmp_path / "ckpt" / "m.pt")
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(path, model, optimizer, epoch=3, config=Config())
    other = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, K, 1))
    state = load_checkpoint(path, other, torch.optim.Adam(other.parameters()))
    assert state["epoch"] == 3 and state["config"]["model"]["arch"] == "unet"
    for a, b in zip(model.parameters(), other.parameters()):
        assert torch.equal(a, b)


# Evaluation


def window_loader(past=2, future=1):
    torch.manual_seed(0)
    frames = torch.rand(4, past + future + 1, 3, 6, 7)
    labels = torch.randint(-1, K, (4, 6, 7))
    return DataLoader(TensorDataset(frames, labels), batch_size=2)


def test_make_predictor_uses_current_frame():
    model = toy_model().eval()
    config = EvalConfig(past=2, future=1)
    frames = next(iter(window_loader()))[0]
    torch.testing.assert_close(make_predictor("dnn", [model], config)(frames), predict_dnn(model, frames[:, 2]))
    torch.testing.assert_close(
        make_predictor("vq", [model], config)(frames), predict_vq(model, frames, past=2, future=1)
    )
    torch.testing.assert_close(
        make_predictor("ensemble", [model, model], config)(frames), predict_dnn(model, frames[:, 2])
    )
    with pytest.raises(ValueError):
        make_predictor("mc", [model, model], config)
    with pytest.raises(ValueError):
        make_predictor("nope", [model], config)


@pytest.mark.parametrize("method", ["dnn", "temp", "mc", "vq", "ensemble", "ensemble_vq"])
def test_evaluate(method):
    config = EvalConfig(past=2, future=1, mc_samples=2, cutoffs=(0.5,), edge=0.1)
    models = [toy_model().eval(), toy_model().eval()]
    predictor = make_predictor(method, models if method.startswith("ensemble") else models[:1], config)
    results = evaluate(predictor, window_loader(), K, config, torch.device("cpu"))
    for key in ("nll", "acc", "iou", "ece", "acc_50", "unc_50", "freq_50", "time_ms", "throughput", "bins"):
        assert key in results
    assert 0.0 <= results["acc"] <= 1.0
    json.dumps(results)  # serializable


# End to end on a tiny CamVid-like dataset


def run_script(*args):
    env = dict(os.environ, PYTHONPATH=REPO)
    proc = subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_train_and_eval_scripts(camvid_root, tmp_path):
    out = str(tmp_path / "run")
    common = [
        "--config", "configs/camvid_unet_bnn.yaml", "--set",
        f"data.root={camvid_root}", "data.size=[16, 24]", "data.num_workers=0", f"output_dir={out}",
        "train.epochs=2", "train.eval_every=1", "train.eval_samples=2",
        "eval.mc_samples=2", "eval.past=2", "eval.future=1", "eval.batch_size=2",
    ]
    stdout = run_script("scripts/train_seg.py", *common)
    assert "epoch=2" in stdout and "val_nll" in stdout
    assert {"model.pt", "last.pt", "metrics.jsonl", "config.json"} <= set(os.listdir(out))
    with open(os.path.join(out, "metrics.jsonl")) as f:
        assert len(f.readlines()) == 2

    stdout = run_script("scripts/train_seg.py", *common, "train.epochs=3", "--resume")
    assert "resumed" in stdout and "epoch=3" in stdout

    ckpt = os.path.join(out, "model.pt")
    stdout = run_script("scripts/eval_seg.py", "--ckpt", ckpt, ckpt, "--methods", "dnn", "mc", "vq", "ensemble_vq", *common)
    for method in ("dnn", "mc", "vq", "ensemble_vq"):
        with open(os.path.join(out, f"eval_{method}.json")) as f:
            results = json.load(f)
        assert results["method"] == method and 0.0 <= results["acc"] <= 1.0
        assert os.path.exists(os.path.join(out, f"calibration_{method}.png"))
    assert "method" in stdout
