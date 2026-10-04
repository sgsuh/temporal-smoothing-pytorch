"""Evaluate DNN / MC (BNN) / VQ (temporal smoothing) / ensemble predictions on CamVid test windows.

Usage (inside the container)::

    python scripts/eval_seg.py --config configs/camvid_unet_bnn.yaml --ckpt runs/camvid_unet_bnn/model.pt
    python scripts/eval_seg.py --config configs/camvid_unet_bnn.yaml --ckpt CKPT --methods dnn vq \\
        --set eval.past=5 eval.future=1 eval.tau=1.25
    python scripts/eval_seg.py --config configs/camvid_unet_dnn.yaml --ckpt A.pt B.pt C.pt --methods ensemble ensemble_vq

The model architecture is read from each checkpoint (falling back to the config). Results are
written to ``OUT/eval_{method}.json`` together with a calibration plot, where ``OUT`` defaults
to the directory of the first checkpoint.
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from temporal_smoothing.data import camvid_sequence, num_classes
from temporal_smoothing.engine import METHODS, ModelConfig, build_model, evaluate, load_checkpoint, load_config, make_predictor
from temporal_smoothing.engine.train import seed_everything
from temporal_smoothing.metrics import plot_calibration

SUMMARY_KEYS = ("nll", "acc", "acc_90", "iou", "iou_90", "unc_90", "freq_90", "ece", "time_ms", "throughput")


def load_model(path: str, fallback: ModelConfig, n_classes: int, device: torch.device) -> torch.nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model_config = ModelConfig(**state["config"]["model"]) if "config" in state else fallback
    model = build_model(model_config, n_classes)
    load_checkpoint(path, model)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument("--ckpt", nargs="+", required=True, help="checkpoint(s); several for ensembles")
    parser.add_argument("--methods", nargs="+", choices=METHODS, help="overrides eval.methods")
    parser.add_argument("--out", help="output directory (default: directory of the first checkpoint)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    eval_config = config.eval
    methods = args.methods or eval_config.methods
    out = args.out or os.path.dirname(os.path.abspath(args.ckpt[0]))
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_classes = num_classes(config.data.name)
    models = [load_model(path, config.model, n_classes, device) for path in args.ckpt]
    dataset = camvid_sequence(
        config.data.root,
        name=config.data.name,
        size=config.data.size,
        past=eval_config.past,
        future=eval_config.future,
        boundary=eval_config.boundary,
    )
    loader = DataLoader(dataset, eval_config.batch_size, shuffle=False,
                        num_workers=config.data.num_workers, pin_memory=device.type == "cuda")
    print(f"{len(dataset)} test windows (past={eval_config.past}, future={eval_config.future}), device={device}", flush=True)

    summary = {}
    for method in methods:
        seed_everything(eval_config.seed)
        predictor = make_predictor(method, models if method.startswith("ensemble") else models[:1], eval_config)
        print(f"[{method}]", flush=True)
        results = evaluate(predictor, loader, n_classes, eval_config, device, log=lambda m: print(m, flush=True))
        results["method"] = method
        results["checkpoints"] = args.ckpt
        results["eval"] = {k: list(v) if isinstance(v, tuple) else v for k, v in vars(eval_config).items()}
        with open(os.path.join(out, f"eval_{method}.json"), "w") as f:
            json.dump(results, f, indent=2)
        plot_calibration(results["bins"]).savefig(os.path.join(out, f"calibration_{method}.png"))
        summary[method] = {k: results[k] for k in SUMMARY_KEYS if k in results}
        print("  " + ", ".join(f"{k}={v:.4f}" for k, v in summary[method].items()), flush=True)

    header = f"{'method':<12}" + "".join(f"{k:>11}" for k in SUMMARY_KEYS)
    print("\n" + header)
    for method, values in summary.items():
        print(f"{method:<12}" + "".join(f"{values.get(k, float('nan')):>11.4f}" for k in SUMMARY_KEYS))


if __name__ == "__main__":
    main()
