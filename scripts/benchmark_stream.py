"""Streaming throughput: frames are processed one at a time, in order, as from a camera.

- ``dnn``: one forward pass per frame
- ``mc``: ``mc_samples`` forward passes per frame (BNN)
- ``vq``: one forward pass per frame, smoothed over cached predictions (``StreamSmoother``)

Frames are preloaded to the device, so only prediction is timed.

Usage (inside the container)::

    python scripts/benchmark_stream.py --config configs/camvid_unet_bnn.yaml --ckpt runs/camvid_unet_bnn/model.pt
"""

import argparse
import json
import os
import time

import torch

from temporal_smoothing import StreamSmoother
from temporal_smoothing.data import CAMVID_SIZE, load_image, num_classes
from temporal_smoothing.engine import ModelConfig, build_model, load_checkpoint, load_config

METHODS = ("dnn", "mc", "vq")


def load_model(path, fallback, n_classes, device):
    model_config = fallback
    if path:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model_config = ModelConfig(**state["config"]["model"]) if "config" in state else fallback
    model = build_model(model_config, n_classes)
    if path:
        load_checkpoint(path, model)
    return model.to(device).eval()


@torch.inference_mode()
def run(method, model, frames, mc_samples, past, tau):
    smoother = StreamSmoother(past=past, tau=tau)
    for frame in frames:
        x = frame.unsqueeze(0)
        if method == "dnn":
            probs = model(x).softmax(dim=1)
        elif method == "mc":
            probs = sum(model(x).softmax(dim=1) for _ in range(mc_samples)) / mc_samples
        else:
            probs = smoother.update(model(x).softmax(dim=1))
    return probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", help="checkpoint (random weights if omitted; timing is unaffected)")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--frames", type=int, default=200, help="number of frames to stream")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", help="JSON output path")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, config.model, num_classes(config.data.name), device)

    seq_dir = os.path.join(config.data.root, "seq")
    paths = sorted(os.path.join(seq_dir, f) for f in os.listdir(seq_dir) if f.endswith(".png"))
    size = config.data.size or CAMVID_SIZE
    frames = [load_image(p, size).to(device) for p in paths[: args.frames + args.warmup]]
    warmup, frames = frames[: args.warmup], frames[args.warmup :]

    results = {}
    for method in args.methods:
        run(method, model, warmup, config.eval.mc_samples, config.eval.past, config.eval.tau)
        if device.type == "cuda":
            torch.cuda.synchronize()
        tic = time.perf_counter()
        run(method, model, frames, config.eval.mc_samples, config.eval.past, config.eval.tau)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - tic
        results[method] = {"fps": len(frames) / elapsed, "ms_per_frame": elapsed / len(frames) * 1e3}
        print(f"{method:<4} {results[method]['fps']:8.2f} frames/s  {results[method]['ms_per_frame']:8.2f} ms/frame", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"device": str(device), "frames": len(frames), "mc_samples": config.eval.mc_samples,
                       "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
