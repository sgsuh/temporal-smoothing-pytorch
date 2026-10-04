"""Train a (Bayesian) U-Net / SegNet on CamVid.

Usage (inside the container)::

    python scripts/train_seg.py --config configs/camvid_unet_bnn.yaml
    python scripts/train_seg.py --config configs/camvid_unet_bnn.yaml --set train.epochs=10 output_dir=runs/debug
    python scripts/train_seg.py --config configs/camvid_unet_bnn.yaml --resume
"""

import argparse

from temporal_smoothing.engine import fit, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    parser.add_argument("--resume", action="store_true", help="resume from OUTPUT_DIR/last.pt")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    print(f"training {config.model.arch} (rate={config.model.rate}) -> {config.output_dir}", flush=True)
    fit(config, resume=args.resume, log=lambda msg: print(msg, flush=True))


if __name__ == "__main__":
    main()
