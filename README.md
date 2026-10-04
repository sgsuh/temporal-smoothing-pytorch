# temporal-smoothing-pytorch

PyTorch implementation of [Vector Quantized Bayesian Neural Network Inference for Data Streams](https://arxiv.org/abs/1907.05911) (VQ-BNN, AAAI 2021).
The official TensorFlow implementation is [xxxnell/temporal-smoothing](https://github.com/xxxnell/temporal-smoothing).

Work in progress. See [docs/DESIGN.md](docs/DESIGN.md) for the design and roadmap.

## Development

All dependencies are installed and all code is run inside the Docker container.

```bash
docker compose build                       # build the dev image (PyTorch + CUDA)
docker compose run --rm dev pytest         # run the test suite
docker compose run --rm dev                # interactive shell
```

The repository is bind-mounted at `/workspace`, and the container runs as the host user
(`HOST_UID`/`HOST_GID`, default `1000`).

Datasets are mounted from `DATA_ROOT` (set in `.env`, see `.env.example`) at `/data`.
CamVid is expected at `$DATA_ROOT/CamVid` with `train`, `val`, `test` and `*_labels`
directories; tests that need the real dataset are skipped when it is absent.

Temporal smoothing additionally needs the 30 Hz frames around the labeled test frames.
Download the original videos from the [CamVid mirror](http://vis.cs.ucl.ac.uk/Download/G.Brostow/CamVid/)
(`01TP_extract.avi`, `0006R0.MXF`, `0005VD.MXF`, `0016E5.zip.001`, `0016E5.zip.002`) into
`$DATA_ROOT/CamVid/videos`, then extract and verify the frames:

```bash
docker compose run --rm dev python scripts/prepare_camvid_seq.py --past 5 --future 2
```

## Training and evaluation

```bash
docker compose run --rm dev python scripts/train_seg.py --config configs/camvid_unet_bnn.yaml
docker compose run --rm dev python scripts/eval_seg.py --config configs/camvid_unet_bnn.yaml \
    --ckpt runs/camvid_unet_bnn/model.pt --methods dnn mc vq
```

Config values can be overridden with `--set key.path=value`, e.g. `--set train.epochs=10 eval.future=1`.

## Usage

```python
import torch
from temporal_smoothing import StreamSmoother, exp_decay_weights, smooth_categorical

# Windowed: probs for the past K=5 frames and the current one, [B, T, num_classes, H, W]
weights = exp_decay_weights(past=5, future=0, tau=1.25)
probs = smooth_categorical(window_probs, weights, dim=1)

# Streaming: one forward pass per frame
smoother = StreamSmoother(past=5, tau=1.25)
for frame in stream:
    probs = smoother.update(model(frame).softmax(dim=1))
```
