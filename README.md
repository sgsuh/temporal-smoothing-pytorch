# temporal-smoothing-pytorch

PyTorch implementation of [Vector Quantized Bayesian Neural Network Inference for Data Streams](https://arxiv.org/abs/1907.05911)
(VQ-BNN, Park, Lee and Kim, AAAI 2021). The official TensorFlow implementation is
[xxxnell/temporal-smoothing](https://github.com/xxxnell/temporal-smoothing).

Bayesian neural networks estimate uncertainty well but need many forward passes per input.
For data streams such as video, VQ-BNN instead averages the predictions for recent frames,
each from a single forward pass, with exponentially decaying importances:

```
p(y | x_0, D) ~= sum_{t=J}^{-K} pi_t p(y | x_t, w_t),    pi_t ∝ exp(-|t| / tau)
```

This *temporal smoothing* costs about as much as a deterministic network (VQ-DNN when applied
to one) and needs no extra training. See [docs/DESIGN.md](docs/DESIGN.md) for the design and
how this port relates to the official code.

## Setup

All dependencies are installed and all code is run inside the Docker container (PyTorch 2.5.1,
CUDA 12.4, ffmpeg).

```bash
cp .env.example .env              # set DATA_ROOT, mounted at /data in the container
docker compose build
docker compose run --rm dev pytest
```

The repository is bind-mounted at `/workspace` and the container runs as the host user
(`HOST_UID`/`HOST_GID`, default `1000`). Tests that need the real dataset are skipped when it
is absent.

## Data

CamVid (the 701-image release used by SegNet) is expected at `$DATA_ROOT/CamVid`:

```
CamVid/
  train/ train_labels/ val/ val_labels/ test/ test_labels/   # 369 / 100 / 232 images
  videos/                                                    # original videos (for seq/)
  seq/                                                       # extracted 30 Hz frames
```

Temporal smoothing needs the 30 Hz frames around the labeled test frames. Download the original
videos from the [CamVid mirror](http://vis.cs.ucl.ac.uk/Download/G.Brostow/CamVid/)
(`01TP_extract.avi`, `0006R0.MXF`, `0005VD.MXF`, `0016E5.zip.001`, `0016E5.zip.002`, 7.7 GB) into
`CamVid/videos`, then extract the frames:

```bash
docker compose run --rm dev python scripts/prepare_camvid_seq.py --past 5 --future 2
```

The script finds the frame-number offset of every video and verifies it against all 701 labeled
stills before writing `CamVid/seq/{sequence}_{frame}.png`.

## Training and evaluation

```bash
# one model
docker compose run --rm dev python scripts/train_seg.py --config configs/camvid_unet_bnn.yaml
docker compose run --rm dev python scripts/eval_seg.py --config configs/camvid_unet_bnn.yaml \
    --ckpt runs/camvid_unet_bnn/model.pt --methods dnn mc vq

# all four models (U-Net / SegNet, deterministic / MC dropout), resumable
docker compose run -d --name ts-reproduce dev bash scripts/reproduce_camvid.sh

# streaming throughput (one frame at a time)
docker compose run --rm dev python scripts/benchmark_stream.py --config configs/camvid_unet_bnn.yaml \
    --ckpt runs/camvid_unet_bnn/model.pt
```

Config values can be overridden with `--set key.path=value`, e.g. `--set train.epochs=10 eval.future=1`.

Evaluation methods: `dnn` (single pass), `temp` (temperature scaling), `mc` (MC dropout BNN),
`vq` (temporal smoothing: VQ-DNN for a deterministic model, VQ-BNN for an MC dropout model),
`ensemble` and `ensemble_vq` (several `--ckpt`). Reported metrics: NLL, accuracy, mean IoU,
ECE, and Acc / IoU / Unc / Freq for confident pixels at each cutoff (e.g. `acc_90`).

## Results

CamVid-11 test set (232 labeled frames), trained with `scripts/reproduce_camvid.sh` (one seed,
100 epochs, official setup) on an RTX 4070 Laptop GPU. Temporal smoothing uses K = 5 past
frames and tau = 1.25. Stream throughput is measured with `scripts/benchmark_stream.py`
(one 360 x 480 frame at a time, frames/s).

| Model | Method | NLL ↓ | Acc (%) | Acc-90 (%) | Unc-90 (%) | IoU (%) | ECE (%) ↓ | Stream (fps) |
|---|---|---|---|---|---|---|---|---|
| U-Net | DNN | 0.358 | 91.3 | 95.3 | 52.1 | 68.4 | 5.13 | 32.0 |
| U-Net | VQ-DNN | 0.318 | 91.5 | 96.4 | 64.1 | 68.9 | 3.49 | 30.9 |
| U-Net | BNN, single pass | 0.320 | 90.9 | 96.1 | 63.5 | 66.4 | 4.05 | 31.9 |
| U-Net | BNN (MC dropout, 30 samples) | 0.312 | 91.0 | 96.3 | 64.9 | 66.5 | 3.85 | 1.1 |
| U-Net | **VQ-BNN** | **0.293** | 91.1 | **97.1** | **73.3** | 66.8 | **2.47** | 30.6 |
| SegNet | DNN | 0.659 | 84.0 | 91.9 | 59.1 | 51.3 | 9.30 | 49.0 |
| SegNet | VQ-DNN | 0.576 | 84.6 | 94.0 | 70.8 | 52.0 | 6.85 | 46.8 |
| SegNet | BNN, single pass | 0.726 | 79.2 | 92.0 | 73.6 | 47.2 | 9.81 | 48.9 |
| SegNet | BNN (MC dropout, 30 samples) | 0.608 | 80.8 | 95.5 | 85.3 | 49.0 | 5.47 | 1.6 |
| SegNet | **VQ-BNN** | 0.619 | 80.4 | 95.3 | 85.3 | 48.3 | 5.76 | 46.5 |

For reference, the paper reports for U-Net: DNN 0.314 / 91.1 / 96.1 / 61.3 / 66.1 / 4.31,
BNN 0.276 / 91.8 / 96.5 / 63.0 / 68.1 / 3.71 and VQ-BNN 0.253 / 92.0 / 97.4 / 72.4 / 68.6 / 2.24
(NLL / Acc / Acc-90 / Unc-90 / IoU / ECE).

- Temporal smoothing improves NLL, ECE, Acc-90 and Unc-90 over the corresponding single-pass
  model in every case, at almost the cost of a single pass (VQ-BNN is ~29x faster than MC
  dropout with 30 samples). For U-Net, VQ-BNN also beats the 30-sample BNN on every metric.
- Absolute U-Net NLLs are 0.03-0.04 higher than in the paper (single seed here). Validation
  metrics fluctuate strongly during training (batch size 3 with Keras batch norm momentum), so
  the final-epoch checkpoint matters.
- SegNet, trained from scratch as in the official code, is clearly weaker on the test set
  (val accuracy ~86-89 % vs. 80-84 % on test).

## Library usage

```python
import torch
from temporal_smoothing import StreamSmoother, predict_vq
from temporal_smoothing.nn import UNet

model = UNet(num_classes=11, rate=0.5).eval()  # MC dropout stays active in eval mode

# Windowed: frames [B, K + J + 1, 3, H, W], oldest first
probs = predict_vq(model, frames, past=5, future=0, tau=1.25)

# Streaming: one forward pass per frame
smoother = StreamSmoother(past=5, tau=1.25)
for frame in stream:
    probs = smoother.update(model(frame).softmax(dim=1))
```

## Differences from the official code

Behavior that differs from the TensorFlow code by default, each with a switch to reproduce it:

| | Official code | Default here | Switch |
|---|---|---|---|
| ECE bins | accuracy `(lo, hi]`, confidence `[lo, hi)` | both `(lo, hi]` | `eval.legacy=true` |
| Frame windows | may cross sequence boundaries | clamped to the sequence | `eval.boundary=legacy` |
| Ensemble smoothing | last member drawn twice as often | uniform | `eval.legacy=true` |
| Loss reduction | sum over pixels | mean over valid pixels | `train.loss_reduction=sum` |

## Citation

```bibtex
@inproceedings{park2021vector,
  title={Vector Quantized Bayesian Neural Network Inference for Data Streams},
  author={Park, Namuk and Lee, Taekyu and Kim, Songkuk},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={10},
  pages={9322--9330},
  year={2021}
}
```
