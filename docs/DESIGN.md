# Design: temporal-smoothing-pytorch

PyTorch implementation of **VQ-BNN / temporal smoothing**
(Park, Lee, Kim. *Vector Quantized Bayesian Neural Network Inference for Data Streams*, AAAI 2021, [arXiv:1907.05911](https://arxiv.org/abs/1907.05911)).
Reference implementation: [xxxnell/temporal-smoothing](https://github.com/xxxnell/temporal-smoothing) (TensorFlow 2.0).

## 1. 범위

| 포함 (1차) | 제외 (추후) |
|---|---|
| smoothing 코어 (window / stream / EMA) | Depth estimation (NYUDv2) |
| Predictor: DNN, MC(BNN), temp scaling, VQ, ensemble, ensemble smoothing | UCI 분류 (Flipout BNN, OCH 기반 일반 VQ-BNN — 원본에도 미공개) |
| U-Net, SegNet (+ MC dropout) | Simple linear regression 시각화 노트북 |
| CamVid-11 / CamVid-31 / Cityscapes 데이터셋 + 시퀀스 윈도우 | |
| 세그멘테이션 지표, reliability diagram, 학습/평가 스크립트 | |

`smoothing.py`는 Gaussian(moment matching) 버전까지 포함해 depth 확장 시 재사용한다.

## 2. 방법 요약

```
p(y | x_0, D) ≈ Σ_{t=J}^{-K} π_t · p(y | x_t, w_t),    π_t = exp(-|t|/τ) / Σ_s exp(-|s|/τ)
```

- `p(y|x_t,w_t)`: 프레임 t에 대한 **softmax 확률** (logit 평균 아님). BNN이면 프레임마다 다른 dropout mask.
- 기본값: K=5 (과거), J=0 (미래), τ=1.25. 원본 코드의 `l=0.8`은 `1/τ`.
- VQ-DNN = dropout 없는 모델 + 동일 smoothing. 학습 변경 없음 (추론 전용).
- 재귀형(README의 EMA): `q_0 = α p_0 + (1-α) q_{-1}`.

재현 목표 (CamVid, U-Net): DNN NLL 0.314 / Acc 91.1 / ECE 4.31, BNN 0.276 / 91.8 / 3.71,
VQ-DNN 0.284 / 91.2 / 3.00, VQ-BNN 0.253 / 92.0 / 2.24.

## 3. 패키지 구조

```
Dockerfile, docker-compose.yml   # 모든 설치/실행은 컨테이너 안에서 (pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime)
pyproject.toml
temporal_smoothing/
  __init__.py
  smoothing.py
  predictors.py
  nn/
    __init__.py
    mc_dropout.py
    blocks.py          # ConvBlock, DeconvBlock, TF 호환 초기화 (glorot_uniform, zero bias) / BN 설정
    unet.py
    segnet.py
  data/
    __init__.py
    labels.py          # CamVid / Cityscapes Label 테이블, color→index 매핑
    camvid.py
    cityscapes.py
    sequence.py        # SequenceWindowDataset
    transforms.py
  metrics/
    __init__.py
    segmentation.py
    calibration.py     # ECE, confidence histogram, reliability diagram
  engine/
    __init__.py
    train.py
    evaluate.py
scripts/
  train_seg.py
  eval_seg.py
configs/
  camvid_unet_bnn.yaml, camvid_unet_dnn.yaml, camvid_segnet_bnn.yaml, cityscapes_unet_bnn.yaml ...
tests/
```

텐서 레이아웃은 PyTorch 관례(NCHW, 시퀀스는 `[B, T, C, H, W]`, 시간축 오름차순 = 과거→미래)를 따른다.

## 4. 모듈 설계

### 4.1 `smoothing.py`

```python
def exp_decay_weights(past: int, future: int = 0, tau: float = 1.25, *, device=None, dtype=None) -> Tensor:
    """shape [past + future + 1], 인덱스 past 가 현재 프레임. 합 = 1."""

def smooth_categorical(probs: Tensor, weights: Tensor, dim: int = 1) -> Tensor:
    """probs[..., T(dim), ...] 의 가중합. 확률 공간에서 평균."""

def smooth_gaussian(mean: Tensor, var: Tensor, weights: Tensor, dim: int = 1) -> tuple[Tensor, Tensor]:
    """혼합 분포 moment matching: E[μ], E[σ²] + Var[μ]."""

class StreamSmoother:
    """실시간용. 예측을 ring buffer(길이 K+1)에 캐시 → 프레임당 forward 1회.
    update(pred) -> smoothed. 버퍼가 덜 찼을 때는 존재하는 프레임만으로 재정규화. reset()."""

class EMASmoother:
    """무한 EMA. alpha 또는 tau(alpha = 1 - exp(-1/τ))로 생성. update(pred) -> smoothed.
    bias correction(누적 가중치로 정규화) 적용 → 무한 윈도우 StreamSmoother와 동일."""
```

- `StreamSmoother`/`EMASmoother`는 인과적(causal) 형태라 미래 프레임(J>0)은 지원하지 않는다. J>0은 윈도우 방식(`predict_vq`)으로 처리한다.
- 두 smoother는 단순 가중평균이므로 확률뿐 아니라 임의 텐서(예: Gaussian mean)에도 쓸 수 있다.

### 4.2 `predictors.py`

모든 predictor는 `(model, xs) -> probs[B, num_classes, H, W]` 형태이고, 모델은 logit을 반환한다.

| 함수 | 입력 | 설명 |
|---|---|---|
| `predict_dnn(model, x)` | `[B,C,H,W]` | softmax 1회 |
| `predict_temp_scaling(model, x, temp)` | `[B,C,H,W]` | `softmax(logit / temp)` |
| `predict_mc(model, x, n_samples)` | `[B,C,H,W]` | MC dropout N회 평균 (BNN) |
| `predict_vq(model, xs, past, future, tau)` | `[B,T,C,H,W]` | `[B·T]`로 펼쳐 forward 1회 → softmax → `smooth_categorical` |
| `predict_ensemble(models, x)` | `[B,C,H,W]` | 모델별 softmax 평균 |
| `predict_ensemble_smoothing(models, xs, past, future, tau, generator=None, legacy=False)` | `[B,T,C,H,W]` | 프레임마다 무작위 모델 1개(배치 공유) → smoothing |

- `predict_vq`의 배치화는 TF의 "프레임별 개별 forward"와 수학적으로 동일하다 (dropout mask는 샘플 단위로 독립). 메모리 한계용 `chunk_size` 옵션 제공.
- 모든 predictor는 `@torch.no_grad()`, 호출 측에서 `model.eval()` 보장 (`MCDropout`은 eval에서도 활성).
  배치화가 프레임별 forward와 동치인 것은 BN이 running stats를 쓸 때뿐이므로, train 모드 모델이 들어오면 `UserWarning`.

### 4.3 `nn/`

- `MCDropout(nn.Dropout)`: `forward`에서 항상 `F.dropout(x, p, training=True)`. `p=0`이면 identity → 같은 클래스로 DNN/BNN 표현. `set_mc_dropout(model, enabled=False)`이면 일반 dropout으로 돌아가 `model.training`을 따름 (BNN의 deterministic 근사 평가용).
- `UNet(num_classes, in_channels=3, rate=0.0)` — 원본 그대로:
  - ConvBlock = Conv3x3 → **ReLU → BN** (원본 순서)
  - encoder 64-64 / 128-128 / 256×3 / 512×3 / 1024×3, MaxPool 2x2 `ceil_mode=True` (TF `SAME`)
  - decoder: ConvTranspose 3x3 s2 → ReLU → BN, skip은 **덧셈**, 512×3 / 256×3 / 128×3 / 64×2, 1x1 conv head
  - MCDropout 위치: block 3·4·5 (encoder, conv 후) / 6·7·8 (decoder) — 총 6개
  - 업샘플 출력 크기는 `output_size=skip.shape[-2:]`로 지정 (홀수 해상도 대응)
  - deconv는 bias 없음 (원본은 `tf.Variable` filter만 사용)
  - TF `conv2d_transpose(SAME)` 정렬 재현: `ConvTranspose2d(padding=0)`로 전체 출력을 만든 뒤 앞쪽을 `pad_total // 2`만큼 crop.
    PyTorch 관용 방식(`padding=1, output_padding=1`)은 1픽셀 어긋남. 테스트에서 SAME conv와의 adjoint 관계로 검증.
- `SegNet(num_classes, in_channels=3, rate=0.0)` — 원본 그대로:
  - ConvBlock = Conv3x3 → BN → ReLU
  - `MaxPool2d(2, 2, ceil_mode=True, return_indices=True)` + `MaxUnpool2d(output_size=...)`
  - MCDropout 위치: pool3·pool4 후, block 5·6·7·8 후 — 총 6개
- TF 호환 기본값: Conv/ConvTranspose `xavier_uniform_` + zero bias, `BatchNorm2d(eps=1e-3, momentum=0.01)`. `tf_compat_init=False`로 PyTorch 기본값 사용 가능.
- TF 체크포인트가 공개되지 않았으므로 weight 변환기는 만들지 않는다. 목표는 지표 재현이다.

### 4.4 `data/`

- `labels.py`: 원본 `camvid_labels`, `cityscape_labels` 테이블 이식. `color_map(name) -> dict[rgb, index]`
  - `camvid` / `camvid-11`: category 기준 11 클래스, `camvid-31`: 원 라벨, `cityscapes`: trainId 19 클래스
  - void / ignore → `-1` (loss·지표에서 마스킹)
- 전처리 (원본 일치): PIL 로드 → **nearest** 리사이즈 (CamVid 360×480, Cityscapes 512×1024) → `/255` (정규화 없음).
  학습 augmentation: random crop (CamVid 기본은 이미지 크기 = 사실상 없음), flip (기본 off).
- `CamVid(root, split, ...)`, `Cityscapes(root, split, ...)`: `(image[3,H,W] float, label[H,W] long)`.
  Cityscapes는 원본처럼 `val`을 test로 사용한다.
- `SequenceWindowDataset(frames, labels, past, future)`:
  - 라벨이 있는 프레임마다 `(frames[T,3,H,W], label[H,W])`.
  - 시퀀스 ID(CamVid: `0001TP` 등 prefix, Cityscapes: `city_seq`)로 그룹핑해서 **윈도우가 시퀀스를 넘지 않게** 한다. 넘는 경우 `boundary="clamp"`(가장자리 프레임 반복, 기본) / `"skip"` / `"legacy"`(원본처럼 경계 무시).
- 데이터 준비: CamVid 시퀀스 프레임(30fps 영상 → `{seq}_{frame:06d}.png`)과 Cityscapes `leftImg8bit_sequence`는 사용자가 직접 받는다. 디렉터리 규약은 원본과 같게 하고, README에 준비 방법을 적는다.

### 4.5 `metrics/`

- `SegmentationMeter(num_classes, cutoffs=(0.0, 0.9), n_bins=10, legacy=False)`:
  - `update(probs, target, mask=None)`, `compute() -> dict`
  - NLL: 마스킹된 픽셀 평균 `-log(clamp(p_y, 1e-7))` (Keras와 같은 clip)
  - cutoff별 certain/uncertain confusion matrix → Acc, mIoU(GT에 존재하는 클래스만 평균), Unc = p(unconfident | inaccurate), Freq(Cov) = p(confident)
  - 결과 키: `nll, acc, acc_90, iou, iou_90, unc_90, freq_90, ece, bins{count, acc, conf}`
- ECE: 10 bin. 기본은 count/acc/conf 모두 `(lo, hi]` 경계(첫 bin은 0 포함). `legacy=True`면 원본의 불일치 경계(cm `(lo,hi]`, conf `[lo,hi)`)를 재현한다.
- edge mask (Sobel 크기 > edge) 옵션 이식. 시퀀스 입력이면 현재 프레임(`index=past`) 기준이다. 원본의 `xs[:, -1]`은 J>0일 때 틀리므로 수정한다.
- `plot_calibration(bins) -> Figure` (confidence histogram + reliability diagram).

### 4.6 `engine/` & `scripts/`

- `train_one_epoch(model, loader, optimizer, class_weights, loss_reduction="mean")`:
  - weighted CE, `ignore_index=-1`. 원본은 사실상 `sum`이고 Adam은 스케일에 거의 불변이므로 기본은 `mean`, `"sum"` 옵션을 둔다.
  - `model.train()` (BN 학습 모드, MCDropout 항상 on).
- class weights: 원본의 memorized median-frequency 값을 이식하고 `compute_median_freq_weights(dataset)`도 제공한다.
- `evaluate(predictor, loader, meter)` + 추론 시간 측정. CUDA면 `synchronize` 후 측정한다.
- 기본 하이퍼파라미터: Adam(lr 1e-3, β=(0.9, 0.999)), batch 3, CamVid 100 epoch / Cityscapes 500 epoch, rate 0.5, MC 30 samples, K=5, J=0, τ=1.25.
- `scripts/train_seg.py --config configs/camvid_unet_bnn.yaml`, `scripts/eval_seg.py --config ... --ckpt ... --method {dnn,mc,vq,temp,ensemble,ensemble_vq}`
- 설정은 yaml → dataclass. CLI 인자로 덮어쓸 수 있다. 로깅은 TensorBoard(선택) + stdout.
- 재현성: `seed` 설정, `torch.Generator` 전달.

## 5. 원본 코드와의 차이 (legacy 플래그)

| 항목 | 원본 동작 | 기본 동작 | 호환 |
|---|---|---|---|
| ECE bin 경계 | cm `(lo,hi]`, conf `[lo,hi)` 불일치 | 둘 다 `(lo,hi]` | `legacy=True` |
| 시퀀스 윈도우 | 평탄화된 파일 리스트에서 슬라이딩 → 시퀀스 경계 넘음 | 시퀀스 내부로 제한 (clamp) | `boundary="legacy"` |
| ensemble smoothing 모델 선택 | `randint(0,n)-1` → 마지막 모델 2배 확률 | 균등 | `legacy=True` |
| edge mask (seq) | `xs[:, -1]` (J>0이면 미래 프레임) | 현재 프레임 | `legacy=True` |
| loss reduction | 픽셀 벡터 gradient = sum | mean | `loss_reduction="sum"` |
| VQ 평가 forward | 프레임별 개별 호출 | `[B·T]` 배치 1회 (동치) | — |

## 6. 테스트 계획 (pytest)

- `exp_decay_weights`: 합 1, 인접 비율 `e^{1/τ}`, TF 수식(`range(K) + range(K, K-J-1, -1)`)과 일치.
- `StreamSmoother` 결과 == `predict_vq` 윈도우 결과 (DNN 모델, 버퍼가 찬 이후).
- `MCDropout`: `model.eval()`에서도 두 번의 forward 결과가 다름. `p=0`이면 같음.
- UNet/SegNet: 홀수 해상도(예: 360×480 → 45×60 → 23×30) 입출력 shape 일치.
- `SegmentationMeter`: 작은 수작업 예제와 numpy 기준 구현(원본 함수 이식)으로 Acc/mIoU/Unc/Freq/ECE 검증. legacy 모드 차이 확인.
- `SequenceWindowDataset`: 경계 clamp/skip 동작, 라벨 프레임 인덱스 정렬.

## 7. 구현 순서

1. `pyproject.toml`, 패키지 골격, `smoothing.py` + 테스트
2. `nn/` (MCDropout, UNet, SegNet) + shape 테스트
3. `predictors.py` + 테스트
4. `metrics/` + numpy 기준 테스트
5. `data/` (labels, CamVid, Cityscapes, SequenceWindowDataset)
6. `engine/`, `scripts/`, configs
7. README (설치·데이터 준비·사용법), CamVid 재현 실험
8. (추후) depth estimation, Gaussian smoothing 경로
