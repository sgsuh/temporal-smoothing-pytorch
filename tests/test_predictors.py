from collections import Counter

import pytest
import torch
from torch import nn

from temporal_smoothing import (
    StreamSmoother,
    exp_decay_weights,
    predict_dnn,
    predict_ensemble,
    predict_ensemble_smoothing,
    predict_mc,
    predict_temp_scaling,
    predict_vq,
    smooth_categorical,
)
from temporal_smoothing.nn import MCDropout, UNet
from temporal_smoothing.predictors import sample_model_indices

NUM_CLASSES = 4


def toy_model(rate=0.0, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=1),
        nn.BatchNorm2d(8),
        MCDropout(rate),
        nn.Conv2d(8, NUM_CLASSES, kernel_size=1),
    ).eval()


class ConstantModel(nn.Module):
    """Predicts the same logits for every pixel."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.as_tensor(logits, dtype=torch.float32))

    def forward(self, x):
        return self.logits.view(1, -1, 1, 1).expand(x.shape[0], -1, *x.shape[-2:])


def frames(batch=2, length=6, size=(5, 7), seed=1):
    torch.manual_seed(seed)
    return torch.randn(batch, length, 3, *size)


def assert_probs(probs, shape):
    assert probs.shape == shape
    torch.testing.assert_close(probs.sum(dim=1), torch.ones(shape[0], *shape[2:]))


# Single-frame predictors


def test_predict_dnn_is_softmax():
    model, x = toy_model(), frames()[:, 0]
    probs = predict_dnn(model, x)
    assert_probs(probs, (2, NUM_CLASSES, 5, 7))
    torch.testing.assert_close(probs, model(x).softmax(dim=1))


def test_predict_temp_scaling():
    model, x = toy_model(), frames()[:, 0]
    torch.testing.assert_close(predict_temp_scaling(model, x, 1.0), predict_dnn(model, x))
    flat = predict_temp_scaling(model, x, 1e6)
    torch.testing.assert_close(flat, torch.full_like(flat, 1.0 / NUM_CLASSES))


def test_predict_mc_dnn_equals_single_pass():
    model, x = toy_model(rate=0.0), frames()[:, 0]
    torch.testing.assert_close(predict_mc(model, x, n_samples=5), predict_dnn(model, x))


def test_predict_mc_averages_samples():
    model, x = toy_model(rate=0.5), frames()[:, 0]
    torch.manual_seed(123)
    probs = predict_mc(model, x, n_samples=5)
    torch.manual_seed(123)
    with torch.no_grad():
        expected = torch.stack([model(x).softmax(dim=1) for _ in range(5)]).mean(dim=0)
    assert_probs(probs, (2, NUM_CLASSES, 5, 7))
    torch.testing.assert_close(probs, expected)
    assert not torch.allclose(probs, predict_dnn(model, x))


def test_predict_mc_invalid_samples():
    with pytest.raises(ValueError):
        predict_mc(toy_model(), frames()[:, 0], n_samples=0)


# predict_vq


@pytest.mark.parametrize("past,future", [(5, 0), (4, 1), (0, 0)])
def test_predict_vq_matches_per_frame_smoothing(past, future):
    model, xs = toy_model(), frames(length=past + future + 1)
    probs = predict_vq(model, xs, past=past, future=future, tau=1.25)
    with torch.no_grad():
        per_frame = torch.stack([model(xs[:, t]).softmax(dim=1) for t in range(xs.shape[1])], dim=1)
    expected = smooth_categorical(per_frame, exp_decay_weights(past, future, 1.25), dim=1)
    assert_probs(probs, (2, NUM_CLASSES, 5, 7))
    torch.testing.assert_close(probs, expected)


def test_predict_vq_chunking_is_equivalent():
    model, xs = toy_model(), frames()
    torch.testing.assert_close(predict_vq(model, xs, chunk_size=5), predict_vq(model, xs))


def test_predict_vq_static_scene_equals_dnn():
    model = toy_model()
    x = frames()[:, :1]
    xs = x.expand(-1, 6, -1, -1, -1)
    torch.testing.assert_close(predict_vq(model, xs), predict_dnn(model, x[:, 0]))


def test_predict_vq_matches_stream_smoother():
    past, tau = 5, 1.25
    model, stream = toy_model(), frames(batch=1, length=12)
    smoother = StreamSmoother(past=past, tau=tau)
    for t in range(stream.shape[1]):
        streamed = smoother.update(predict_dnn(model, stream[:, t]))
        if t >= past:
            windowed = predict_vq(model, stream[:, t - past : t + 1], past=past, tau=tau)
            torch.testing.assert_close(streamed, windowed)


def test_predict_vq_bnn_uses_independent_samples_per_frame():
    model = toy_model(rate=0.5)
    xs = frames()[:, :1].expand(-1, 6, -1, -1, -1)
    # Identical frames still give different predictions because every frame draws new weights.
    assert not torch.allclose(predict_vq(model, xs), predict_dnn(toy_model(rate=0.0), xs[:, 0]))


def test_predict_vq_window_size_mismatch():
    with pytest.raises(ValueError):
        predict_vq(toy_model(), frames(length=5), past=5, future=0)


def test_predict_vq_with_unet():
    model = UNet(NUM_CLASSES, rate=0.5).eval()
    probs = predict_vq(model, torch.randn(1, 3, 3, 32, 48), past=2)
    assert_probs(probs, (1, NUM_CLASSES, 32, 48))


def test_warns_in_training_mode():
    model = toy_model().train()
    with pytest.warns(UserWarning, match="training mode"):
        predict_dnn(model, frames()[:, 0])


# Ensembles


def test_predict_ensemble_is_mean():
    models = [toy_model(seed=s) for s in range(3)]
    x = frames()[:, 0]
    expected = torch.stack([predict_dnn(m, x) for m in models]).mean(dim=0)
    torch.testing.assert_close(predict_ensemble(models, x), expected)


def test_sample_model_indices_uniform_and_legacy():
    n, draws = 4, 40000
    gen = torch.Generator().manual_seed(0)
    uniform = Counter(sample_model_indices(n, draws, gen))
    legacy = Counter(sample_model_indices(n, draws, gen, legacy=True))
    assert set(uniform) == set(range(n))
    for i in range(n):
        assert uniform[i] / draws == pytest.approx(1 / n, abs=0.01)
    # randint(0, n) - 1 in the official code: the last member also absorbs index -1.
    for i in range(n - 1):
        assert legacy[i] / draws == pytest.approx(1 / (n + 1), abs=0.01)
    assert legacy[n - 1] / draws == pytest.approx(2 / (n + 1), abs=0.01)


def test_predict_ensemble_smoothing_uses_sampled_members():
    eye = torch.eye(NUM_CLASSES) * 100.0
    models = [ConstantModel(eye[i]).eval() for i in range(NUM_CLASSES)]
    xs = frames(length=6)
    gen = torch.Generator().manual_seed(0)
    probs = predict_ensemble_smoothing(models, xs, past=5, generator=gen)

    indices = sample_model_indices(NUM_CLASSES, 6, torch.Generator().manual_seed(0))
    weights = exp_decay_weights(5)
    expected = torch.zeros(NUM_CLASSES)
    for t, i in enumerate(indices):
        expected[i] += weights[t]
    assert_probs(probs, (2, NUM_CLASSES, 5, 7))
    torch.testing.assert_close(probs[0, :, 0, 0], expected)


def test_ensembles_reject_empty():
    with pytest.raises(ValueError):
        predict_ensemble([], frames()[:, 0])
    with pytest.raises(ValueError):
        predict_ensemble_smoothing([], frames())
