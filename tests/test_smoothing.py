import math

import pytest
import torch

from temporal_smoothing import (
    EMASmoother,
    StreamSmoother,
    exp_decay_weights,
    smooth_categorical,
    smooth_gaussian,
)


def tf_reference_weights(past, future, l):
    """Weights as computed by `predict_temporal_smoothing` in the official TF code."""
    weight = list(range(past)) + list(range(past, past - future - 1, -1))
    weight = torch.tensor(weight, dtype=torch.float64) * l
    weight = torch.exp(weight)
    return weight / weight.sum()


def random_probs(*shape, dim):
    return torch.randn(*shape, dtype=torch.float64).softmax(dim=dim)


# exp_decay_weights


@pytest.mark.parametrize("past,future", [(0, 0), (5, 0), (5, 1), (5, 2), (3, 3), (0, 2)])
def test_weights_shape_and_normalization(past, future):
    weights = exp_decay_weights(past, future, tau=1.25, dtype=torch.float64)
    assert weights.shape == (past + future + 1,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert weights.argmax().item() == past


def test_weights_decay_ratio():
    tau = 1.25
    weights = exp_decay_weights(5, 2, tau=tau, dtype=torch.float64)
    ratio = math.exp(1.0 / tau)
    torch.testing.assert_close(weights[1:6] / weights[0:5], torch.full((5,), ratio, dtype=torch.float64))
    torch.testing.assert_close(weights[5:7] / weights[6:8], torch.full((2,), ratio, dtype=torch.float64))


@pytest.mark.parametrize("past,future,l", [(5, 0, 0.8), (5, 1, 0.8), (5, 2, 1.0), (10, 3, 0.3)])
def test_weights_match_tf_reference(past, future, l):
    weights = exp_decay_weights(past, future, tau=1.0 / l, dtype=torch.float64)
    torch.testing.assert_close(weights, tf_reference_weights(past, future, l))


def test_weights_dtype_and_device():
    assert exp_decay_weights(5).dtype == torch.get_default_dtype()
    assert exp_decay_weights(5, dtype=torch.float16).dtype == torch.float16


@pytest.mark.parametrize("kwargs", [dict(past=-1), dict(past=1, future=-1), dict(past=1, tau=0.0)])
def test_weights_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        exp_decay_weights(**kwargs)


# smooth_categorical


def test_smooth_categorical_matches_loop():
    probs = random_probs(2, 6, 4, 3, 5, dim=2)
    weights = exp_decay_weights(5, dtype=torch.float64)
    expected = sum(weights[t] * probs[:, t] for t in range(6))
    out = smooth_categorical(probs, weights, dim=1)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(out.sum(dim=1), torch.ones(2, 3, 5, dtype=torch.float64))


def test_smooth_categorical_other_dim_and_single_frame():
    probs = random_probs(6, 2, 4, dim=2)
    weights = exp_decay_weights(5, dtype=torch.float64)
    torch.testing.assert_close(
        smooth_categorical(probs, weights, dim=0),
        smooth_categorical(probs.transpose(0, 1), weights, dim=1),
    )
    single = random_probs(2, 1, 4, dim=2)
    torch.testing.assert_close(smooth_categorical(single, exp_decay_weights(0), dim=1), single[:, 0])


def test_smooth_categorical_casts_weights():
    probs = random_probs(2, 6, 4, dim=2).float()
    out = smooth_categorical(probs, exp_decay_weights(5, dtype=torch.float64), dim=1)
    assert out.dtype == torch.float32


def test_smooth_categorical_size_mismatch():
    with pytest.raises(ValueError):
        smooth_categorical(random_probs(2, 6, 4, dim=2), exp_decay_weights(4), dim=1)
    with pytest.raises(ValueError):
        smooth_categorical(random_probs(2, 6, 4, dim=2), torch.ones(6, 1), dim=1)


# smooth_gaussian


def test_smooth_gaussian_matches_mixture_moments():
    mean = torch.randn(2, 6, 1, 3, 3, dtype=torch.float64)
    var = torch.rand(2, 6, 1, 3, 3, dtype=torch.float64)
    weights = exp_decay_weights(5, dtype=torch.float64)
    w = weights.view(1, -1, 1, 1, 1)

    second_moment = (w * (var + mean**2)).sum(dim=1)
    expected_mean = (w * mean).sum(dim=1)
    expected_var = second_moment - expected_mean**2

    out_mean, out_var = smooth_gaussian(mean, var, weights, dim=1)
    torch.testing.assert_close(out_mean, expected_mean)
    torch.testing.assert_close(out_var, expected_var)


def test_smooth_gaussian_constant_variance():
    mean = torch.randn(2, 6, 3, dtype=torch.float64)
    weights = exp_decay_weights(5, dtype=torch.float64)
    out_mean, out_var = smooth_gaussian(mean, 0.4, weights, dim=1)
    ref_mean, ref_var = smooth_gaussian(mean, torch.full_like(mean, 0.4), weights, dim=1)
    torch.testing.assert_close(out_mean, ref_mean)
    torch.testing.assert_close(out_var, ref_var)


def test_smooth_gaussian_identical_frames_have_no_spread():
    mean = torch.randn(2, 1, 3, dtype=torch.float64).expand(2, 6, 3)
    out_mean, out_var = smooth_gaussian(mean, 0.4, exp_decay_weights(5, dtype=torch.float64), dim=1)
    torch.testing.assert_close(out_mean, mean[:, 0])
    torch.testing.assert_close(out_var, torch.full_like(out_var, 0.4))


# StreamSmoother


def test_stream_smoother_first_frame_is_identity():
    smoother = StreamSmoother(past=5, tau=1.25)
    pred = random_probs(2, 4, 3, 3, dim=1)
    torch.testing.assert_close(smoother.update(pred), pred)
    assert len(smoother) == 1


def test_stream_smoother_matches_window_once_full():
    past, tau = 5, 1.25
    stream = random_probs(10, 2, 4, 3, 3, dim=2)
    smoother = StreamSmoother(past=past, tau=tau)
    weights = exp_decay_weights(past, 0, tau, dtype=torch.float64)

    for t in range(stream.shape[0]):
        out = smoother.update(stream[t])
        if t >= past:
            window = stream[t - past : t + 1].transpose(0, 1)  # [B, T, ...]
            torch.testing.assert_close(out, smooth_categorical(window, weights, dim=1))
    assert len(smoother) == past + 1


def test_stream_smoother_renormalizes_partial_buffer():
    stream = random_probs(3, 2, 4, dim=2)
    smoother = StreamSmoother(past=5, tau=1.25)
    for t in range(3):
        out = smoother.update(stream[t])
    weights = exp_decay_weights(2, 0, 1.25, dtype=torch.float64)
    torch.testing.assert_close(out, smooth_categorical(stream.transpose(0, 1), weights, dim=1))


def test_stream_smoother_reset_and_shape_mismatch():
    smoother = StreamSmoother(past=2)
    smoother.update(torch.rand(2, 4))
    with pytest.raises(ValueError):
        smoother.update(torch.rand(3, 4))
    smoother.reset()
    assert len(smoother) == 0
    pred = torch.rand(3, 4)
    torch.testing.assert_close(smoother.update(pred), pred)


# EMASmoother


def test_ema_alpha_from_tau():
    tau = 1.25
    smoother = EMASmoother(tau=tau)
    infinite_sum = 1.0 / (1.0 - math.exp(-1.0 / tau))
    assert smoother.alpha == pytest.approx(1.0 / infinite_sum)


def test_ema_matches_unbounded_stream_smoother():
    tau, steps = 1.25, 8
    stream = random_probs(steps, 2, 4, dim=2)
    ema = EMASmoother(tau=tau)
    window = StreamSmoother(past=steps, tau=tau)
    for t in range(steps):
        torch.testing.assert_close(ema.update(stream[t]), window.update(stream[t]))


def test_ema_recursion_after_warmup():
    alpha = 0.3
    ema = EMASmoother(alpha=alpha)
    stream = random_probs(200, 4, dim=1)
    for t in range(199):
        prev = ema.update(stream[t])
    out = ema.update(stream[199])
    # Bias correction vanishes after many frames, leaving the plain recursion.
    torch.testing.assert_close(out, alpha * stream[199] + (1 - alpha) * prev)


def test_ema_reset_and_shape_mismatch():
    ema = EMASmoother(alpha=0.5)
    ema.update(torch.rand(2, 4))
    with pytest.raises(ValueError):
        ema.update(torch.rand(3, 4))
    ema.reset()
    pred = torch.rand(3, 4)
    torch.testing.assert_close(ema.update(pred), pred)


@pytest.mark.parametrize("kwargs", [dict(), dict(alpha=0.5, tau=1.0), dict(alpha=0.0), dict(alpha=1.5), dict(tau=-1.0)])
def test_ema_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        EMASmoother(**kwargs)
