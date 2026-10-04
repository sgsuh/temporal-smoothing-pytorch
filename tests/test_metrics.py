import math

import numpy as np
import pytest
import torch

from temporal_smoothing.metrics import (
    SegmentationMeter,
    edge_mask,
    expected_calibration_error,
    plot_calibration,
)

K = 5


# Reference: numpy port of ops/semantic_segmentation/tests.py from the official code.


def ref_cm(ys, probs, num_classes, filter_min=0.0, filter_max=1.0):
    result = probs.argmax(-1)
    conf = probs.max(-1)
    cond = (conf > filter_min) & (conf <= filter_max)
    k = (ys >= 0) & (ys < num_classes) & cond
    return np.bincount(num_classes * ys[k] + result[k], minlength=num_classes**2).reshape(num_classes, num_classes)


def ref_miou(cm):
    weights = (cm.sum(axis=1) > 0).astype(float)
    num = np.diag(cm)
    den = cm.sum(axis=1) + cm.sum(axis=0) - num
    ious = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)
    return np.average(ious, weights=weights) if weights.sum() > 0 else 0.0


def ref_gacc(cm):
    return np.diag(cm).sum() / cm.sum() if cm.sum() else 0.0


def ref_metrics(batches, num_classes, cutoffs, bins=np.linspace(0.0, 1.0, 11)):
    """Accumulate like `test()` in the official code. Inputs are channels-last numpy arrays."""
    nll_sum, nll_count = 0.0, 0
    cms = [[0, 0] for _ in cutoffs]
    cms_bin = [0] * (len(bins) - 1)
    conf_sum = np.zeros(len(bins) - 1)
    conf_count = np.zeros(len(bins) - 1)
    for ys, probs, mask in batches:
        ys, probs = ys[mask], probs[mask]
        # Keras SparseCategoricalCrossentropy on probabilities: clip, log, softmax-normalize.
        logp = np.log(np.clip(probs, 1e-7, 1 - 1e-7))
        logp = logp - np.log(np.exp(logp).sum(-1, keepdims=True))
        nll_sum += -logp[np.arange(len(ys)), ys].sum()
        nll_count += len(ys)
        for (cutoff, group) in zip(cutoffs, cms):
            group[0] = group[0] + ref_cm(ys, probs, num_classes, filter_min=cutoff)
            group[1] = group[1] + ref_cm(ys, probs, num_classes, filter_max=cutoff)
        conf = probs.max(-1)
        for i, (start, end) in enumerate(zip(bins, bins[1:])):
            cms_bin[i] = cms_bin[i] + ref_cm(ys, probs, num_classes, filter_min=start, filter_max=end)
            cond = (conf >= start) & (conf < end)
            conf_sum[i] += conf[cond].sum()
            conf_count[i] += cond.sum()

    count_bin = np.array([np.sum(cm) for cm in cms_bin])
    accs_bin = np.array([ref_gacc(cm) for cm in cms_bin])
    confs_bin = np.divide(conf_sum, conf_count, out=np.zeros_like(conf_sum), where=conf_count != 0)
    results = {
        "nll": nll_sum / nll_count,
        "ece": float(np.sum(np.abs(accs_bin - confs_bin) * count_bin / count_bin.sum())),
        "bin_count": count_bin,
        "bin_acc": accs_bin,
        "bin_conf": confs_bin,
    }
    for cutoff, (certain, uncertain) in zip(cutoffs, cms):
        suffix = f"_{round(cutoff * 100)}"
        inacc_c = certain.sum() - np.diag(certain).sum()
        inacc_u = uncertain.sum() - np.diag(uncertain).sum()
        results["acc" + suffix] = ref_gacc(certain)
        results["iou" + suffix] = ref_miou(certain)
        results["unc" + suffix] = inacc_u / (inacc_c + inacc_u)
        results["freq" + suffix] = certain.sum() / (certain.sum() + uncertain.sum())
    return results


def random_batches(n_batches=3, batch=2, size=(6, 7), seed=0):
    gen = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        logits = torch.randn(batch, K, *size, generator=gen) * 3
        probs = logits.softmax(dim=1)
        target = torch.randint(-1, K, (batch, *size), generator=gen)  # -1 = void
        mask = torch.rand(batch, *size, generator=gen) > 0.2
        batches.append((probs, target, mask))
    return batches


def to_numpy(batches):
    out = []
    for probs, target, mask in batches:
        valid = (mask & (target >= 0)).numpy()
        out.append((target.numpy(), probs.movedim(1, -1).numpy(), valid))
    return out


@pytest.mark.parametrize("legacy", [False, True])
def test_matches_official_reference(legacy):
    cutoffs = (0.0, 0.7, 0.9)
    batches = random_batches()
    meter = SegmentationMeter(K, cutoffs=cutoffs, legacy=legacy)
    for probs, target, mask in batches:
        meter.update(probs, target, mask)
    results, bins = meter.compute(), meter.bins()
    ref = ref_metrics(to_numpy(batches), K, cutoffs)

    assert results["nll"] == pytest.approx(ref["nll"], abs=1e-5)
    assert results["ece"] == pytest.approx(ref["ece"], abs=1e-6)
    np.testing.assert_array_equal(bins["count"].numpy(), ref["bin_count"])
    np.testing.assert_allclose(bins["acc"].numpy(), ref["bin_acc"])
    np.testing.assert_allclose(bins["conf"].numpy(), ref["bin_conf"], atol=1e-6)
    for key in ("acc", "iou", "unc", "freq"):
        for cutoff in cutoffs:
            name = f"{key}_{round(cutoff * 100)}"
            assert results[name] == pytest.approx(ref[name], abs=1e-9), name
    assert results["acc"] == pytest.approx(ref["acc_0"])
    assert results["iou"] == pytest.approx(ref["iou_0"])


def test_bin_boundaries_fixed_vs_legacy():
    # Pixel 1: confidence exactly 1.0 and correct. Pixel 2: confidence exactly 0.5 and wrong.
    probs = torch.tensor([[1.0, 0.0], [0.5, 0.5]]).t().reshape(1, 2, 2)
    target = torch.tensor([[0, 1]])

    fixed = SegmentationMeter(2)
    fixed.update(probs, target)
    bins = fixed.bins()
    assert bins["count"][9] == 1 and bins["conf"][9] == 1.0 and bins["acc"][9] == 1.0
    assert bins["count"][4] == 1 and bins["conf"][4] == 0.5 and bins["acc"][4] == 0.0
    assert fixed.compute()["ece"] == pytest.approx(0.25)

    legacy = SegmentationMeter(2, legacy=True)
    legacy.update(probs, target)
    # Confidence 1.0 drops out of the mean and 0.5 lands in the next bin.
    assert legacy.compute()["ece"] == pytest.approx(0.5)
    ref = ref_metrics([(target.numpy(), probs.movedim(1, -1).numpy(), np.ones((1, 2), bool))], 2, (0.9,))
    assert legacy.compute()["ece"] == pytest.approx(ref["ece"])


def test_hand_computed_example():
    # 4 pixels, 2 classes: (conf, correct) = (0.95, yes), (0.95, no), (0.6, no), (0.6, yes)
    p = torch.tensor([[0.95, 0.05], [0.95, 0.05], [0.6, 0.4], [0.4, 0.6]])
    probs = p.t().reshape(1, 2, 4)
    target = torch.tensor([[0, 1, 1, 1]])
    meter = SegmentationMeter(2, cutoffs=(0.9,))
    meter.update(probs, target)
    r = meter.compute()

    assert r["acc"] == pytest.approx(0.5)
    assert r["acc_90"] == pytest.approx(0.5)
    assert r["freq_90"] == pytest.approx(0.5)
    assert r["unc_90"] == pytest.approx(0.5)  # 1 of 2 errors is unconfident
    # Predictions [0, 0, 0, 1] for targets [0, 1, 1, 1]: both classes have IoU 1/3.
    assert r["class_iou"] == pytest.approx([1 / 3, 1 / 3])
    assert r["iou"] == pytest.approx(1 / 3)
    expected_nll = -(math.log(0.95) + math.log(0.05) + math.log(0.4) + math.log(0.6)) / 4
    assert r["nll"] == pytest.approx(expected_nll, rel=1e-6)


def test_mean_iou_ignores_absent_classes():
    probs = torch.tensor([[0.9, 0.05, 0.05]]).t().reshape(1, 3, 1)
    meter = SegmentationMeter(3)
    meter.update(probs, torch.tensor([[0]]))
    assert meter.compute()["iou"] == pytest.approx(1.0)


def test_void_and_mask_are_ignored():
    meter = SegmentationMeter(K)
    probs, target, _ = random_batches(n_batches=1)[0]
    meter.update(probs, torch.full_like(target, -1))
    meter.update(probs, target, torch.zeros_like(target, dtype=torch.bool))
    assert meter.count == 0
    assert meter.compute()["nll"] == 0.0


def test_reset():
    meter = SegmentationMeter(K)
    probs, target, mask = random_batches(n_batches=1)[0]
    meter.update(probs, target, mask)
    meter.reset()
    assert meter.count == 0


def test_shape_mismatch():
    meter = SegmentationMeter(K)
    with pytest.raises(ValueError):
        meter.update(torch.rand(2, K, 4, 4), torch.zeros(2, 4, 5, dtype=torch.long))
    with pytest.raises(ValueError):
        meter.update(torch.rand(2, K + 1, 4, 4), torch.zeros(2, 4, 4, dtype=torch.long))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu():
    cpu, gpu = SegmentationMeter(K, cutoffs=(0.7, 0.9)), SegmentationMeter(K, cutoffs=(0.7, 0.9))
    for probs, target, mask in random_batches():
        cpu.update(probs, target, mask)
        gpu.update(probs.cuda(), target.cuda(), mask.cuda())
    a, b = cpu.compute(), gpu.compute()
    for key in a:
        assert a[key] == pytest.approx(b[key], abs=1e-9), key


# Edge mask


def test_edge_mask():
    flat = torch.full((1, 3, 8, 8), 0.5)
    assert not edge_mask(flat, 0.1).any()

    step = torch.zeros(1, 3, 8, 8)
    step[..., 4:] = 1.0
    mask = edge_mask(step, 0.1)[0]
    assert mask[:, 3:5].all()
    assert not mask[:, :3].any() and not mask[:, 5:].any()


# Calibration


def test_expected_calibration_error():
    assert expected_calibration_error([1, 1], [1.0, 0.0], [0.9, 0.3]) == pytest.approx(0.2)
    assert expected_calibration_error([0, 0], [0.0, 0.0], [0.0, 0.0]) == 0.0


def test_plot_calibration():
    meter = SegmentationMeter(K)
    for probs, target, mask in random_batches():
        meter.update(probs, target, mask)
    fig = plot_calibration(meter.bins())
    assert len(fig.axes) == 2
