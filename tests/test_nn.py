import pytest
import torch
from torch import nn
from torch.nn import functional as F

from temporal_smoothing.nn import DeconvBlock, MCDropout, SegNet, UNet, set_mc_dropout

MODELS = [UNet, SegNet]


def tf_same_conv2d(y, weight, stride):
    """Reference `tf.nn.conv2d(..., padding='SAME')` (pads `pad_total // 2` before)."""
    pads = []
    for size, k in zip(reversed(y.shape[-2:]), reversed(weight.shape[-2:])):
        out = -(-size // stride)
        total = max((out - 1) * stride + k - size, 0)
        pads += [total // 2, total - total // 2]
    return F.conv2d(F.pad(y, pads), weight, stride=stride)


# MCDropout


def test_mc_dropout_active_in_eval():
    layer = MCDropout(0.5).eval()
    x = torch.ones(1000)
    a, b = layer(x), layer(x)
    assert not torch.equal(a, b)
    assert set(a.unique().tolist()) <= {0.0, 2.0}


def test_mc_dropout_zero_rate_is_identity():
    layer = MCDropout(0.0).train()
    x = torch.randn(10)
    assert layer(x) is x


def test_set_mc_dropout_restores_standard_dropout():
    model = set_mc_dropout(nn.Sequential(MCDropout(0.5)), enabled=False)
    x = torch.ones(1000)
    model.eval()
    assert torch.equal(model(x), x)
    model.train()
    assert not torch.equal(model(x), x)
    set_mc_dropout(model, enabled=True).eval()
    assert not torch.equal(model(x), x)


# DeconvBlock


@pytest.mark.parametrize("in_size,out_size", [((4, 5), (8, 10)), ((23, 30), (45, 60)), ((3, 4), (5, 8))])
def test_deconv_matches_tf_same_alignment(in_size, out_size):
    torch.manual_seed(0)
    block = DeconvBlock(3, 2).double()
    x = torch.randn(2, 3, *in_size, dtype=torch.float64)
    y = torch.randn(2, 2, *out_size, dtype=torch.float64)

    # TF's transposed convolution is the adjoint of the SAME convolution: <T x, y> = <x, C y>.
    lhs = (block.upsample(x, out_size) * y).sum()
    rhs = (x * tf_same_conv2d(y, block.deconv.weight, stride=2)).sum()
    torch.testing.assert_close(lhs, rhs)


def test_deconv_rejects_oversized_output():
    with pytest.raises(ValueError):
        DeconvBlock(3, 2).upsample(torch.randn(1, 3, 4, 4), (12, 12))


# UNet / SegNet


@pytest.mark.parametrize("model_cls", MODELS)
@pytest.mark.parametrize("size", [(32, 48), (45, 61)])
def test_output_shape(model_cls, size):
    model = model_cls(num_classes=11, rate=0.5).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, *size))
    assert out.shape == (2, 11, *size)


@pytest.mark.parametrize("model_cls", MODELS)
def test_has_six_mc_dropout_layers(model_cls):
    model = model_cls(num_classes=11, rate=0.5)
    dropouts = [m for m in model.modules() if isinstance(m, MCDropout)]
    assert len(dropouts) == 6
    assert all(m.p == 0.5 for m in dropouts)


@pytest.mark.parametrize("model_cls", MODELS)
def test_bnn_is_stochastic_and_dnn_deterministic_in_eval(model_cls):
    torch.manual_seed(0)
    x = torch.randn(1, 3, 32, 48)
    with torch.no_grad():
        bnn = model_cls(num_classes=5, rate=0.5).eval()
        assert not torch.allclose(bnn(x), bnn(x))
        dnn = model_cls(num_classes=5, rate=0.0).eval()
        assert torch.equal(dnn(x), dnn(x))


@pytest.mark.parametrize("model_cls", MODELS)
def test_tf_compat_initialization(model_cls):
    model = model_cls(num_classes=5)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert module.eps == 1e-3 and module.momentum == 0.01
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and module.bias is not None:
            assert torch.count_nonzero(module.bias) == 0

    default = model_cls(num_classes=5, tf_compat=False)
    bn = next(m for m in default.modules() if isinstance(m, nn.BatchNorm2d))
    assert bn.eps == 1e-5 and bn.momentum == 0.1


@pytest.mark.parametrize("model_cls", MODELS)
def test_backward(model_cls):
    model = model_cls(num_classes=5, rate=0.5).train()
    x = torch.randn(2, 3, 32, 48)
    target = torch.randint(0, 5, (2, 32, 48))
    F.cross_entropy(model(x), target).backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("model_cls", MODELS)
def test_camvid_resolution_on_cuda(model_cls):
    model = model_cls(num_classes=11, rate=0.5).cuda().eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 360, 480, device="cuda"))
    assert out.shape == (1, 11, 360, 480)
