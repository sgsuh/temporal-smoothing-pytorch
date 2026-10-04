"""Building blocks shared by the segmentation networks.

Defaults follow the official TensorFlow (Keras) implementation when ``tf_compat=True``:
Glorot-uniform kernels with zero biases, and batch norm with ``epsilon=1e-3`` and
``momentum=0.99`` (``0.01`` in PyTorch's convention).
"""

from typing import Sequence, Tuple

from torch import Tensor, nn

__all__ = ["ConvBlock", "DeconvBlock", "conv_stack", "batch_norm", "tf_init_"]

TF_BN_EPS = 1e-3
TF_BN_MOMENTUM = 0.01


def batch_norm(channels: int, tf_compat: bool = True) -> nn.BatchNorm2d:
    if tf_compat:
        return nn.BatchNorm2d(channels, eps=TF_BN_EPS, momentum=TF_BN_MOMENTUM)
    return nn.BatchNorm2d(channels)


def tf_init_(model: nn.Module) -> nn.Module:
    """Initialize convolutions like Keras: Glorot-uniform kernels and zero biases."""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model


class ConvBlock(nn.Module):
    """3x3 same-padded convolution with batch norm and ReLU.

    Args:
        order: ``"conv-bn-relu"`` (SegNet) or ``"conv-relu-bn"`` (U-Net in the official code).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        order: str = "conv-bn-relu",
        tf_compat: bool = True,
    ) -> None:
        super().__init__()
        if order not in ("conv-bn-relu", "conv-relu-bn"):
            raise ValueError(f"unknown order: {order}")
        self.order = order
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = batch_norm(out_channels, tf_compat)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.order == "conv-bn-relu":
            return self.relu(self.bn(x))
        return self.bn(self.relu(x))


def conv_stack(
    in_channels: int,
    channels: Sequence[int],
    order: str = "conv-bn-relu",
    tf_compat: bool = True,
) -> nn.Sequential:
    layers = []
    for out_channels in channels:
        layers.append(ConvBlock(in_channels, out_channels, order, tf_compat))
        in_channels = out_channels
    return nn.Sequential(*layers)


def _tf_same_crop(in_size: int, out_size: int, kernel_size: int, stride: int) -> int:
    """Leading crop that aligns a full transposed convolution with TF ``padding='SAME'``.

    TF defines the transposed convolution as the adjoint of a ``SAME`` convolution from
    ``out_size`` to ``in_size``, which pads ``pad_total // 2`` before the input.
    """
    pad_total = max((in_size - 1) * stride + kernel_size - out_size, 0)
    return pad_total // 2


class DeconvBlock(nn.Module):
    """Stride-2 transposed convolution (no bias) followed by ReLU and batch norm.

    The output size is given explicitly (the size of the skip connection) and the result is
    aligned exactly like ``tf.nn.conv2d_transpose(..., padding='SAME')``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        tf_compat: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn = batch_norm(out_channels, tf_compat)

    def upsample(self, x: Tensor, output_size: Tuple[int, int]) -> Tensor:
        """Transposed convolution with TF ``SAME`` alignment (no activation)."""
        out_h, out_w = output_size
        in_h, in_w = x.shape[-2:]
        full_h = (in_h - 1) * self.stride + self.kernel_size
        full_w = (in_w - 1) * self.stride + self.kernel_size
        if out_h > full_h or out_w > full_w:
            raise ValueError(
                f"output size {tuple(output_size)} is too large for input size {(in_h, in_w)}"
            )
        x = self.deconv(x)
        top = _tf_same_crop(in_h, out_h, self.kernel_size, self.stride)
        left = _tf_same_crop(in_w, out_w, self.kernel_size, self.stride)
        return x[..., top : top + out_h, left : left + out_w]

    def forward(self, x: Tensor, output_size: Tuple[int, int]) -> Tensor:
        return self.bn(self.relu(self.upsample(x, output_size)))
