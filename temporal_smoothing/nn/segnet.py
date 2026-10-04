"""SegNet as defined in the official VQ-BNN implementation (``models/segnet.py``).

Encoder-decoder with max-unpooling by the stored pooling indices. Conv blocks are ordered
conv -> batch norm -> ReLU. The encoder has four pooling stages (no pooling after block 5).
A Bayesian SegNet uses MC dropout after pooling stages 3-4 and after blocks 5-8.
"""

from torch import Tensor, nn

from .blocks import conv_stack, tf_init_
from .mc_dropout import MCDropout

__all__ = ["SegNet"]

_ORDER = "conv-bn-relu"


class SegNet(nn.Module):
    """SegNet for dense prediction.

    Args:
        num_classes: Number of output channels (logits).
        in_channels: Number of input channels.
        rate: MC dropout rate. ``0`` gives the deterministic network.
        tf_compat: Use Keras initialization and batch norm hyperparameters.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        rate: float = 0.0,
        tf_compat: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.rate = rate

        def stack(in_ch, channels):
            return conv_stack(in_ch, channels, _ORDER, tf_compat)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # Encoding
        self.enc1 = stack(in_channels, [64, 64])
        self.enc2 = stack(64, [128, 128])
        self.enc3 = stack(128, [256, 256, 256])
        self.drop3 = MCDropout(rate)
        self.enc4 = stack(256, [512, 512, 512])
        self.drop4 = MCDropout(rate)
        self.enc5 = stack(512, [512, 512, 512])
        self.drop5 = MCDropout(rate)

        # Decoding
        self.dec6 = stack(512, [512, 512, 512])
        self.drop6 = MCDropout(rate)
        self.dec7 = stack(512, [512, 512, 256])
        self.drop7 = MCDropout(rate)
        self.dec8 = stack(256, [256, 256, 128])
        self.drop8 = MCDropout(rate)
        self.dec9 = stack(128, [128, 64])
        self.dec10 = stack(64, [64])

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

        if tf_compat:
            tf_init_(self)

    def forward(self, x: Tensor) -> Tensor:
        # Encoding
        x = self.enc1(x)
        size1 = x.shape[-2:]
        x, idx1 = self.pool(x)
        x = self.enc2(x)
        size2 = x.shape[-2:]
        x, idx2 = self.pool(x)
        x = self.enc3(x)
        size3 = x.shape[-2:]
        x, idx3 = self.pool(x)
        x = self.drop3(x)
        x = self.enc4(x)
        size4 = x.shape[-2:]
        x, idx4 = self.pool(x)
        x = self.drop4(x)
        x = self.enc5(x)
        x = self.drop5(x)

        # Decoding
        x = self.dec6(x)
        x = self.drop6(x)
        x = self.dec7(self.unpool(x, idx4, output_size=size4))
        x = self.drop7(x)
        x = self.dec8(self.unpool(x, idx3, output_size=size3))
        x = self.drop8(x)
        x = self.dec9(self.unpool(x, idx2, output_size=size2))
        x = self.dec10(self.unpool(x, idx1, output_size=size1))

        return self.head(x)
