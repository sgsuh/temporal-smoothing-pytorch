"""U-Net as defined in the official VQ-BNN implementation (``models/unet.py``).

Differences from the original U-Net (Ronneberger et al., 2015), kept for fidelity:
conv blocks are ordered conv -> ReLU -> batch norm, skip connections are added (not
concatenated), the encoder follows VGG-16 widths with a 1024-channel bottleneck, and the
network is trained from scratch. A Bayesian U-Net uses MC dropout after encoder blocks 3-5
and decoder blocks 6-8.
"""

from torch import Tensor, nn

from .blocks import DeconvBlock, conv_stack, tf_init_
from .mc_dropout import MCDropout

__all__ = ["UNet"]

_ORDER = "conv-relu-bn"


class UNet(nn.Module):
    """U-Net for dense prediction.

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

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # Encoding
        self.enc1 = stack(in_channels, [64, 64])
        self.enc2 = stack(64, [128, 128])
        self.enc3 = stack(128, [256, 256, 256])
        self.drop3 = MCDropout(rate)
        self.enc4 = stack(256, [512, 512, 512])
        self.drop4 = MCDropout(rate)
        self.enc5 = stack(512, [1024, 1024, 1024])
        self.drop5 = MCDropout(rate)

        # Decoding
        self.up6 = DeconvBlock(1024, 512, tf_compat=tf_compat)
        self.dec6 = stack(512, [512, 512, 512])
        self.drop6 = MCDropout(rate)
        self.up7 = DeconvBlock(512, 256, tf_compat=tf_compat)
        self.dec7 = stack(256, [256, 256, 256])
        self.drop7 = MCDropout(rate)
        self.up8 = DeconvBlock(256, 128, tf_compat=tf_compat)
        self.dec8 = stack(128, [128, 128, 128])
        self.drop8 = MCDropout(rate)
        self.up9 = DeconvBlock(128, 64, tf_compat=tf_compat)
        self.dec9 = stack(64, [64, 64])

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

        if tf_compat:
            tf_init_(self)

    def forward(self, x: Tensor) -> Tensor:
        # Encoding
        x = self.enc1(x)
        skip1 = x
        x = self.enc2(self.pool(x))
        skip2 = x
        x = self.enc3(self.pool(x))
        skip3 = x
        x = self.drop3(x)
        x = self.enc4(self.pool(x))
        skip4 = x
        x = self.drop4(x)
        x = self.enc5(self.pool(x))
        x = self.drop5(x)

        # Decoding
        x = self.dec6(self.up6(x, skip4.shape[-2:]) + skip4)
        x = self.drop6(x)
        x = self.dec7(self.up7(x, skip3.shape[-2:]) + skip3)
        x = self.drop7(x)
        x = self.dec8(self.up8(x, skip2.shape[-2:]) + skip2)
        x = self.drop8(x)
        x = self.dec9(self.up9(x, skip1.shape[-2:]) + skip1)

        return self.head(x)
