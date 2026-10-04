"""Monte Carlo dropout."""

from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["MCDropout", "set_mc_dropout"]


class MCDropout(nn.Dropout):
    """Dropout that stays active in eval mode, turning a network into an MC dropout BNN.

    Calling ``model.eval()`` freezes batch norm statistics while every forward pass still
    samples a new dropout mask, i.e. new weights ``w ~ q(w)``. With ``p=0`` the layer is the
    identity, so the same architecture also represents the deterministic network.

    Set ``mc=False`` (or use :func:`set_mc_dropout`) to fall back to standard dropout that
    follows ``module.training``.
    """

    def __init__(self, p: float = 0.5, mc: bool = True) -> None:
        super().__init__(p=p)
        self.mc = mc

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0.0:
            return x
        return F.dropout(x, self.p, training=self.training or self.mc)

    def extra_repr(self) -> str:
        return f"p={self.p}, mc={self.mc}"


def set_mc_dropout(model: nn.Module, enabled: bool) -> nn.Module:
    """Enable or disable MC sampling for every :class:`MCDropout` in ``model``."""
    for module in model.modules():
        if isinstance(module, MCDropout):
            module.mc = enabled
    return model
