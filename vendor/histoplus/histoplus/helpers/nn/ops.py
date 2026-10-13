"""Operation implementation with a torch fallback for xFormers SwiGLU."""

from typing import Callable

import torch
from torch import nn

try:
    from xformers.ops import SwiGLU as _SwiGLU
except ImportError:
    class _SwiGLU(nn.Module):
        """State-dict-compatible implementation of ``xformers.ops.SwiGLU``."""

        def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int,
            bias: bool = True,
        ) -> None:
            super().__init__()
            self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
            self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            hidden, gate = self.w12(inputs).chunk(2, dim=-1)
            return self.w3(torch.nn.functional.silu(hidden) * gate)


class SwiGLUFFNFused(_SwiGLU):
    """SwiGLUFFNFused as implemented in DINO v2 original paper."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] | None = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )
