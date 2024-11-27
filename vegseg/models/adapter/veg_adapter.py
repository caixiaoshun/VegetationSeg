from mmseg.registry import MODELS
from mmengine.model import BaseModule
from torch import nn as nn
from torch.nn import functional as F
from typing import Callable, Optional
from torch import Tensor
from timm.models.layers import trunc_normal_


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@MODELS.register_module()
class VegAdapter(BaseModule):
    def __init__(self, in_channels, rank_dim=4, init_cfg=None):
        super().__init__(init_cfg)
        self.adapters = nn.ModuleList()

        for in_channel in in_channels:
            self.adapters.append(
                Mlp(
                    in_channel,
                    hidden_features=in_channel // rank_dim,
                    out_features=in_channel,
                )
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        outs = []
        for index, x in enumerate(inputs):
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(B, -1, C)
            x = self.adapters[index](x)
            x = x.reshape(B, H, W, C)
            x = x.permute(0, 3, 1, 2)
            outs.append(x)
        return tuple(outs)
