from mmseg.registry import MODELS
from mmengine.model import BaseModule
from torch import nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_


@MODELS.register_module()
class VegAdapter(BaseModule):
    def __init__(self, in_channels, init_cfg=None):
        super().__init__(init_cfg)
        self.adapters = nn.ModuleList()

        for in_channel in in_channels:
            self.adapters.append(nn.Linear(in_channel, in_channel))

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
