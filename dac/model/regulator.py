from typing import Tuple

import torch.nn as nn
from torch.nn import functional as F

from .utils import make_pad_mask


class InterpolateRegulator(nn.Module):
    def __init__(
        self,
        sampling_ratios: Tuple,
        in_channels: int,
        channels: int,
        out_channels: int,
        groups: int = 1,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for i, _ in enumerate(sampling_ratios):
                module = nn.Conv1d(
                    in_channels if i == 0 else channels, channels, 3, 1, 1
                )
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
            model.append(nn.Conv1d(channels, out_channels, 1, 1))  # 这个是改channel的维度
        elif len(sampling_ratios) == 0:
            model.append(nn.Conv1d(in_channels, out_channels, 1))
        self.model = nn.Sequential(*model)

    def forward(self, x, xlens=None, ylens=None):
        # x in (B, T, D)
        mask = ~make_pad_mask(lengths=xlens, max_len=ylens.max()).unsqueeze(-1)
        # mask = (~make_pad_mask(ylens)).to(x).unsqueeze(-1) # torch.Size([16, 150, 1])
        x = x.transpose(1, 2).contiguous()  # (B, T, D) -> (B, D, T)
        x = F.interpolate(x, size=ylens.max(), mode="linear")
        # print(f"x.shape{x.shape}")
        # print(self.model)
        out = self.model(x).transpose(1, 2).contiguous()
        olens = ylens
        # print(f"out {out.shape} | mask {mask.shape} | olens {olens.shape}")
        return out * mask, olens
