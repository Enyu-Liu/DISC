import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):

    def __init__(self, channels, dilations):
        super().__init__()
        self.blks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm3d(channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels, channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm3d(channels),
            ) for d in dilations
        ])

    def forward(self, x):
        outs = [x]
        for blk in self.blks:
            outs.append(blk(x))
        outs = torch.stack(outs).sum(dim=0)
        return F.relu_(outs)



class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm3d):
        super().__init__()
        self.up_bn = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            norm_layer(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.up_bn(x)