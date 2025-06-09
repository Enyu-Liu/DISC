import torch
import torch.nn as nn

from mmengine.config import Config
from mmseg.registry import MODELS

class Unet2dWrapper(nn.Module):
    def __init__(self, out_channel, config_path) -> None:
        super().__init__()
        config = Config.fromfile(config_path)
        unet_config = config.model.backbone
        unet_config.in_channels = out_channel
        unet_config._scope_='mmseg'      # change current scope to mmseg

        self.unet2d = MODELS.build(unet_config)
        self.outconv = nn.Sequential(
            nn.Conv2d(unet_config.base_channels, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.unet2d(x)[-1]
        x = self.outconv(x)
        return x