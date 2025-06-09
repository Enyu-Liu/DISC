import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.config import Config
from mmdet.registry import MODELS

class TPVPooler(nn.Module):
    def __init__(
        self,
        embed_dims=128,
        split=[8,8,8],
        grid_size=[128, 128, 16],
    ):
        super().__init__()
        # self.pool_xy = nn.MaxPool3d(
        #     kernel_size=[1, 1, grid_size[2]//split[2]],
        #     stride=[1, 1, grid_size[2]//split[2]], padding=0
        # )

        self.pool_yz = nn.MaxPool3d(
            kernel_size=[grid_size[0]//split[0], 1, 1],
            stride=[grid_size[0]//split[0], 1, 1], padding=0
        )

        self.pool_zx = nn.MaxPool3d(
            kernel_size=[1, grid_size[1]//split[1], 1],
            stride=[1, grid_size[1]//split[1], 1], padding=0
        )

        in_channels = [int(embed_dims * s) for s in split]
        out_channels = [int(embed_dims) for s in split]

        # self.mlp_xy = nn.Sequential(
        #     nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1), 
        #     nn.ReLU(), 
        #     nn.Conv2d(out_channels[2], out_channels[2], kernel_size=1, stride=1))
        
        self.mlp_yz = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1), 
            nn.ReLU(), 
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=1, stride=1))
        
        self.mlp_zx = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1), 
            nn.ReLU(), 
            nn.Conv2d(out_channels[1], out_channels[1], kernel_size=1, stride=1))
    
    def forward(self, x):
        # tpv_xy = self.mlp_xy(self.pool_xy(x).permute(0, 4, 1, 2, 3).flatten(start_dim=1, end_dim=2))
        tpv_yz = self.mlp_yz(self.pool_yz(x).permute(0, 2, 1, 3, 4).flatten(start_dim=1, end_dim=2))
        tpv_zx = self.mlp_zx(self.pool_zx(x).permute(0, 3, 1, 2, 4).flatten(start_dim=1, end_dim=2))

        # tpv_list = [tpv_xy, tpv_yz, tpv_zx]
        tpv_list = [tpv_yz, tpv_zx]

        return tpv_list


class TPVGlobalAggregator(nn.Module):
    def __init__(
        self,
        embed_dims=128,
        split=[8,8,8],
        grid_size=[128, 128, 16],
        config_path='./configs/mmconfig/CGformer.py',
    ):
        super().__init__()
        mmconfig = Config.fromfile(config_path)
        bev_config = mmconfig.model.occ_encoder_backbone.global_aggregator
        self.global_encoder_backbone = MODELS.build(bev_config.global_encoder_backbone)
        MODELS_3d = importlib.import_module('mmdet3d' + '.registry').MODELS
        self.global_encoder_neck = MODELS_3d.build(bev_config.global_encoder_neck)
        # max pooling
        self.tpv_pooler = TPVPooler(
            embed_dims=embed_dims, split=split, grid_size=grid_size
        )

    
    def forward(self, x):
        """
        xy: [b, c, h, w, z] -> [b, c, h, w]
        yz: [b, c, h, w, z] -> [b, c, w, z]
        zx: [b, c, h, w, z] -> [b, c, h, z]
        """
        x_3view = self.tpv_pooler(x)

        tpv_list = []
        for x_tpv in x_3view:
            x_tpv = self.global_encoder_backbone(x_tpv)
            x_tpv = self.global_encoder_neck(x_tpv)
            if not isinstance(x_tpv, torch.Tensor):
                x_tpv = x_tpv[0]
            tpv_list.append(x_tpv)

        # tpv_list[0] = F.interpolate(tpv_list[0], size=(128, 128), mode='bilinear').unsqueeze(-1)
        tpv_list[0] = F.interpolate(tpv_list[0], size=(128, 16), mode='bilinear').unsqueeze(2)
        tpv_list[1] = F.interpolate(tpv_list[1], size=(128, 16), mode='bilinear').unsqueeze(3)

        return tpv_list




# Custom Encoder
class BEVSwinEncoder(nn.Module):
    def __init__(self, config_path='./configs/mmconfig/CGformer.py') -> None:
        super().__init__()
        mmconfig = Config.fromfile(config_path)
        bev_config = mmconfig.model.occ_encoder_backbone.global_aggregator
        self.global_encoder_backbone = MODELS.build(bev_config.global_encoder_backbone)
        MODELS_3d = importlib.import_module('mmdet3d' + '.registry').MODELS
        self.global_encoder_neck = MODELS_3d.build(bev_config.global_encoder_neck)
    
    def forward(self, xs):
        """
        Args:  
            xs : a list of ins and bk bev feature
        Return:
            A list of encodered bev feature
        """
        feat_list = []
        for x in xs:
            x = self.global_encoder_backbone(x)
            x = self.global_encoder_neck(x)
            if not isinstance(x, torch.Tensor):
                x = x[0]
            feat_list.append(x)
        
        feat_list[0] = F.interpolate(feat_list[0], size=(128, 128), mode='bilinear')
        feat_list[1] = F.interpolate(feat_list[1], size=(128, 128), mode='bilinear')
        return feat_list