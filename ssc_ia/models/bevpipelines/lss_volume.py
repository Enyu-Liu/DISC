import torch
import torch.nn as nn

from mmengine.config import Config

from .GeometryDepth_Net import GeometryDepthNet
from .viewtransformer_lss import ViewTransformerLSS
from ..decoders.utils import _down_sample2D, _up_sample2D

class LssVolume(nn.Module):
    def __init__(self, image_shape, embed_dims=128, config_path='./configs/mmconfig/CGformer.py') -> None:
        super().__init__()

        mmconfig = Config.fromfile(config_path)

        # transformer net
        depthnet_cfg = mmconfig.model.depth_net
        lss_trans_cfg = mmconfig.model.img_view_transformer
        self.depth_net = GeometryDepthNet(**depthnet_cfg)
        self.view_transformer = ViewTransformerLSS(**lss_trans_cfg, input_size=image_shape)

        # feature aggretion
        self.downsample = _down_sample2D(1, embed_dims, embed_dims)
        self.upsample = _up_sample2D(1, embed_dims, embed_dims)
        self.pre_conv = nn.Conv2d(embed_dims*3, depthnet_cfg.numC_input, 1, 1, 0)
    
    def forward(self, feats, inputs):
        """Generate 3D volume feature using lss method
        Args:
            feats: a list of img feature
            inputs: camera parameters
        Returns:
            volume:
            depth_feat:
            mono_depth_pred:
        """
        # TODO : Second FPN format feature concate
        x0 = self.downsample(feats[0])
        x1 = feats[1]
        x2 = self.upsample(feats[2])
        feat = torch.cat([x0, x1, x2], dim=1)
        feat = self.pre_conv(feat) #(bs, 640, H//8, W//8)

        # depth net and view transformer
        lidar2cam = inputs['cam_pose']
        cam2lidar = lidar2cam.inverse()      
        rots = cam2lidar[:, :3, :3].unsqueeze(1) #(bs, 1, 3, 3)
        trans = cam2lidar[:, :3, 3].unsqueeze(1) #(bs, 1, 3)
        intrins, post_rots, post_trans = [inputs['cam_K'].unsqueeze(1), inputs['post_rot'].unsqueeze(1),\
                                           inputs['post_tran'].unsqueeze(1)] #[(bs, 1, 3, 3), (bs, 1, 3, 3), (bs, 1, 3)]
        mlp_input = self.depth_net.get_mlp_input(rots, trans, intrins, post_rots, post_trans) #(bs, 1, 27)
        geo_inputs = [rots, trans, intrins, post_rots, post_trans, mlp_input]   
        context, depth, depth_feat = self.depth_net([feat.unsqueeze(1)] + geo_inputs, inputs['depth'].unsqueeze(1)) #(1, 1, 128, 48, 160) (1, 112, 48, 160)
        view_trans_inputs = [rots[:, 0:1, ...], trans[:, 0:1, ...], intrins[:, 0:1, ...], post_rots[:, 0:1, ...], post_trans[:, 0:1, ...]]
        volume = self.view_transformer(context, depth, view_trans_inputs) # (1, 128, 128, 128, 16)

        return volume, depth_feat, depth