import torch
import torch.nn as nn
import torch.nn.functional as F

from . import encoders
from ..decoders.transformer_detr import DeformableTransformerLayer
from ..decoders.pos_embed import LearnableSqueezePositionalEncoding
from ..utils import (flatten_multi_scale_feats, get_level_start_index, index_fov_back_to_voxels,
                     nlc_to_nchw, cumprod, pix2vox, generate_grid)
from ...utils import build_from_configs

class VoxelProposalLayer(nn.Module):

    def __init__(self, embed_dims, scene_shape, num_heads=8, num_levels=3, num_points=4):
        super().__init__()
        self.attn = DeformableTransformerLayer(embed_dims, num_heads, num_levels, num_points)
        self.scene_shape = scene_shape

    def forward(self, scene_embed, feats, scene_pos=None, vol_pts=None, ref_pix=None):
        keep = ((vol_pts[..., 0] >= 0) & (vol_pts[..., 0] < self.scene_shape[0]) &
                (vol_pts[..., 1] >= 0) & (vol_pts[..., 1] < self.scene_shape[1]) &
                (vol_pts[..., 2] >= 0) & (vol_pts[..., 2] < self.scene_shape[2]))
        
        scene_embed_list = []
        for i in range(vol_pts.shape[0]):
            # assert vol_pts.shape[0] == 1
            geom = vol_pts[i].squeeze()[keep[i].squeeze()] #[91158, 3]

            pts_mask = torch.zeros(self.scene_shape, device=scene_embed.device, dtype=torch.bool) #[64, 64, 4]
            pts_mask[geom[:, 0], geom[:, 1], geom[:, 2]] = True
            pts_mask = pts_mask.flatten() # [16384]

            feat_flatten, shapes = flatten_multi_scale_feats([feat[i].unsqueeze(0) for feat in feats]) #[1, 9507, 128] [3, 2]
            pts_embed = self.attn(
                scene_embed[i][pts_mask].unsqueeze(0),
                feat_flatten,
                query_pos=scene_pos[i][pts_mask].unsqueeze(0) if scene_pos is not None else None,
                ref_pts=ref_pix[i][pts_mask].unsqueeze(0).unsqueeze(2).expand(-1, -1, len(feats), -1),
                spatial_shapes=shapes,
                level_start_index=get_level_start_index(shapes)) #[1, 3510, 128]
            scene_embed_list.append(index_fov_back_to_voxels
                                    (nlc_to_nchw(scene_embed[i].unsqueeze(0), self.scene_shape), pts_embed, pts_mask))
        return torch.cat(scene_embed_list, 0)
    

class BEVPooler(nn.Module):
    def __init__(
        self,
        embed_dims=128,
        split=[8,8,8],
        grid_size=[128, 128, 16],
    ):
        super().__init__()
        self.pool_xy = nn.MaxPool3d(
            kernel_size=[1, 1, grid_size[2]//split[2]],
            stride=[1, 1, grid_size[2]//split[2]], padding=0
        )

        in_channels = [int(embed_dims * s) for s in split]
        out_channels = [int(embed_dims) for s in split]

        self.mlp_xy = nn.Sequential(
            nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1), 
            nn.ReLU(), 
            nn.Conv2d(out_channels[2], out_channels[2], kernel_size=1, stride=1))
    
    def forward(self, x):
        bev_feature = self.mlp_xy(self.pool_xy(x).permute(0, 4, 1, 2, 3).flatten(start_dim=1, end_dim=2))

        return bev_feature

class BEVFeature(nn.Module):
    def __init__(self, 
                 in_channels, 
                 encoder, 
                 num_cls, 
                 embed_dims, 
                 voxel_size,
                 scene_shape, 
                 image_shape,
                 bev_seg=False) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.scene_shape = scene_shape
        self.voxel_size = voxel_size
        image_grid = generate_grid(image_shape)
        image_grid = torch.flip(image_grid, dims=[0]).unsqueeze(0)  # 2(wh), h, w
        self.register_buffer('image_grid', image_grid)

        self.voxel_proposal = VoxelProposalLayer(embed_dims, scene_shape)

        self.scene_pos = LearnableSqueezePositionalEncoding((self.scene_shape[0], self.scene_shape[1], 
                                                             self.scene_shape[2]//4),
                                                            embed_dims,
                                                            squeeze_dims=(2, 2, 1))
        self.bev_pooler = BEVPooler(in_channels)
        # self.bev_encoder = build_from_configs(encoders, encoder)

        # seg head for all cls
        self.bev_seg = nn.Conv2d(in_channels, num_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, volume, feats, depth, K, E, voxel_origin, post_rot, post_tran, projected_pix):
        # encoder with unet
        B = feats[0].shape[0]
        vol_pts = pix2vox(self.image_grid, depth.unsqueeze(1), K, E, \
                          voxel_origin, self.voxel_size, post_rot, post_tran).long()
        ref_pix = (torch.flip(projected_pix, dims=[-1]) + 0.5) / torch.tensor(
            self.image_shape).to(projected_pix)
        ref_pix = torch.flip(ref_pix, dims=[-1])
        scene_embed = volume.flatten(2).transpose(1, 2).contiguous() #(bs, 262144, 128)
        scene_pos = self.scene_pos().repeat(B, 1, 1)
        volume = self.voxel_proposal(scene_embed, feats, scene_pos, vol_pts, ref_pix) #(bs, C, L, W, H)

        bev_f = self.bev_pooler(volume)
        # bev_f = self.bev_encoder(bev_f)  #(bs, 128, 128, 128)

        bev_seg_logits = self.bev_seg(bev_f)

        return volume, bev_f, bev_seg_logits