import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS

from .layers import InsLayer, BkLayer
from ..encoders import LocalAggregator, TPVGlobalAggregator
from .utils import _make_stack_3x3_convs, generate_grid_bev, ins_mask_projected
from ..modules import Upsample
from ..utils import pix2vox, generate_grid


class IASceneDecoder(nn.Module):
    def __init__(self, 
                 embed_dims, 
                 bk_cls, 
                 ins_cls, 
                 num_cls, 
                 scene_shape, 
                 feats_shape,
                 project_scale, 
                 ins_layers=3,
                 bk_layers=1) -> None:
        super().__init__()
        L, W, H = scene_shape
        self.H = H
        self.scene_shape = scene_shape
        self.ins_num_layers = ins_layers
        self.bk_num_layers = bk_layers
        self.feat_scale = 384 // feats_shape[0] # NOTE : TEMP

        # Select bk and img points
        bk_grid = generate_grid_bev(L//4, W//4, interval=1)
        bk_grid = (bk_grid / (torch.tensor([L//4, W//4]) - 1.))
        self.register_buffer('bk_grid', bk_grid) #(N, 2)
        img_grid = generate_grid(feats_shape)
        img_grid = torch.flip(img_grid, dims=[0]).unsqueeze(0)
        self.register_buffer('img_grid', img_grid)

        img_grid_up = generate_grid([l*4 for l in feats_shape]) # TODO: Temp Default up 4 
        img_grid_up = torch.flip(img_grid_up, dims=[0]).unsqueeze(0)
        self.register_buffer('img_grid_up', img_grid_up)

        # Decompose feat
        self.bev_f_bk = _make_stack_3x3_convs(2, embed_dims, embed_dims)
        self.bev_f_ins = _make_stack_3x3_convs(2, embed_dims, embed_dims)

        # Tail Encoder
        self.volume_e = LocalAggregator()
        # CGFormer Gloable TPV encoder
        self.global_tpv_e = TPVGlobalAggregator() # NOTE: Only encoder front view and right view here
        self.combine_coeff = nn.Sequential(
            nn.Conv3d(embed_dims, 3, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )

        # Build Layers
        self.ins_layers = nn.ModuleList()
        self.seg_img_conv = nn.Sequential(_make_stack_3x3_convs(1, embed_dims, embed_dims//2), 
                                          nn.Conv2d(embed_dims//2, out_channels=1, kernel_size=1))
        for i in range(ins_layers):
            use_e = True if i == ins_layers-1 else False
            self.ins_layer = InsLayer(embed_dims, ins_cls, scene_shape, use_e=use_e)
            self.ins_layers.append(self.ins_layer)
        
        self.bk_layers = nn.ModuleList()
        for i in range(bk_layers):
            use_e = True if i == bk_layers-1 else False
            self.bk_layer = BkLayer(embed_dims, scene_shape, bk_cls, use_e=use_e)
            self.bk_layers.append(self.bk_layer)
        
        # cls_head
        self.cls_head = nn.Sequential(
            Upsample(embed_dims, embed_dims) if project_scale == 2 else nn.Identity(),
            nn.Conv3d(embed_dims, num_cls, kernel_size=1))


    def forward(self, feats, volume, bev_f, depth, params):
        """IASceneDecoder
            * Decompose bkfeat and insfeat
            * Enhance feats
            * Generate volume
            * Cls each voxel
        Args:
            volume : Original volume , temp for test
            bev_f : BEV feature generate by decoder
        """    
        # Decompose bev feature into Bk_feat and Ins_feat
        bev_f_ins = self.bev_f_ins(bev_f)
        bev_f_bk = self.bev_f_bk(bev_f)
        
        # Generate img Coordinate in BEV
        depth = F.interpolate(depth.unsqueeze(1), size=feats[-1].shape[-2:], mode='bilinear')
        vol_pts = pix2vox(self.img_grid*self.feat_scale, depth, *params[:4], *params[-2:]) #(bs, N, 3)
        vol_pts = (vol_pts / (torch.tensor(self.scene_shape).to(vol_pts) - 1))

        # For Instance projection
        depth = F.interpolate(depth, size=feats[0].shape[-2:], mode='bilinear')
        vol_pts_up = pix2vox(self.img_grid_up*self.feat_scale//4, depth, *params[:4], *params[-2:]) #(bs, N*4, 3)
        # vol_pts_up = (vol_pts_up / (torch.tensor(self.scene_shape).to(vol_pts_up) - 1))
        
        # Decoder layer
        ssc_outs = []
        seg_img = []
        seg_ins, seg_bk = [], []
        h_ins, h_bk = [], []
        # BackGround Layer
        bk_seg_single, bk_h_single = [], []
        for i in range(self.bk_num_layers):
            bev_f_bk, seg_logits_bk, weight_h_bk = self.bk_layers[i](bev_f_bk, feats, self.bk_grid.unsqueeze(0), vol_pts[..., :2])
            bk_seg_single.append(seg_logits_bk)
            bk_h_single.append(weight_h_bk)
        seg_bk.append(bk_seg_single)
        h_bk.append(bk_h_single)
        
        # Instance iterative Layer
        # TODO: Generate Ins query using front Image
        seg_img_logits = self.seg_img_conv(feats[0]) #(bs, 1, img_h, img_w)
        bev_ins_mask = ins_mask_projected(seg_img_logits.sigmoid(), vol_pts_up, self.scene_shape) #(bs, L, W)
        ins_seg_single, ins_h_single = [], []
        for i in range(self.ins_num_layers):
            bev_f_ins, seg_logits_ins, weight_h_ins = self.ins_layers[i](feats, bev_f_ins, bev_f_bk, params, bev_ins_mask)
            ins_seg_single.extend(seg_logits_ins)
            ins_h_single.extend(weight_h_ins)
        seg_img.append(seg_img_logits)
        seg_ins.append(ins_seg_single)
        h_ins.append(ins_h_single)
            
        bk_volume = bev_f_bk.unsqueeze(4).repeat(1, 1, 1, 1, self.H) * bk_h_single[-1].unsqueeze(1).sigmoid()
        ins_volume = bev_f_ins.unsqueeze(4).repeat(1, 1, 1, 1, self.H) * ins_h_single[-1].unsqueeze(1).sigmoid()
        volume = bk_volume + ins_volume + volume
        
        # Local and Gloable Backbone
        local_feats = self.volume_e(volume)
        volume = local_feats   # NOTE : For Simple-Version(16.19)
        global_feats = self.global_tpv_e(volume)
        weights = self.combine_coeff(local_feats)
        volume = local_feats * weights[:, 0:1, ...] + global_feats[0] * weights[:, 1:2, ...] + \
            global_feats[1] * weights[:, 2:3, ...]
        
        ssc_outs.append(self.cls_head(volume))
        
        return {'ssc_logits': ssc_outs, 'bev_seg_ins':seg_ins, 'bev_seg_bk': seg_bk, 'seg_img': seg_img, 'h_ins':h_ins, 'h_bk':h_bk, \
                'bev_ins_mask': bev_ins_mask, 'params':params}  # For Vis