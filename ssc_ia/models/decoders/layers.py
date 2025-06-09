import torch
import torch.nn as nn
import torch.nn.functional as F

from .pos_embed import pos2posemb2d
from .utils import select_ins_points, _make_stack_3x3_convs,  _up_sample2D, _down_sample2D
from ..utils import flatten_multi_scale_feats, get_level_start_index
from .transformer_detr import DeformableTransformerLayer, TransformerLayer, FFN

from ..bevpipelines.encoders import SparseUNet

class InsLayer(nn.Module):
    def __init__(self, embed_dims, ins_cls, scene_shape, use_e=False, \
                 num_points=5, num_heads=8, num_attn_p=8) -> None:
        super().__init__()
        self.scene_shape = scene_shape
        self.num_points = num_points

        L, W, H = scene_shape
        self.h_weight_ins = nn.Conv2d(embed_dims, H, 1, 1, 0)
        self.bev_seg_ins = nn.Conv2d(embed_dims, ins_cls, kernel_size=1, stride=1, padding=0)

        self.ins_self_attention = TransformerLayer(embed_dims, num_heads)
        self.feats_cross_attention = DeformableTransformerLayer(embed_dims, num_heads, 3, num_attn_p)
        self.bk_cross_attention = DeformableTransformerLayer(embed_dims, num_heads, 1, num_attn_p)
        self.FFN = FFN(embed_dims)
        self.conv_ins = _make_stack_3x3_convs(1, embed_dims, embed_dims)

        self.query_embedding = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims))
        
        # Pos embedding (2D pos)
        self.pos_embedding = nn.Sequential(
            nn.Linear(embed_dims*2, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
        )
        if use_e:
            self.bev_e_ins = SparseUNet(embed_dims)
    
    
    def forward(self, ins_feats, bev_f_ins, bev_f_bk, params, bev_ins_mask):
        """Instance Layer pipeline
        Args: insbev
        Return: insbev
        """
        B, C, L, W = bev_f_bk.shape
        # select ins point  # TODO : Change seg_logits_ins in pretrain model
        weight_h_ins = self.h_weight_ins(bev_f_ins).permute(0, 2, 3, 1).contiguous() #(bs, L, W, H)
        seg_logits_ins = self.bev_seg_ins(bev_f_ins) #(bs, num_ins, L, W)
        mask_ins = bev_ins_mask #(bs, L, W)
        ref3ds, ref2ds, mask2ds, hs_w, qs_w = \
            select_ins_points(mask_ins, weight_h_ins.sigmoid(), params, self.scene_shape, num_points=self.num_points)
        query = bev_f_ins.squeeze(0)[..., ref3ds[:, 0], ref3ds[:, 1]].transpose(1, 0).contiguous() #(N*num_points, C)
        query = self.query_embedding(query) #(N*num_points, C)
        # img-cross attention
        ref2ds = torch.flip(ref2ds.unsqueeze(0), dims=[-1]).unsqueeze(2) #(bs, M, 1, 2)
        query_ref = query[mask2ds].unsqueeze(0) #(bs, M, C)
        feats_flatten, feat_shapes = flatten_multi_scale_feats(ins_feats) 
        feats_level_index = get_level_start_index(feat_shapes)
        ref2ds = ref2ds.expand(-1, -1, len(ins_feats), -1) # (bs, M, num_f, 2)
        query_ref = self.feats_cross_attention(
            query_ref,
            feats_flatten,
            query_pos=None,
            ref_pts=ref2ds,
            spatial_shapes=feat_shapes,
            level_start_index=feats_level_index)  #(bs, M, C)

        
        query_t = torch.clone(query)
        query_t[mask2ds] = query_ref.squeeze(0)
        query = query_t  #(N*num_points, C)
        query = (query * hs_w).unsqueeze(0) #(bs, N*num_points, C)
        query = query.reshape(B, -1, self.num_points, C) #(bs, N, num_points, C)
        # query = torch.sum(query, dim=2) / self.num_points  #(bs, N, C)
        query = torch.sum(query, dim=2) / torch.sum(hs_w.reshape(-1, self.num_points), dim=-1).unsqueeze(-1)

        # bk-cross attention
        ref3ds = ref3ds.reshape(-1, self.num_points, 3)[:, 0, :] #(N, 3)
        refbevs = ref3ds[:, :2].reshape(1, ref3ds.shape[0], 1, 2) #(bs, N, 1, 2)
        feats_flatten, feat_shapes = flatten_multi_scale_feats([bev_f_bk])
        feats_level_index = get_level_start_index(feat_shapes)
        query = self.bk_cross_attention(
            query,
            feats_flatten,
            query_pos=None,
            ref_pts=refbevs,
            spatial_shapes=feat_shapes,
            level_start_index=feats_level_index
        ) #(bs, N, C)

        # self attention
        ref3ds_norms = (ref3ds / (torch.tensor(self.scene_shape) - 1).to(weight_h_ins)) #(N, 3)
        query_pos = self.pos_embedding(pos2posemb2d(ref3ds_norms[:, :2]).unsqueeze(0)) #(bs, N, C)

        query = self.ins_self_attention(query, query_pos=query_pos) #(bs, N, C)
        query = self.FFN(query) #(bs, N, C)

        # ref3ds = ref3ds.reshape(-1, self.num_points, 3)[:, 0, :] #(N, 3)
        bev_f_ins_t = torch.clone(bev_f_ins).squeeze(0).permute(1, 2, 0) #(L, W, C)
        bev_f_ins_t[ref3ds[:, 0], ref3ds[:, 1], :] = query.squeeze(0)*qs_w.unsqueeze(-1) +\
                bev_f_ins_t[ref3ds[:, 0], ref3ds[:, 1], :]*(1-qs_w).unsqueeze(-1)#(L, W, C)
        bev_f_ins = bev_f_ins_t.permute(2, 0, 1).contiguous().unsqueeze(0) #(bs, C, L, W)
        
        if hasattr(self, 'bev_e_ins'):   # Last Layer
            bev_f_ins = self.bev_e_ins(bev_f_ins)
            seg_logits_ins_tail = self.bev_seg_ins(bev_f_ins)
            weight_h_ins_tail = self.h_weight_ins(bev_f_ins).permute(0, 2, 3, 1).contiguous()
            return bev_f_ins, [seg_logits_ins, seg_logits_ins_tail], [weight_h_ins, weight_h_ins_tail]
        else:
            bev_f_ins = self.conv_ins(bev_f_ins)
            return bev_f_ins, [seg_logits_ins], [weight_h_ins]


class BkLayer(nn.Module):
    def __init__(self, embed_dims, scene_shape, bk_cls, num_heads=8, use_e=False) -> None:
        super().__init__()
        L, W, H = scene_shape
        # BK feature enhance
        self.h_weight_bk = nn.Conv2d(embed_dims, H, 1, 1, 0)
        self.bev_seg_bk = nn.Conv2d(embed_dims, bk_cls, kernel_size=1, stride=1, padding=0)
        self.down_bk = _down_sample2D(2, embed_dims, embed_dims)
        self.up_bk = _up_sample2D(2, embed_dims, embed_dims)
        self.bk_img_cross_attention = TransformerLayer(embed_dims, num_heads, mlp_ratio=0)
        self.bk_self_attention = TransformerLayer(embed_dims, num_heads)
        self.bk_FFN = FFN(embed_dims)
        self.conv_bk = _make_stack_3x3_convs(1, embed_dims, embed_dims)

        # Pos embedding (2D pos)
        self.pos_embedding = nn.Sequential(
            nn.Linear(embed_dims*2, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
        )

        if use_e:
            self.bev_e_bk = SparseUNet(embed_dims)

    def forward(self, bev_f_bk, ins_feats, bk_grid, img_grid):
        B, C, L, W = bev_f_bk.shape
        bev_f_bk = self.down_bk(bev_f_bk) #(bs, C, L//4, W//4)
        query_bk = bev_f_bk.reshape(B, C, -1).transpose(2, 1) #(bs, N, C)
        query_img = ins_feats[-1].flatten(2).transpose(2, 1) #(bs, K, C) 
        pos_bk = self.pos_embedding(pos2posemb2d(bk_grid.squeeze(0)).unsqueeze(0)) #(bs, N, C)
        pos_img = self.pos_embedding(pos2posemb2d(img_grid.squeeze(0)).unsqueeze(0)) #(bs, K, C)
        
        # random mask
        indices = torch.randperm(query_img.shape[1])[:int(query_img.shape[1]*0.8)]
        if self.training:
            query_img = query_img[:, indices, :]
            pos_img = pos_img[:, indices, :]
        # Bk-img Cross Attention
        query_bk = self.bk_img_cross_attention(query_bk, query_img, query_img, pos_bk, pos_img)
        # Bk-self Attention
        query_bk = self.bk_self_attention(query_bk, query_pos=pos_bk)
        query_bk = self.bk_FFN(query_bk) #(bs, N, C)
        bev_f_bk = query_bk.transpose(2, 1).reshape(B, C, bev_f_bk.shape[-2], bev_f_bk.shape[-2])
        bev_f_bk = self.up_bk(bev_f_bk) #(bs, C, L, W)

        if hasattr(self, 'bev_e_ins'):  # TODO: BUG Here
            bev_f_bk = self.bev_e_bk(bev_f_bk)
        else:
            bev_f_bk = self.conv_bk(bev_f_bk)

        weight_h_bk = self.h_weight_bk(bev_f_bk).permute(0, 2, 3, 1) #(bs, L, W, H)
        seg_logits_bk = self.bev_seg_bk(bev_f_bk)
        return bev_f_bk, seg_logits_bk, weight_h_bk