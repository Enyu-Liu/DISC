import torch.nn as nn
from mmcv.ops import MultiScaleDeformableAttention


class TransformerLayer(nn.Module):

    def __init__(self, embed_dims, num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dims = embed_dims
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=0.1, bias=qkv_bias, batch_first=True)
        self.norm = norm_layer(embed_dims)

    def forward(self, query, key=None, value=None, query_pos=None, key_pos=None):
        if key is None and value is None:
            key = value = query
            key_pos = query_pos
        if key_pos is not None:
            key = key + key_pos
        if query_pos is not None:
            query = query + self.attn(query + query_pos, key, value)[0]
        else:
            query = query + self.attn(query, key, value)[0]
        query = self.norm(query)
        return query


class DeformableTransformerLayer(nn.Module):
    """For multiscaleDeformableAttention
    """

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 num_levels=3,
                 num_points=4,
                 attn_layer=MultiScaleDeformableAttention,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.norm = norm_layer(embed_dims)
        self.attn = attn_layer(
            embed_dims, num_heads, num_levels, num_points, batch_first=True, **kwargs)

    def forward(self,
                query,
                value=None,
                query_pos=None,
                ref_pts=None,
                spatial_shapes=None,
                level_start_index=None):
        # Drop(Attn(query)) + query
        query = query + self.attn(
            query,
            value=value,
            query_pos=query_pos,
            reference_points=ref_pts,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        # LayerNorm
        query = self.norm(query)
        return query
    
class FFN(nn.Module):
    """
    """
    def __init__(self, embed_dims, mlp_ratio=4, dropout=0.1) -> None:
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * mlp_ratio),
            nn.GELU()
        )
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dims * mlp_ratio, embed_dims)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dims)
    
    def forward(self, query):
        query2 = self.linear1(query)
        query2 = self.linear2(self.dropout1(query2))
        query = query + self.dropout2(query2)
        query = self.norm(query)
        return query