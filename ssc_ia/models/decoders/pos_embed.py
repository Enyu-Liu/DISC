from itertools import chain
import math
import torch
import torch.nn as nn

def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    """
    Args:
        pos : normalized points' 3D coors in shape of (N, 2)
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb

class LearnableSqueezePositionalEncoding(nn.Module):

    def __init__(self, num_embeds, embed_dims, squeeze_dims=None):
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Embedding(n_emb * squeeze_dims[i] if squeeze_dims else 1, embed_dims // 2)
            for i, n_emb in enumerate(num_embeds)
        ])
        self.proj = nn.Linear(embed_dims // 2 * len(num_embeds), embed_dims)
        self.shape = num_embeds
        self.squeeze_dims = squeeze_dims

    def forward(self):
        embeds = []
        for i, s in enumerate(self.shape):
            shape = [1 for _ in self.shape]
            shape[i] = s * self.squeeze_dims[i] if self.squeeze_dims else 1
            embed = self.embeds[i].weight.reshape(1, *shape, -1).expand(
                1, *[
                    s * self.squeeze_dims[i] if self.squeeze_dims else 1
                    for i, s in enumerate(self.shape)
                ], -1)
            embeds.append(embed)
        embed = torch.cat(embeds, dim=-1)
        if self.squeeze_dims:
            shape = list(chain(*[(s, self.squeeze_dims[i]) for i, s in enumerate(self.shape)]))
            dims = [2 * i + 1
                    for i in range(len(self.shape))] + [2 * i + 2 for i in range(len(self.shape))]
            embed = embed.reshape(1, *shape, -1).permute(0, *dims, -1).flatten(1, -2)
        embed = self.proj(embed)
        return embed


class FactorizedPositionEmbedding(nn.Module):

    def __init__(self, num_embeds, embed_dims, squeeze_dims=None):
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Embedding(n_emb * squeeze_dims[i] if squeeze_dims else 1, embed_dims)
            for i, n_emb in enumerate(num_embeds)
        ])
        self.shape = num_embeds
        self.squeeze_dims = squeeze_dims

    def forward(self):
        embeds = 1
        for i, s in enumerate(self.shape):
            shape = [1 for _ in self.shape]
            shape[i] = s * self.squeeze_dims[i] if self.squeeze_dims else 1
            embed = self.embeds[i].weight.reshape(1, *shape, -1).expand(
                1, *[
                    s * self.squeeze_dims[i] if self.squeeze_dims else 1
                    for i, s in enumerate(self.shape)
                ], -1)
            embeds = embeds * embed
        if self.squeeze_dims:
            shape = list(chain(*[(s, self.squeeze_dims[i]) for i, s in enumerate(self.shape)]))
            dims = [2 * i + 1
                    for i in range(len(self.shape))] + [2 * i + 2 for i in range(len(self.shape))]
            embed = embed.reshape(1, *shape, -1).permute(0, *dims, -1).flatten(1, -2)
        return embed
