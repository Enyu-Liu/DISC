import torch
import torch.nn as nn
from ..modules import ASPP



class SeperateAspp(nn.Module):
    def __init__(self, embed_dims) -> None:
        super().__init__()

        self.aspp_ins = ASPP(embed_dims, (1, 2))
        self.aspp_bk = ASPP(embed_dims, (1, 3))
        self.tail_conv = nn.Sequential(
            nn.Conv3d(embed_dims, embed_dims, kernel_size=3, padding=1),
            nn.BatchNorm3d(embed_dims), 
            nn.ReLU(inplace=True)
        )
    
    def forward(self, volume, ins_v, bk_v):
        ins_v = self.aspp_ins(ins_v)
        bk_v = self.aspp_bk(bk_v)
        return self.tail_conv((volume + ins_v + bk_v))