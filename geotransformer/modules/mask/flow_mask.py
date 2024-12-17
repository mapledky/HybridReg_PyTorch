import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from geotransformer.modules.geotransformer import (
    GeometricTransformer,
)


class LaplaceMask(nn.Module):
    def __init__(self, cfg):
        super(LaplaceMask, self).__init__()
        #self.sinkmatch = Matching(dim)
        attn_out_dim = cfg.maskformer.output_dim

        self.linear_1 = nn.Linear(attn_out_dim, attn_out_dim // 2, bias=False)
        self.linear_2 = nn.Linear(attn_out_dim // 2, attn_out_dim // 4, bias=False)

        self.linear_3 = nn.Linear(attn_out_dim // 4, attn_out_dim // 8, bias=False)
        self.linear_4 = nn.Linear(attn_out_dim // 8, 1, bias=False)

        self.attention = GeometricTransformer(
                    cfg.maskformer.input_dim,
                    cfg.maskformer.output_dim,
                    cfg.maskformer.hidden_dim,
                    cfg.maskformer.num_heads,
                    cfg.maskformer.blocks,
                    cfg.maskformer.sigma_d,
                    cfg.maskformer.sigma_a,
                    cfg.maskformer.angle_k,
                    reduction_a=cfg.maskformer.reduction_a,
                )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,ref_c, src_c, ref_f, src_f):
        #src_f B,N,C
        #ref_f B,M,C
        B,N,_ = src_f.shape
        _,M,_ = ref_f.shape
        ref_f, src_f = self.attention(
            ref_c,
            src_c,
            ref_f,
            src_f,
        )
        #(B,N+M,attn_ouput)
        x = torch.cat((ref_f, src_f), dim=1)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = self.relu(x)
        x = self.linear_4(x)
        x = self.sigmoid(x)# B, N+M, 1
        
        return x.reshape(B, 1, N + M).contiguous()





