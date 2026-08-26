import GTR
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import DLinear
from iTransformer import iTransformer

class Combined_Model(nn.Module):
    def __init__(self, configs):
        super(Combined_Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.backbone = configs.backbone
        self.gtr = GTR(configs.seq_len, configs.pred_len, configs.cycle_len, configs.enc_in, configs.period_len)
        self.backboneModel = self._build_model()
        

    def _build_model(self):
        model_dict = {
            'DLinear': DLinear,
            'iTransformer': iTransformer,
        }
        model = model_dict[self.backbone].BackboneModel(self.configs).float()

        return model
    
    def forward(self, x, cycle_index, x_mark_enc=None):
        if self.configs.use_revin:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        # B: batch_size;    N: number of variate, can also includes covariates
        # T: seq_len;       S: pred_len

        x_gtr_out = self.gtr(x, cycle_index) # (B, T, N) -> (B, T, N)
        x_backbone_out = self.backboneModel(x_gtr_out, x_mark_enc) # (B, T, N) -> (B, S, N)

        if self.configs.use_revin:
            x_backbone_out = x_backbone_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            x_backbone_out = x_backbone_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return x_backbone_out[:, -self.pred_len:, :] # (B, T, D)