import GTR
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import DLinear
from iTransformer import iTransformer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.backbone = configs.backbone
        self.gtr = GTR(configs.seq_len, configs.pred_len, configs.cycle_len, configs.var_num, configs.period_len)
        self.backboneModel = self._build_model()
        

    def _build_model(self):
        model_dict = {
            'DLinear': DLinear,
            'iTransformer': iTransformer,
        }
        model = model_dict[self.backbone].BackboneModel(self.configs).float()

        return model
    
    def forward(self, x, q, cycle_index):
        return