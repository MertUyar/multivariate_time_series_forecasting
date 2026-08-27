import torch
import torch.nn as nn
from .transformer_encoder import Encoder, EncoderLayer
from .self_attention import FullAttention, AttentionLayer
from .embed import DataEmbedding_inverted


class BackboneModel(nn.Module):
    def __init__(self, configs):
        super(BackboneModel, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_revin
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)


    def forward(self, x_enc=None, x_mark_enc=None):
        _, _, N = x_enc.shape  # B T N
        # B: batch_size;    E: d_model;
        # T: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        
        # Embedding
        # B T N -> B N E                (B T N -> B T E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B T E -> B T E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out)
        
        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates
        #dec_out = dec_out[:, -self.pred_len:, :] this will done in combined_model [B, T, D]
        return dec_out  