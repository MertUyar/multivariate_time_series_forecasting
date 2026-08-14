class ModelConfig:
        def __init__(self, cycle_len, var_num, period_len=24, seq_len=96, pred_len=24, output_attention=False, use_revin=True,
                    factor=5, d_model=256, n_heads=8, e_layers=2, d_ff=512, dropout=0.1,
                    embed='fixed', freq='h', activation='gelu', backbone='iTransformer', individual=True):
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.output_attention = output_attention
            self.use_revin = use_revin
            self.factor = factor
            self.d_model = d_model
            self.n_heads = n_heads
            self.e_layers = e_layers
            self.d_ff = d_ff
            self.dropout = dropout
            self.embed = embed
            self.freq = freq
            self.activation = activation
            self.cycle_len = cycle_len
            self.var_num = var_num
            self.period_len = period_len
            self.backbone = backbone
            self.individual = individual