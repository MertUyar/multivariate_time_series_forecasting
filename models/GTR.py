import torch
import torch.nn as nn

class GTR(nn.Module):
    def __init__(self, seq_len, pred_len, cycle_len, enc_in, period_len=24):
        super(GTR, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.cycle_len = cycle_len
        self.linear = nn.Linear(seq_len, seq_len)
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 1 + 2 * period_len // 2), 
                                padding=(0, period_len // 2), stride=1, padding_mode='zeros', bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.Q = nn.Parameter(torch.zeros(cycle_len, enc_in), requires_grad=True)
    
    def forward(self, x, cycle_index): #x -> (B, T, N)
        _, T, N = x.shape
        x = x.permute(0, 2, 1) # (B, N, T)
        q_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len # (B, T)
        q = self.Q[q_index] # (B, T, N)
        q = q.permute(0, 2, 1) # (B, N, T)
        q = self.linear(q) # (B, N, T)
        stack = torch.stack([x, q], dim=2) # (B, N, 2, T)
        conv_out = self.conv2d(stack.reshape(-1, 1, 2, T)).reshape(-1, N, T) # (B, N, T)
        out = conv_out + x # (B, N, T)
        return out.permute(0, 2, 1) # (B, T, N)
