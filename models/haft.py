
import torch, torch.nn as nn
class HAFT(nn.Module):
    def __init__(self, d_model=128, heads=4, depth=2):
        super().__init__(); self.layers=nn.ModuleList([nn.TransformerEncoderLayer(d_model, heads, batch_first=True) for _ in range(depth)])
        self.proj=nn.Linear(d_model,d_model)
    def forward(self, tokens):
        x=tokens
        for l in self.layers: x=l(x)
        return self.proj(x.mean(1))
