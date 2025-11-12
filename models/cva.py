
import torch, torch.nn as nn
class CrossViewAttention(nn.Module):
    def __init__(self, d_q=128, d_kv=64, heads=4):
        super().__init__(); self.q=nn.Linear(d_q,d_q); self.k=nn.Linear(d_kv,d_q); self.v=nn.Linear(d_kv,d_q)
        self.att=nn.MultiheadAttention(d_q, heads, batch_first=True)
    def forward(self, v_tokens, g_embed):
        q=self.q(v_tokens); kv=g_embed.unsqueeze(1); k=self.k(kv); v=self.v(kv)
        out,_=self.att(q,k,v); return out, out.mean(1)
