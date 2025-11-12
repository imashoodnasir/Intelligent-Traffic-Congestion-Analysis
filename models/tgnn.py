
import torch, torch.nn as nn
class GraphConv(nn.Module):
    def __init__(self, fin, fout): super().__init__(); self.lin=nn.Linear(fin,fout)
    def forward(self, x, A): return torch.relu(self.lin(A@x))
class TGNN(nn.Module):
    def __init__(self, node_dim=32, gcn_dim=64, gru_dim=64):
        super().__init__(); self.proj=nn.Linear(1, node_dim); self.gcn=GraphConv(node_dim,gcn_dim)
        self.gru=nn.GRU(gcn_dim, gru_dim, batch_first=True); self.out=nn.Linear(gru_dim, gru_dim)
    def forward(self, seq, A):
        B,W,N=seq.shape
        x=seq.view(B*W,N,1); x=torch.relu(self.proj(x)); x=self.gcn(x, A)  # [B*W,N,g]
        x=x.view(B,W,N,-1).mean(2)                                         # [B,W,g]
        y,_=self.gru(x); return self.out(y[:,-1,:])
