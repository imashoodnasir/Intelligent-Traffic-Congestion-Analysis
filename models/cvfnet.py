
import torch, torch.nn as nn
from .vit_encoder import TinyViT
from .tgnn import TGNN
from .cva import CrossViewAttention
from .haft import HAFT
class CVFNet(nn.Module):
    def __init__(self, cfg):
        super().__init__(); m=cfg["model"]
        self.vit=TinyViT(**m["vit"])
        self.tgnn=TGNN(**m["tgnn"])
        self.cva=CrossViewAttention(m["vit"]["embed_dim"], m["tgnn"]["gru_dim"], heads=4)
        self.haft=HAFT(m["vit"]["embed_dim"], heads=4, depth=2)
        self.head=nn.Linear(m["vit"]["embed_dim"], 1)
    def forward(self, imgs, seq, A):
        vt, _ = self.vit(imgs)
        ge = self.tgnn(seq, A)
        ft, _ = self.cva(vt, ge)
        g = self.haft(ft)
        y = self.head(g).squeeze(-1)
        return y, {"graph": ge, "global": g}
