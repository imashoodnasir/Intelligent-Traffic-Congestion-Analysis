
import torch, torch.nn as nn
class PatchEmbed(nn.Module):
    def __init__(self,img=224,patch=16,embed=128): 
        super().__init__(); self.proj=nn.Conv2d(3,embed,patch,patch)
    def forward(self,x): return self.proj(x).flatten(2).transpose(1,2)
class TinyViT(nn.Module):
    def __init__(self,img_size=224, patch_size=16, embed_dim=128, depth=2, num_heads=4):
        super().__init__()
        self.patch=PatchEmbed(img_size,patch_size,embed_dim)
        enc=nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True)
        self.enc=nn.TransformerEncoder(enc, depth)
    def forward(self, imgs):
        tok=self.patch(imgs); out=self.enc(tok); return out, out.mean(1)
