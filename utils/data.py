
import os, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

def ensure(path): os.makedirs(path, exist_ok=True)

def synthetic_series(num_nodes=32, num_steps=1000, seed=42):
    rng=np.random.default_rng(seed)
    t=np.arange(num_steps)
    base = 5+2*np.sin(2*np.pi*t/144)[:,None] + rng.normal(0,0.5,(num_steps,num_nodes))
    return base.astype(np.float32)

def load_or_synth(path):
    f=os.path.join(path,'speed.csv')
    if os.path.exists(f): return np.loadtxt(f,delimiter=',').astype(np.float32)
    return synthetic_series()

def ring_adjacency(n, sigma=3.0):
    coords=np.stack([np.cos(np.linspace(0,2*np.pi,n,endpoint=False)),
                     np.sin(np.linspace(0,2*np.pi,n,endpoint=False))],1)
    d=np.sqrt(((coords[:,None,:]-coords[None,:,:])**2).sum(-1))
    A=np.exp(-(d**2)/(sigma**2)).astype(np.float32); np.fill_diagonal(A,0.0); return A

class SeqDataset(Dataset):
    def __init__(self, series, window=12, horizon=12):
        if series.shape[0] < series.shape[1]: series = series.T
        self.X=series; self.W=window; self.H=horizon
    def __len__(self): return max(0, len(self.X)-self.W-self.H)
    def __getitem__(self, i):
        x=self.X[i:i+self.W]; y=self.X[i+self.W:i+self.W+self.H]
        return torch.from_numpy(x), torch.from_numpy(y)

def dataloaders(name, base="data", window=12, horizon=12, batch=16):
    path=os.path.join(base,name); ensure(path)
    S=load_or_synth(path); N=S.shape[1]; A=ring_adjacency(N)
    T=len(S); ntr=int(0.7*T); nv=int(0.15*T)
    tr,va,te=S[:ntr],S[ntr:ntr+nv],S[ntr+nv:]
    make=lambda arr: DataLoader(SeqDataset(arr,window,horizon), batch_size=batch, shuffle=True)
    return make(tr), make(va), make(te), torch.from_numpy(A)
