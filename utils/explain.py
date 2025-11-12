
import numpy as np, matplotlib.pyplot as plt
def temporal_correlation(attn, dS, max_lag=30):
    attn=(attn-attn.mean())/(attn.std()+1e-8); dS=(dS-dS.mean())/(dS.std()+1e-8)
    lags=np.arange(0,max_lag+1); corr=[]
    for L in lags:
        if L==0: c=np.corrcoef(attn,dS)[0,1]
        else:    c=np.corrcoef(attn[L:], dS[:-L])[0,1]
        corr.append(c)
    return lags, np.array(corr)
def save_eps_curve(x, y, xlabel, ylabel, out_path):
    plt.figure(); plt.plot(x,y, marker='o'); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(out_path, format='eps'); plt.close()
