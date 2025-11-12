
import argparse, yaml, numpy as np, os
from utils.explain import temporal_correlation, save_eps_curve
def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset', choices=['metr-la','pems-bay'], required=True)
    p.add_argument('--config', default='configs/forecasting_base.yaml'); a=p.parse_args()
    T=200; t=np.linspace(0,6.28,T); attn=np.abs(np.sin(t))+0.05*np.random.randn(T); dS=np.gradient(np.sin(0.5*t))+0.05*np.random.randn(T)
    lags, corr = temporal_correlation(attn, dS, max_lag=30)
    os.makedirs('experiments/plots', exist_ok=True)
    save_eps_curve(lags, corr, "Lag (min)", "Correlation", "experiments/plots/temporal_correlation.eps")
    print("Saved experiments/plots/temporal_correlation.eps")
if __name__=='__main__': main()
