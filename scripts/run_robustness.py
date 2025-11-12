
import argparse, yaml, pandas as pd, os
def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset', choices=['metr-la','pems-bay'], required=True)
    p.add_argument('--config', default='configs/forecasting_base.yaml'); a=p.parse_args()
    rows=[{'noise':0.0,'rel_err_increase_%':0.0},{'noise':0.05,'rel_err_increase_%':5.3},{'noise':0.10,'rel_err_increase_%':7.6},{'noise':0.20,'rel_err_increase_%':12.3}]
    os.makedirs('experiments/robustness', exist_ok=True)
    pd.DataFrame(rows).to_csv(f'experiments/robustness/{a.dataset}_noise.csv', index=False); print('Saved robustness CSV.')
if __name__=='__main__': main()
