
import argparse, yaml, torch, os, copy, pandas as pd
from utils.data import dataloaders
from models.cvfnet import CVFNet
from utils.train import train_epoch, eval_epoch

def run_one(cfg, dataset, window, lam):
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl_tr,dl_va,dl_te,A=dataloaders(dataset, window=window, horizon=cfg['model']['forecast_horizon'], batch=cfg['training']['batch_size'])
    model=CVFNet(cfg).to(dev); opt=torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    for ep in range(cfg['training']['epochs']):
        train_epoch(model, dl_tr, opt, dev, A, lam)
    te=eval_epoch(model, dl_te, dev, A); return {'window':window,'lam1':lam[0],'lam2':lam[1], **te}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset', choices=['metr-la','pems-bay'], required=True)
    p.add_argument('--config', default='configs/forecasting_base.yaml'); a=p.parse_args()
    cfg=yaml.safe_load(open(a.config)); rows=[]
    for w in [6,12,18]:
        for lam in [(0.8,0.2),(0.6,0.4),(0.4,0.6)]: rows.append(run_one(copy.deepcopy(cfg), a.dataset, w, lam))
    os.makedirs('experiments/ablation', exist_ok=True)
    pd.DataFrame(rows).to_csv(f'experiments/ablation/{a.dataset}_ablation.csv', index=False); print('Saved ablation CSV.')
if __name__=='__main__': main()
