
import argparse, yaml, torch, os
from utils.data import dataloaders
from models.cvfnet import CVFNet
from utils.train import train_epoch, eval_epoch

def device_of(flag): return torch.device("cuda" if (flag=='cuda_if_available' and torch.cuda.is_available()) else "cpu")

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['metr-la','pems-bay'], required=True)
    p.add_argument('--config', default='configs/forecasting_base.yaml')
    a=p.parse_args()
    cfg=yaml.safe_load(open(a.config))
    torch.manual_seed(cfg['seed'])
    dev=device_of(cfg['device'])
    dl_tr,dl_va,dl_te,A = dataloaders(a.dataset, window=cfg['model']['temporal_window'],
                                      horizon=cfg['model']['forecast_horizon'],
                                      batch=cfg['training']['batch_size'])
    model=CVFNet(cfg).to(dev)
    opt=torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    best=1e9; patience=0
    for ep in range(cfg['training']['epochs']):
        tr=train_epoch(model, dl_tr, opt, dev, A, (cfg['loss']['lambda_rec'], cfg['loss']['lambda_temp']))
        va=eval_epoch(model, dl_va, dev, A)
        print(f"Epoch {ep+1}: train={tr:.4f} | val_MAE={va['MAE']:.4f} R2={va['R2']:.3f}")
        if va['MAE']<best: best=va['MAE']; patience=0; os.makedirs('experiments/checkpoints', exist_ok=True); 
        torch.save(model.state_dict(), f'experiments/checkpoints/best_{a.dataset}.pt')
        patience+=1
        if patience>=cfg['training']['early_stop_patience']: break
    te=eval_epoch(model, dl_te, dev, A)
    print('TEST:', te)
if __name__=='__main__': main()
