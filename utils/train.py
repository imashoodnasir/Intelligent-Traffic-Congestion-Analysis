
import torch
from .metrics import mae, rmse, mape, r2_score
def train_epoch(model, dl, opt, device, A, lam):
    model.train(); total=0; n=0
    for xb,yb in dl:
        xb,yb=xb.to(device), yb.to(device)
        B,W,N=xb.shape; imgs=torch.zeros((B,3,224,224), device=device)
        pred,_=model(imgs, xb, A.to(device))
        target=yb.mean(dim=(1,2)) if yb.dim()==3 else yb.mean(1)
        rec=torch.nn.functional.l1_loss(pred,target)
        temp=torch.mean(torch.abs(xb[:,1:]-xb[:,:-1]))
        loss=lam[0]*rec + lam[1]*temp
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item(); n+=1
    return total/max(1,n)
@torch.no_grad()
def eval_epoch(model, dl, device, A):
    model.eval(); M={"MAE":0,"RMSE":0,"MAPE":0,"R2":0}; n=0
    for xb,yb in dl:
        xb,yb=xb.to(device), yb.to(device)
        B,W,N=xb.shape; imgs=torch.zeros((B,3,224,224), device=device)
        pred,_=model(imgs, xb, A.to(device))
        target=yb.mean(dim=(1,2)) if yb.dim()==3 else yb.mean(1)
        M["MAE"]+=mae(pred,target).item(); M["RMSE"]+=rmse(pred,target).item()
        M["MAPE"]+=mape(pred,target).item(); M["R2"]+=r2_score(pred,target).item(); n+=1
    for k in M: M[k]/=max(1,n)
    return M
