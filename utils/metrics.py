
import torch
def mae(pred, target): return torch.mean(torch.abs(pred-target))
def rmse(pred, target): return torch.sqrt(torch.mean((pred-target)**2))
def mape(pred, target, eps=1e-5): return torch.mean(torch.abs((pred-target)/(target.abs()+eps)))*100.0
def r2_score(pred, target):
    m= torch.mean(target); ss_tot=torch.sum((target-m)**2); ss_res=torch.sum((target-pred)**2)
    return 1-ss_res/(ss_tot+1e-8)
def bias_variance(errors): return torch.mean(errors), torch.std(errors)
