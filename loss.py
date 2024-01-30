from typing import Union

import torch

_EPS = 1e-8

def sil_loss(y_hat, y_true, alpha=0.7, mask: Union[torch.Tensor, bool] = True):
  '''Scale invariant loss function, with tiny epsilon'''
  
  if isinstance(mask, bool):
    mask = (y_true > 0) if mask else torch.ones_like(y_true)
  
  diff = torch.log(y_hat[mask] + _EPS) - torch.log(y_true[mask] + _EPS)
  loss = (diff ** 2).mean() - (alpha * (diff.mean() ** 2))
  
  return loss