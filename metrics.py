import torch

_EPS = 1e-8

def absolute_relative_mean_errror(y_hat, y_true):
  return ((y_true - y_hat).abs() / (y_true + _EPS)).mean()

def root_mean_squared_error(y_hat, y_true):
  return torch.sqrt(
    ((y_hat - y_true) ** 2).mean()
  )
  
def average_absolute_log_error(y_hat, y_true):
  return (
    torch.log10(y_hat + _EPS) - torch.log10(y_true + _EPS)
  ).abs().mean()
  

def threshold_acc(y_hat, y_true):
  thresh = torch.max(
    y_hat / (y_true + _EPS),
    y_true / (y_hat + _EPS)
  )
  N = torch.numel(y_hat)
  a1 = (thresh < 1.25).sum() / N
  a2 = (thresh < 1.25 ** 2).sum() / N
  a3 = (thresh < 1.25 ** 3).sum() / N
    
  return a1, a2, a3