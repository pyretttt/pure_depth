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
  
def threshold_accuracy(threshold):
  def _threshold_acc(y_hat, y_true):
    thr = torch.max(
      y_hat / (y_true + _EPS),
      y_true / (y_hat + _EPS)
    ) < threshold
    
    return torch.sum(thr) / torch.numel(thr)
  
  return _threshold_acc

THRESHOLD = 1.25
threshold_acc_delta_1 = threshold_accuracy(THRESHOLD)
threshold_acc_delta_2 = threshold_accuracy(THRESHOLD ** 2)
threshold_acc_delta_3 = threshold_accuracy(THRESHOLD ** 3)