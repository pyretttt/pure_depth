from typing import Any, Callable

import torch
from torch.utils.data import DataLoader


def apply(
  data_loader: DataLoader,
  fn: Callable[[torch.Tensor, torch.Tensor], Any],
  reduction='sum'
):
  results = [fn(x, y) for x, y in data_loader]
  results = torch.stack(results, dim=0)
  if reduction == 'sum':
    return results.sum(dim=0)
  elif reduction == 'mean':
    return results.float().mean(dim=0)    

def count_mean_and_std_per_channel(data_loader: DataLoader) -> torch.Tensor:
  def stat_counter(x: torch.Tensor, _):
    N, C, _, _ = x.shape
    
    mean_perchannel = torch.mean(x, dim=(2, 3))
    std_perchannel = torch.std(x, dim=(2, 3))
  
    counts = torch.tensor([N] * C).float()
    mean_sum = mean_perchannel.sum(dim=0)
    mean_std = std_perchannel.sum(dim=0)
    
    return torch.stack((counts, mean_sum, mean_std), dim=0)

  result = apply(data_loader, stat_counter)
  
  count, mean, std = result
  return (mean / count, std / count)