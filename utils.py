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

def count_min_max(data_loader: DataLoader) -> torch.Tensor:
  minimum, maximum = float('inf'), -float('inf')
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  for _, y_batch in data_loader:
    y_batch: torch.Tensor = y_batch.to(device)
    
    batch_min = y_batch.min().cpu().detach().item()
    minimum = min(batch_min, minimum)
    batch_max = y_batch.max().cpu().detach().item()
    maximum = max(batch_max, maximum)
  
  return minimum, maximum