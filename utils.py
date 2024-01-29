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


if __name__ == '__main__':
  from nuy_v2_loader import NYUV2Dataset
  from torchvision.transforms import ToTensor, CenterCrop, Resize, Compose, Normalize
  
  NYU_V2_IMG_DIMENSION = (448, 608)

  MEAN = (0.4794, 0.4017, 0.3578)
  STD = (0.2069, 0.2064, 0.2123)

  transforms = Compose([
    ToTensor(), CenterCrop(NYU_V2_IMG_DIMENSION), Resize(size=(224, 320))
  ])
  
  train_dataset = NYUV2Dataset("ds", Compose([transforms, Normalize(mean=MEAN, std=STD, inplace=True)]), transforms, 'train', download=False, max_lenght=None)
  data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  
  for x, y in data_loader:
    print("mean:", x.mean(dim=(0, 2, 3)))
    print("std:", x.std(dim=(0, 2, 3)))