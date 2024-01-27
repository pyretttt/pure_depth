from torch import nn
from torch.nn import functional as F
from torchvision.models import densenet169
import torchvision

class Densenet169Pyramid(nn.Module):
  def __init__(self, pretained: bool = True):
    super().__init__()
    weights = torchvision.models.DenseNet169_Weights.IMAGENET1K_V1 if pretained else None
    self.encoder = densenet169(weights=weights, progress=True)
    for p in self.encoder.parameters():
      p.requires_grad_(False)

if __name__ == '__main__':
  model = densenet169(weights=torchvision.models.DenseNet169_Weights.IMAGENET1K_V1, progress=True)