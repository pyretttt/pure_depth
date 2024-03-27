import torch
import torchvision
from torchvision.models import densenet169
from torch import nn

from utils import make_write_output_hook


class DensenetBackbone(nn.Module):
  def __init__(
    self,
    to_hook: list[int] = [4, 6, 8, 11],
    pretrained: bool = True
  ):
    super(DensenetBackbone, self).__init__()
    weights = torchvision.models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet169(weights=weights)
    model.classifier = nn.Identity()
    self.to_hook = to_hook
    
    self.model = model
    self.outputs = dict()
    
    model.features[to_hook[0]] \
      .register_forward_hook(make_write_output_hook(self.outputs, '0')) # 256Cx(1/4)HxW
    model.features[to_hook[1]] \
      .register_forward_hook(make_write_output_hook(self.outputs, "1")) # 512Cx(1/8)HxW
    model.features[to_hook[2]] \
      .register_forward_hook(make_write_output_hook(self.outputs, "2")) # 1280Cx(1/16)HxW
    model.features[to_hook[3]] \
      .register_forward_hook(make_write_output_hook(self.outputs, "3")) # 1664Cx(1/32)HxW

    if pretrained:
      for p in model.parameters():
         p.requires_grad_(False)
  
  def forward(self, x):
    self.model(x)
    out1, out2, out3, out4 = self.outputs['0'], self.outputs['1'], self.outputs['2'], self.outputs['3']
    
    return out1, out2, out3, out4

if __name__ == '__main__':
  model = DensenetBackbone()
  x = torch.randn(3, 3, 224, 320)
  
  out = model(x)
  print(out)