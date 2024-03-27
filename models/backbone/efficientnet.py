import torch
from torch import nn

from utils import make_write_output_hook


class EfficientNet(nn.Module):
  def __init__(
    self,
    to_hook: list[int] = [0, 2, 4],
    pretrained: bool = True
  ):
    super(EfficientNet, self).__init__()
    model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_b5_ap', pretrained=pretrained)
    model.global_pool = nn.Identity()
    model.classifier = nn.Identity()
    self.to_hook = to_hook
    
    self.model = model
    self.outputs = dict()
    
    model.blocks[to_hook[0]] \
      .register_forward_hook(make_write_output_hook(self.outputs, '0'))
    model.blocks[to_hook[1]] \
      .register_forward_hook(make_write_output_hook(self.outputs, "1"))
    model.blocks[to_hook[2]] \
      .register_forward_hook(make_write_output_hook(self.outputs, "2"))
    model.conv_head \
      .register_forward_hook(make_write_output_hook(self.outputs, "3"))

    if pretrained:
      for p in model.parameters():
         p.requires_grad_(False)
  
  def forward(self, x):
    self.model(x)
    out1, out2, out3, out4 = self.outputs['0'], self.outputs['1'], self.outputs['2'], self.outputs['3']
    
    return out1, out2, out3, out4

if __name__ == '__main__':
  model = EfficientNet()
  x = torch.randn(3, 3, 224, 320)
  
  out = model(x)
  print(out)