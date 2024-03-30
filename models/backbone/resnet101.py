import torchvision
from torchvision.models import resnet101
from torch import nn
import torch


class ResnetBackbone(nn.Module):
  def __init__(self, pretrained: bool = True):
    super(ResnetBackbone, self).__init__()
    weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
    resnet = resnet101(weights=weights, progress=True)
    
    if pretrained:
      for p in resnet.parameters():
        p.requires_grad_(False)

    self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1) # 256C*(1/4)H*(1/4)W
    self.layer2 = resnet.layer2 # 512C*(1/8)H*(1/8)W
    self.layer3 = resnet.layer3 # 1024C*(1/16)H*(1/16)W
    self.layer4 = resnet.layer4 # 2048C*(1/32)H*(1/32)W
    
  def forward(self, x):
    layer1 = self.layer1(x) 
    layer2 = self.layer2(layer1)
    layer3 = self.layer3(layer2)
    layer4 = self.layer4(layer3)
    return layer1, layer2, layer3, layer4
  
if __name__ == '__main__':
  model = ResnetBackbone()
  x = torch.randn(3, 3, 224, 320)
  
  out = model(x)
  print(out)
  