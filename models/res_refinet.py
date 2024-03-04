import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet101
import torchvision


class Resnet101RefineNet(nn.Module):
    def __init__(self, pretrained=True, channels=256):
      super().__init__()
      self.channels = channels
      self.encoder = self._make_resnet(pretrained)
      self.decoder = self._make_decoder()

    def _make_resnet(self, pretrained: bool) -> nn.Module:
      weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
      resnet = resnet101(weights=weights, progress=True)
      
      if pretrained:
        for p in resnet.parameters():
          p.requires_grad_(False)

      module = nn.Module()
      module.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1) # 256C*(1/4)H*(1/4)W
      module.layer2 = resnet.layer2 # 512C*(1/8)H*(1/8)W
      module.layer3 = resnet.layer3 # 1024C*(1/16)H*(1/16)W
      module.layer4 = resnet.layer4 # 2048C*(1/32)H*(1/32)W

      return module

    def _make_decoder(self) -> nn.Module:
      module = nn.Module()
      
      module.layer1_rn = nn.Conv2d(256, self.channels, 3, stride=1, padding=1, bias=False)
      module.layer2_rn = nn.Conv2d(512, self.channels, 3, stride=1, padding=1, bias=False)
      module.layer3_rn = nn.Conv2d(1024, self.channels, 3, stride=1, padding=1, bias=False)
      module.layer4_rn = nn.Conv2d(2048, self.channels, 3, stride=1, padding=1, bias=False)
      
      module.refinet4 = FeatureFusionModule(self.channels)
      module.refinet3 = FeatureFusionModule(self.channels)
      module.refinet2 = FeatureFusionModule(self.channels)
      module.refinet1 = FeatureFusionModule(self.channels)
      
      module.output_conv = nn.Sequential(
        nn.Conv2d(self.channels, 128, 3, stride=1, padding=1),
        Interpolate(2, mode='bilinear', align_corners=True), # 256C*H*W
        nn.Conv2d(128, 32, 3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        nn.ReLU(True),
      )

      return module
    
    def forward(self, x):
      layer1 = self.encoder.layer1(x)
      layer2 = self.encoder.layer2(layer1)
      layer3 = self.encoder.layer3(layer2)
      layer4 = self.encoder.layer4(layer3)

      layer1_rn = self.decoder.layer1_rn(layer1)
      layer2_rn = self.decoder.layer2_rn(layer2)
      layer3_rn = self.decoder.layer3_rn(layer3)
      layer4_rn = self.decoder.layer4_rn(layer4)

      ref4 = self.decoder.refinet4(layer4_rn) # 256*(1/16)H*(1/16)W
      ref3 = self.decoder.refinet3(ref4, layer3_rn) # 256*(1/8)H*(1/8)W
      ref2 = self.decoder.refinet2(ref3, layer2_rn) # 256*(1/4)H*(1/4)W
      ref1 = self.decoder.refinet1(ref2, layer1_rn) # 256*(1/2)H*(1/2)W

      return self.decoder.output_conv(ref1)
  
class FeatureFusionModule(nn.Module):
  def __init__(self, features):
    super().__init__()
    self.resConv1 = RedisualConvUnit(features)
    self.resConv2 = RedisualConvUnit(features)
  
  def forward(self, *x):
    out = x[0]
    
    if len(x) == 2:
      out += self.resConv1(x[1])
    
    out = self.resConv2(out)
    out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)

    return out

class RedisualConvUnit(nn.Module):
  def __init__(self, features):
    super().__init__()
    self.conv1 = nn.Conv2d(features, features, 3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(features, features, 3, stride=1, padding=1)
    self.relu = nn.ReLU(True)
    
  def forward(self, x):
    out = self.relu(x)
    out = self.relu(self.conv1(out))
    out = self.conv2(out)
    
    return out + x
  
class Interpolate(nn.Module):
  def __init__(self, scale_factor, mode, align_corners=False):
    super(Interpolate, self).__init__()

    self.interp = nn.functional.interpolate
    self.scale_factor = scale_factor
    self.mode = mode
    self.align_corners = align_corners

  def forward(self, x):
    x = self.interp(
        x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
    )

    return x

if __name__ == '__main__':
  # Just sanity check
  try:
    model = Resnet101RefineNet()
    out = model(torch.randn((1, 3, 224, 320)))
  except Exception as e:
    print('Something went wrong with Resnet101RefineNet model')
    print(e)