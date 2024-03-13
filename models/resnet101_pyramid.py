from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet101
import torchvision

# Shoudn't be used

class Resnet101Pyramid(nn.Module):
  def __init__(self, pretrained_encoder=True):
    super().__init__()
    weights = (
      torchvision.models.ResNet101_Weights.IMAGENET1K_V2
      if pretrained_encoder
      else None
    )
    encoder = resnet101(weights=weights, progress=True)
    for p in encoder.parameters():
      p.requires_grad_(False)

    self.layer0 = nn.Sequential(
      encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool
    )
    self.layer1 = nn.Sequential(encoder.layer1)  # CxHxW ~> 256x1/4Hx1/4W
    self.layer2 = nn.Sequential(encoder.layer2)  # CxHxW ~> 512x1/8Hx1/8W
    self.layer3 = nn.Sequential(encoder.layer3)  # CxHxW ~> 1024x1/16Hx1/16W
    self.layer4 = nn.Sequential(encoder.layer4)  # CxHxW ~> 2048x1/32Hx1/32W

    self.dec0 = nn.Conv2d(2048, 256, kernel_size=1)
    self.dec1 = nn.Conv2d(1024, 256, kernel_size=1)
    self.dec2 = nn.Conv2d(512, 256, kernel_size=1)
    self.dec3 = nn.Conv2d(256, 256, kernel_size=1)

    self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    self.predictor0 = self._smooth(256, 64)
    self.predictor1 = self._smooth(64, 1)

  def _smooth(self, in_channels, out_channels):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
      nn.ReLU(True),
    )

  def _upsample(self, x, y):
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode="bilinear") + y

  def forward(self, x):
    _, _, H, W = x.size()
    # Top -> Down
    e0 = self.layer0(x)
    e1 = self.layer1(e0)
    e2 = self.layer2(e1)
    e3 = self.layer3(e2)
    e4 = self.layer4(e3)

    # Down -> Top
    d0 = self.dec0(e4)  # # CxHxW ~> 256Cx1/32Hx1/32W
    d1 = self._upsample(d0, self.dec1(e3))  # 256Cx1/16Hx1/16W
    d1 = self.smooth1(d1)

    d2 = self._upsample(d1, self.dec2(e2))  # 256Cx1/8Hx1/8W
    d2 = self.smooth2(d2)

    d3 = self._upsample(d2, self.dec3(e1))  # 256Cx1/4Hx1/4W
    d3 = self.smooth3(d3)

    p = self.predictor0(d3)  # 64Cx1/4Hx1/4W
    p1 = self.predictor1(p)
    return F.interpolate(p1, size=(H, W), mode="bilinear")  # 1Cx1Hx1W