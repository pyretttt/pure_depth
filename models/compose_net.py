from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.layers import Interpolate, FeatureFusionModule, make_backbone, Backbone, ViT

class ComposeNet(nn.Module):
  def __init__(
    self, 
    inputsize,
    min_max_values: Tuple[int, int] = (0.1, 10),
    channels: int = 256
  ):
    super(ComposeNet, self).__init__()
    self.channels = channels
    self.backbone = make_backbone(Backbone.RESNET, pretrained=True)
    channels_out = 64
    self.structure_estimator = StructureEstimator(channels=channels, channels_out=channels_out)
    
    self.metric_estimator = MetricEstimator(
      inputsize=inputsize,
      min_max_values=min_max_values,
      in_channels=channels_out,
      classes=10,
    )
    
  def forward(self, x):
    layer1 = self.backbone.layer1(x) 
    layer2 = self.backbone.layer2(layer1)
    layer3 = self.backbone.layer3(layer2)
    layer4 = self.backbone.layer4(layer3)
    structure = self.structure_estimator(layer1, layer2, layer3, layer4)
    metric = self.metric_estimator(structure)
    
    return metric


class StructureEstimator(nn.Module):
  def __init__(self, channels=256, channels_out=64, negative_slope=1e-2):
    super(StructureEstimator, self).__init__()
    self.layer1_rn = nn.Conv2d(256, channels, 3, stride=1, padding=1, bias=False)
    self.layer2_rn = nn.Conv2d(512, channels, 3, stride=1, padding=1, bias=False)
    self.layer3_rn = nn.Conv2d(1024, channels, 3, stride=1, padding=1, bias=False)
    self.layer4_rn = nn.Conv2d(2048, channels, 3, stride=1, padding=1, bias=False)
    
    self.refinet4 = FeatureFusionModule(channels, negative_slope=negative_slope)
    self.refinet3 = FeatureFusionModule(channels, negative_slope=negative_slope)
    self.refinet2 = FeatureFusionModule(channels, negative_slope=negative_slope)
    self.refinet1 = FeatureFusionModule(channels, negative_slope=negative_slope)
    
    self.output_conv = nn.Sequential(
      nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(channels),
      Interpolate(2, mode='bilinear', align_corners=True), # 256C*H*W
      nn.Conv2d(channels, 128, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope, inplace=True),
      nn.BatchNorm2d(128),
      nn.Conv2d(128, channels_out, kernel_size=3, stride=1, padding=1),
    )

  def forward(self, *x, with_features: bool = False):
    layer1, layer2, layer3, layer4 = x

    layer1_rn = self.layer1_rn(layer1)
    layer2_rn = self.layer2_rn(layer2)
    layer3_rn = self.layer3_rn(layer3)
    layer4_rn = self.layer4_rn(layer4)

    ref4 = self.refinet4(layer4_rn) # 256*(1/16)H*(1/16)W
    ref3 = self.refinet3(ref4, layer3_rn) # 256*(1/8)H*(1/8)W
    ref2 = self.refinet2(ref3, layer2_rn) # 256*(1/4)H*(1/4)W
    ref1 = self.refinet1(ref2, layer1_rn) # 256*(1/2)H*(1/2)W
    
    return self.output_conv(ref1)

class MetricEstimator(nn.Module):
  def __init__(
    self, 
    inputsize: Tuple[int, int],
    min_max_values: Tuple[int, int],
    in_channels: int,
    classes: int,
    n_queries=128,
    negative_slope=1e-2
  ):
    super(MetricEstimator, self).__init__()
    self.vit = ViT(inputsize, in_channels, n_query_channels=n_queries, number_of_classes=classes)
    self.leaky_relu = nn.LeakyReLU(negative_slope, inplace=True)
    self.min_value = min_max_values[0]
    self.max_value = min_max_values[1]
    self.inception = nn.Sequential(
      nn.Conv2d(n_queries, classes, kernel_size=1),
      nn.Softmax(dim=1)  
    )

  def forward(self, x):
    distance, attention_maps = self.vit(x) # NxEmb, Nx${n_queries}xHxW
    out = self.inception(attention_maps) # Nx${classes}xHxW
    
    ranges = (self.max_value - self.min_value) * distance
    ranges = nn.functional.pad(ranges, (1, 0), mode='constant', value=self.min_value)
    range_edges = torch.cumsum(ranges, dim=1)
    centers = 0.5 * (range_edges[:, :-1] + range_edges[:, 1:])
    centers = centers.view(*centers.size(), 1, 1)

    return torch.sum(out * centers, dim=1, keepdim=True)
    

if __name__ == '__main__':
  x = torch.randn(2, 3, 224, 320)
  grad = torch.randn(2, 1, 224, 320)
  compose = ComposeNet((224, 320), channels=256)
  out = compose(x)
