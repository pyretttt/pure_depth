from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.layers import Interpolate, FeatureFusionModule, make_backbone, Backbone, ViT


class ComposeNet(nn.Module):
  def __init__(
    self, 
    inputsize,
    min_max_values: Tuple[int, int] = (0, 10),
    decoder_channels: int = 256,
    vit_channels_in: int = 64,
    classes: int = 10
  ):
    super(ComposeNet, self).__init__()
    self.inputsize = inputsize
    self.min_max_values = min_max_values
    self.decoder_channels = decoder_channels
    self.classes = classes
    self.vit_channels_in = vit_channels_in
    
    self.backbone, backbone_channels_out = make_backbone(Backbone.DENSENET, pretrained=True)
    self.structure_estimator = StructureEstimator(
      backbone_channels=backbone_channels_out,
      channels=decoder_channels, 
      channels_out=vit_channels_in
    )
    
    self.metric_estimator = MetricEstimator(
      inputsize=inputsize,
      min_max_values=min_max_values,
      in_channels=vit_channels_in,
      classes=classes,
    )
    
  def forward(self, x):
    layer1, layer2, layer3, layer4 = self.backbone(x)
    structure = self.structure_estimator(layer1, layer2, layer3, layer4)
    metric = self.metric_estimator(structure)
    
    return metric

  def __repr__(self):
    return f'''
    ComposeNet(
      inputsize={self.inputsize},
      min_max_values={(self.min_max_values)},
      decoder_channels={self.decoder_channels},
      vit_channels_in={self.vit_channels_in},
      classes={self.classes}
    )
    '''
    

class StructureEstimator(nn.Module):
  def __init__(
    self, 
    backbone_channels: list[int], 
    channels=256, 
    channels_out=64,
    negative_slope=1e-2,
  ):
    super(StructureEstimator, self).__init__()
    self.layer1_rn = nn.Conv2d(backbone_channels[0], channels, 3, stride=1, padding=1, bias=False)
    self.layer2_rn = nn.Conv2d(backbone_channels[1], channels, 3, stride=1, padding=1, bias=False)
    self.layer3_rn = nn.Conv2d(backbone_channels[2], channels, 3, stride=1, padding=1, bias=False)
    self.layer4_rn = nn.Conv2d(backbone_channels[3], channels, 3, stride=1, padding=1, bias=False)
    
    self.refinet4 = FeatureFusionModule(channels, negative_slope=negative_slope)
    self.refinet3 = FeatureFusionModule(channels, negative_slope=negative_slope)
    self.refinet2 = FeatureFusionModule(channels, negative_slope=negative_slope)
    self.refinet1 = FeatureFusionModule(channels, negative_slope=negative_slope)
    
    self.output_conv = nn.Sequential(
      nn.LeakyReLU(negative_slope, inplace=True),
      nn.Conv2d(channels, 128, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(negative_slope, inplace=True),
      Interpolate(2, mode='bilinear', align_corners=True), # C*H*W
      # nn.Conv2d(channels, 128, kernel_size=3, stride=1, padding=1, bias=False),
      # nn.BatchNorm2d(128),
      nn.Conv2d(128, channels_out, kernel_size=3, stride=1, padding=1),
    )

  def forward(self, *x):
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
    self.distribution = nn.Sequential(
      nn.Conv2d(n_queries, classes, kernel_size=1),
      nn.Softmax(dim=1)  
    )

  def forward(self, x):
    distance, attention_maps = self.vit(x) # Nx${classes}, Nx${n_queries}xHxW
    out = self.distribution(attention_maps) # Nx${classes}xHxW
    
    distance_cum = distance.cumsum(dim=1)
    ranges = (self.max_value - self.min_value) * distance_cum
    # ranges = nn.functional.pad(ranges, (1, 0), mode='constant', value=self.min_value)
    # range_edges = torch.cumsum(ranges, dim=1)
    # centers = 0.5 * (range_edges[:, :-1] + range_edges[:, 1:])
    # centers = centers.view(*centers.size(), 1, 1)
    centers = ranges.view(*ranges.size(), 1, 1)

    return torch.sum(out * centers, dim=1, keepdim=True)
    

if __name__ == '__main__':
  x = torch.randn(2, 3, 224, 320)
  grad = torch.randn(2, 1, 224, 320)
  compose = ComposeNet((224, 320), decoder_channels=256)
  out = compose(x)

  print(out)