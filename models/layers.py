from enum import Enum
from warnings import warn

import torch
from torch import nn
from torch.nn import functional as F

from models.backbone.efficientnet import EfficientNet
from models.backbone.resnet101 import ResnetBackbone
from models.backbone.densenet169 import DensenetBackbone


class Backbone(Enum):
  RESNET = 1
  EFFICIENT_NET = 2 # Not implemented
  DENSENET = 3


def make_backbone(model: Backbone, pretrained: bool = True):
  if model is Backbone.RESNET:
    resnet = ResnetBackbone(pretrained=pretrained)
    return resnet, [256, 512, 1024, 2048]
  elif model is Backbone.EFFICIENT_NET:
    return EfficientNet(), [192, 384, 768, 1536] # TOOD: Update
  elif model is Backbone.DENSENET:
    return DensenetBackbone(), [256, 512, 1280, 1664]
  else:
    raise ValueError("Unknown backbone")
    
class FeatureFusionModule(nn.Module):
  def __init__(self, features, negative_slope=0.0):
    super().__init__()
    self.resConv1 = RedisualConvUnit(features, negative_slope=negative_slope)
    self.resConv2 = RedisualConvUnit(features, negative_slope=negative_slope)
  
  def forward(self, *x):
    out = x[0]
    
    if len(x) == 2:
      out += self.resConv1(x[1])
    
    out = self.resConv2(out)
    out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)

    return out

class RedisualConvUnit(nn.Module):
  def __init__(self, features, negative_slope=0.0):
    super().__init__()
    self.conv1 = nn.Conv2d(features, features, 3, stride=1, padding=1, bias=False)
    self.batch_norm1 = nn.BatchNorm2d(features)
    self.conv2 = nn.Conv2d(features, features, 3, stride=1, padding=1, bias=False)
    self.batch_norm2 = nn.BatchNorm2d(features)
    self.leaky_relu = nn.LeakyReLU(negative_slope, inplace=True)
    
  def forward(self, x):
    out = self.leaky_relu(x)
    out = self.leaky_relu(
      self.batch_norm1(self.conv1(out))
    )
    out = self.batch_norm2(self.conv2(out))
    
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

class PatchEncoder(nn.Module):
  def __init__(
    self,
    input_shape, 
    patch_size: int = 16, 
    embedding_dim: int = 128, 
    n_heads: int = 4, 
    feedforward_dim: int = 1024
  ):
    '''
      input:
        input_shape (Tuple[int, int, int]) - CxHxW size of input
        patch_size (int) size of image patch
        embedding_dim (int) size of feature
        n_heads (int) number of transformers heads
        feedforward_dim (int) size of feedforward layer in trasnsformer
    '''
    super(PatchEncoder, self).__init__()
    
    if input_shape[1] % patch_size or input_shape[2] % patch_size:
      warn('Img size isn\'t divisible by patchsize, which may lead to worse performance')
    
    sequence = int((input_shape[1] // patch_size) * (input_shape[2] // patch_size))
    self.pos_encodding = nn.Parameter(
      torch.randn(sequence + 1, embedding_dim), # one extra for cls token
      requires_grad=True
    )
    self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)
    
    encoder_layers = nn.TransformerEncoderLayer(
      embedding_dim, n_heads, dim_feedforward=feedforward_dim
    )
    self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)
    self.patch_extractor = FlattenPatch(patch_size=patch_size)
    self.to_latent = nn.Linear(in_features=patch_size ** 2 * input_shape[0], out_features=embedding_dim)

  def forward(self, x):
    patches = self.to_latent(self.patch_extractor(x)) # NxSxP^2*C ~> NxSxEmb
    patches = torch.cat((self.cls_token.expand(patches.size(0), -1, -1), patches), dim=1)
    patches += self.pos_encodding.unsqueeze(0) # NxSxEmb

    embeddings = patches.permute(1, 0, 2) # SxNxEmb sequence first
    x = self.transformer_encoder(embeddings) # SxNxEmb
    return x


class ViT(nn.Module):
  def __init__(self, imgsize, in_channels, n_query_channels=128, patch_size=16, embedding_dim=128, n_heads=4, number_of_classes=10):
    super(ViT, self).__init__()
    self.n_query_channels = n_query_channels
    self.patch_encoder = PatchEncoder((in_channels, *imgsize), patch_size, embedding_dim, n_heads)
    self.inception = nn.Conv2d(in_channels, embedding_dim, 3, stride=1, padding=1)
    self.regression = nn.Sequential(
      nn.Linear(embedding_dim, 256),
      nn.BatchNorm1d(256),
      nn.LeakyReLU(True),
      nn.Linear(256, 128),
      nn.BatchNorm1d(128),
      nn.LeakyReLU(True),
      nn.Linear(128, number_of_classes),
    )
    
  def forward(self, x):
    '''
    input:
      x (torch.tensor)
        shape: NxCxHxW
    '''
    target = self.patch_encoder(x) # SxNxEmb
    keys = self.inception(x) # NxEmbxHxW

    regression_head, queries = target[0, ...], target[1:self.n_query_channels + 1, ...]
    queries = queries.permute(1, 0, 2) # QueryxNxEmb ~> NxQueryxEmb batch first

    attention_maps = PixelWiseDotProduct(keys, queries) # NxQueryxHxW

    y = F.softmax(self.regression(regression_head), dim=1) # NxNumber_of_classes

    return y, attention_maps

class FlattenPatch(nn.Module):
  def __init__(self, patch_size):
    super(FlattenPatch, self).__init__()
    self.P = patch_size
    self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
  
  def forward(self, x):
    B, C, _, _ = x.size()

    x = self.unfold(x)
    return x.view(B, C, self.P, self.P, -1) \
      .permute(0, 4, 1, 2, 3) \
      .flatten(2) # BxNxCxPxP ~> BxNxC*P^2

def PixelWiseDotProduct(x, y):
  '''
  input:
    x (torch.tesnor) - image features
      shape: NxCxHxW
    y (torch.tensor) - batch first sequence embeddings
      shape: NxSxEmb
  '''
  n, c, h, w = x.size()
  _, seq, emb = y.size()
  assert c == emb, "Number of channels in x and Embedding dimension (at dim 2) of y matrix must match"
  y = torch.matmul(
    x.view(n, c, h * w).permute(0, 2, 1), # NxHWxC(Emb)
    y.permute(0, 2, 1) # NxEmbxS
  )  # NxHWxS
  return y.permute(0, 2, 1).view(n, seq, h, w) # NxSxHxW
      
if __name__ == '__main__':
  # sanity check
  x = torch.rand(2, 3, 224, 320)
  patches = FlattenPatch(16)(x)

  print(patches.shape, (224 // 16) * (320 // 16))
  
  vit = ViT((224, 320), 3)
  vit.eval()
  out = vit(x)