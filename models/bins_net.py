import torch
from torch import nn

from models.layers import Interpolate, FeatureFusionModule, make_backbone, Backbone

class BinsNet(nn.Module):
  def __init__(self, input_size, n_bins=100, min_val=0.1, max_val=10, norm='linear', channels=64):
    super(BinsNet, self).__init__()
    self.channels = channels
    self.encoder = make_backbone(Backbone.RESNET, pretrained=True)
    self.decoder = self._make_decoder()
    self.num_classes = n_bins
    self.min_val = min_val
    self.max_val = max_val

    self.mvit = mViT(
      input_size=input_size,
      in_channels=channels, 
      n_query_channels=channels,
      patch_size=10,
      dim_out=100,
      embedding_dim=channels,
      norm=norm
    )
    self.conv_out = nn.Sequential(
      nn.Conv2d(channels, n_bins, kernel_size=1, stride=1, padding=0),
      nn.Softmax(dim=1)
    )
    
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
      nn.Conv2d(128, self.channels, 3, stride=1, padding=1),
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
    
    enc_dec_out = self.decoder.output_conv(ref1)
    bin_widths_normed, range_attention_maps = self.mvit(enc_dec_out)
    out = self.conv_out(range_attention_maps)

    # Post process
    # n, c, h, w = out.shape
    # hist = torch.sum(out.view(n, c, h * w), dim=2) / (h * w)  # not used for training

    bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
    bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
    bin_edges = torch.cumsum(bin_widths, dim=1)

    centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
    n, dout = centers.size()
    centers = centers.view(n, dout, 1, 1)

    pred = torch.sum(out * centers, dim=1, keepdim=True)

    return pred

    
class PatchEncoder(nn.Module):
  def __init__(self, input_size, in_channels, patch_size=10, embedding_dim=64, heads=4):
    super(PatchEncoder, self).__init__()
    encoder_layers = nn.TransformerEncoderLayer(embedding_dim, nhead=heads, dim_feedforward=1024)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)
    
    self.embedding_conv = nn.Conv2d(in_channels, embedding_dim, patch_size, stride=patch_size, padding=0)
    
    n_embeddings = int(input_size[0] * input_size[1] / (patch_size ** 2))
    self.positional_encodings = nn.Parameter(torch.rand(n_embeddings, embedding_dim), requires_grad=True)
    
  def forward(self, x):
    embeddings = self.embedding_conv(x).flatten(2)
    embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)
    
    embeddings = embeddings.permute(2, 0, 1) # S, N, E
    x = self.transformer_encoder(embeddings)  # .shape = S, N, E
    return x


class PixelWiseDotProduct(nn.Module):
  def __init__(self):
    super(PixelWiseDotProduct, self).__init__()

  def forward(self, x, K):
    n, c, h, w = x.size()
    _, cout, ck = K.size()
    assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
    y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))  # .shape = n, hw, cout
    return y.permute(0, 2, 1).view(n, cout, h, w)

class mViT(nn.Module):
    def __init__(
      self, 
      input_size, 
      in_channels, 
      n_query_channels=128, 
      patch_size=10, 
      dim_out=256,
      embedding_dim=64,
      num_heads=4, 
      norm='linear'
    ):
      super(mViT, self).__init__()
      self.norm = norm
      self.n_query_channels = n_query_channels
      self.patch_transformer = PatchEncoder(input_size, in_channels, patch_size, embedding_dim, num_heads)
      self.dot_product_layer = PixelWiseDotProduct()

      self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
      self.regressor = nn.Sequential(
        nn.Linear(embedding_dim, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 256),
        nn.LeakyReLU(),
        nn.Linear(256, dim_out)
      )

    def forward(self, x):
        # n, c, h, w = x.size()
        tgt = self.patch_transformer(x.clone())  # .shape = S, N, E

        x = self.conv3x3(x)

        regression_head, queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...]

        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(x, queries)  # .shape = n, n_query_channels, h, w

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps  


if __name__ == '__main__':
  enc = PatchEncoder((240, 320), 32)
  x = torch.randn(1, 32, 240, 320)
  enc.forward(x)