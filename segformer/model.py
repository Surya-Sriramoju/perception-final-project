import torch
import torch.nn as nn
import torch.nn.functional as F

class SegFormer(nn.Module):
    def __init__(self, in_channels=3, n_classes=19, embed_dims=64, n_heads=8, depth=5, dropout=0.1):
        super(SegFormer, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dims, nhead=n_heads, dim_feedforward=2048, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims, n_classes, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.encoder(x)

        x = x.permute(2, 3, 0, 1)
        x = self.transformer_encoder(x)
        x = x.permute(2, 3, 0, 1)

        x = self.decoder(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        return x
