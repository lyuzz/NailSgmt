from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = DepthwiseSeparableConv(3, 16, stride=2)
        self.stage2 = DepthwiseSeparableConv(16, 24, stride=2)
        self.stage3 = DepthwiseSeparableConv(24, 40, stride=2)
        self.stage4 = DepthwiseSeparableConv(40, 64, stride=2)
        self.stage5 = DepthwiseSeparableConv(64, 96, stride=2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        x = self.stage1(x)
        feats.append(x)
        x = self.stage2(x)
        feats.append(x)
        x = self.stage3(x)
        feats.append(x)
        x = self.stage4(x)
        feats.append(x)
        x = self.stage5(x)
        feats.append(x)
        return feats


class MobileUNet(nn.Module):
    def __init__(self, encoder_pretrained: bool = True, apply_sigmoid: bool = True):
        super().__init__()
        self.encoder_pretrained = encoder_pretrained
        self.apply_sigmoid = apply_sigmoid
        self.use_torchvision = True

        try:
            weights = MobileNet_V3_Small_Weights.DEFAULT if encoder_pretrained else None
            self.encoder = mobilenet_v3_small(weights=weights).features
        except Exception:
            self.use_torchvision = False
            self.encoder = SimpleEncoder()

        self.enc_indices = None
        self.enc_channels = None
        self._init_decoder()

    def _infer_encoder_channels(self) -> List[int]:
        device = torch.device("cpu")
        dummy = torch.zeros(1, 3, 256, 256, device=device)
        if self.use_torchvision:
            feats = []
            x = dummy
            for i, layer in enumerate(self.encoder):
                x = layer(x)
                if i in self.enc_indices:
                    feats.append(x)
            return [f.shape[1] for f in feats]
        feats = self.encoder(dummy)
        return [f.shape[1] for f in feats]

    def _init_decoder(self) -> None:
        if self.use_torchvision:
            total_layers = len(self.encoder)
            self.enc_indices = [1, 3, 6, 9, total_layers - 1]
        else:
            self.enc_indices = []

        self.enc_channels = self._infer_encoder_channels()
        skip_channels = self.enc_channels[:-1]
        bottleneck_channels = self.enc_channels[-1]

        decoder_channels = [96, 64, 32, 24]
        decoder_blocks = []
        in_ch = bottleneck_channels
        for idx, skip_ch in enumerate(reversed(skip_channels)):
            out_ch = decoder_channels[idx]
            decoder_blocks.append(ConvBlock(in_ch + skip_ch, out_ch))
            in_ch = out_ch

        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.final_conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-2:]
        if self.use_torchvision:
            feats = []
            for i, layer in enumerate(self.encoder):
                x = layer(x)
                if i in self.enc_indices:
                    feats.append(x)
        else:
            feats = self.encoder(x)

        skips = feats[:-1]
        x = feats[-1]

        for skip, block in zip(reversed(skips), self.decoder_blocks):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        x = F.interpolate(x, size=orig_size, mode="bilinear", align_corners=False)
        x = self.final_conv(x)
        if self.apply_sigmoid:
            return torch.sigmoid(x)
        return x
