"""
3D U-Net архитектура для сегментации печени на КТ-снимках.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class UNet3DConfiguration:
    """Конфигурация для 3D U-Net модели."""
    in_channels: int = 1
    out_channels: int = 2
    base_filters: int = 32
    depth: int = 4
    dropout_rate: float = 0.2
    use_batch_norm: bool = True


class DoubleConv3D(nn.Module):
    """Блок двойной 3D свертки"""
    
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch) if use_bn else nn.Identity()
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act(self.bn2(self.conv2(x)))
        return x


class Down3D(nn.Module):
    """Downsampling блок"""
    
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv3D(in_ch, out_ch, use_bn, dropout)
    
    def forward(self, x):
        return self.conv(self.pool(x))


class Up3D(nn.Module):
    """Upsampling блок с skip connection"""
    
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2)
        # После concat: (in_ch // 2) + skip_ch каналов
        self.conv = DoubleConv3D((in_ch // 2) + skip_ch, out_ch, use_bn, dropout)
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Выравнивание размеров
        diff_z = skip.size(2) - x.size(2)
        diff_y = skip.size(3) - x.size(3)
        diff_x = skip.size(4) - x.size(4)
        x = nn.functional.pad(x, [diff_x//2, diff_x-diff_x//2, diff_y//2, diff_y-diff_y//2, diff_z//2, diff_z-diff_z//2])
        
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net для сегментации"""
    
    def __init__(self, config: Optional[UNet3DConfiguration] = None):
        super().__init__()
        
        if config is None:
            config = UNet3DConfiguration()
        
        self.config = config
        bf = config.base_filters
        depth = config.depth
        
        # Каналы на каждом уровне энкодера
        encoder_channels = [bf * (2**i) for i in range(depth + 1)]  # [32, 64, 128, 256, 512]
        
        # Входной слой
        self.inc = DoubleConv3D(config.in_channels, bf, config.use_batch_norm, config.dropout_rate)
        
        # Энкодер
        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            self.down_blocks.append(Down3D(encoder_channels[i], encoder_channels[i+1], config.use_batch_norm, config.dropout_rate))
        
        # Боттлнек
        self.bottleneck = DoubleConv3D(encoder_channels[-1], encoder_channels[-1] * 2, config.use_batch_norm, config.dropout_rate)
        
        # Декодер (в обратном порядке каналов)
        self.up_blocks = nn.ModuleList()
        # Каналы: bottleneck->512->256->128->64->32
        decoder_in = encoder_channels[-1] * 2  # 1024
        for i in range(depth):
            skip_ch = encoder_channels[depth - 1 - i]  # [256, 128, 64, 32]
            out_ch = encoder_channels[depth - 2 - i] if i < depth - 1 else bf  # [128, 64, 32, 32]
            self.up_blocks.append(Up3D(decoder_in, skip_ch, out_ch, config.use_batch_norm, config.dropout_rate))
            decoder_in = out_ch
        
        # Выход
        self.outc = nn.Conv3d(bf, config.out_channels, 1)
    
    def forward(self, x):
        skips = []
        
        # Вход
        x = self.inc(x)
        skips.append(x)
        
        # Энкодер
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)
        
        # Боттлнек
        x = self.bottleneck(x)
        
        # Декодер (пропуски в обратном порядке)
        for i, up in enumerate(self.up_blocks):
            skip_idx = len(skips) - 2 - i
            x = up(x, skips[skip_idx])
        
        return self.outc(x)
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_unet3d(**kwargs):
    return UNet3D(UNet3DConfiguration(**kwargs))


if __name__ == "__main__":
    model = create_unet3d(base_filters=32, depth=4)
    x = torch.randn(1, 1, 64, 128, 128)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"Вход: {x.shape}, Выход: {y.shape}")
    print(f"Параметров: {model.get_num_parameters():,}")
    assert y.shape == (1, 2, 64, 128, 128)
    print("OK!")
