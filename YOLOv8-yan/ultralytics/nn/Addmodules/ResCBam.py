import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResCBAM']
class ChannelAttention(nn.Module):
    """通道注意力机制模块"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared two-layer MLP (implemented using 1x1 conv)
        self.MLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

    def forward(self, x):
        # Average pooling branch
        avg_out = self.MLP(self.avg_pool(x))

        # Max pooling branch
        max_out = self.MLP(self.max_pool(x))

        # Combine and generate channel attention map
        channel_att = torch.sigmoid(avg_out + max_out)

        # Element-wise multiplication for feature refinement
        return x * channel_att


class SpatialAttention(nn.Module):
    """空间注意力机制模块"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size should be odd"

        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Average and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Combine spatial information
        spatial_feat = torch.cat([avg_out, max_out], dim=1)

        # Generate spatial attention map
        spatial_att = self.spatial_att(spatial_feat)

        # Element-wise multiplication for feature refinement
        return x * spatial_att


class CBAM(nn.Module):
    """完整的CBAM模块（通道注意力+空间注意力）"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        # Channel attention -> Spatial attention
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class ResBlock(nn.Module):
    """基础残差块"""

    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock, self).__init__()
        stride = 2 if downsample else 1

        self.res_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Shortcut connection for downsampling
        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return F.relu(self.res_path(x) + self.shortcut(x))


class ResCBAM(nn.Module):
    """完整的ResCBAM模块（ResBlock + CBAM）"""

    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResCBAM, self).__init__()
        self.resblock = ResBlock(in_channels, out_channels, downsample)
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.resblock(x)
        x = self.cbam(x)
        return x