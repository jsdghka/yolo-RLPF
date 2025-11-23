import torch.nn as nn
import torch

__all__ = ['Bi_FPN']

class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Bi_FPN(nn.Module):
    def __init__(self, length, channels=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
        self.swish = nn.SiLU()  # 使用PyTorch内置的SiLU代替自定义swish
        self.epsilon = 1e-4

        # 添加通道对齐卷积层（如果输入通道数不同）
        if channels is not None:
            self.convs = nn.ModuleList([
                nn.Conv2d(channels[i], channels[-1], kernel_size=1) if channels[i] != channels[-1]
                else nn.Identity()
                for i in range(length)
            ])
        else:
            self.convs = nn.ModuleList([nn.Identity() for _ in range(length)])

    def forward(self, x):
        # 检查输入是否合法
        assert len(x) == len(self.convs), f"输入特征图数量({len(x)})与权重数量({len(self.convs)})不匹配"

        # 通道对齐：统一所有特征图的通道数为 channels[-1]
        aligned_feats = [self.convs[i](x[i]) for i in range(len(x))]

        # 加权融合
        weights = self.weight / (torch.sum(self.swish(self.weight), dim=0) + self.epsilon)
        weighted_feature_maps = [weights[i] * aligned_feats[i] for i in range(len(x))]
        stacked_feature_maps = torch.stack(weighted_feature_maps, dim=0)
        result = torch.sum(stacked_feature_maps, dim=0)
        return result