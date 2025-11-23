import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SPPF', 'RFB_SPPF']

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicSepConv(nn.Module):
    def __init__(self, in_planes, kernel_size, stride=1, padding=None, dilation=1, groups=None, relu=True, bn=True,
                 bias=False):
        super(BasicSepConv, self).__init__()
        self.out_channels = in_planes

        # 自动计算 padding 以保持尺寸（当 stride=1 时）
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation  # 考虑 dilation

        # 强制 groups=in_planes（深度可分离卷积）
        groups = in_planes if groups is None else groups

        self.conv = nn.Conv2d(
            in_planes, in_planes, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(in_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        c_ = in_channels // 2
        self.cv1 = BasicConv(in_channels, c_, 1, 1)
        self.cv2 = BasicConv(c_ * 4, out_channels, 1, 1)

        # 修改池化层保持尺寸
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class RFB_SPPF(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=4, sppf_kernel=3):
        super().__init__()

        self.scale = scale
        self.out_channels = out_planes
        inter_planes = max(in_planes // map_reduce, 1)  # 确保至少为1

        # SPPF branch
        self.sppf = SPPF(in_planes, out_planes, k=sppf_kernel)

        # Branch0: 3x3 dilated conv (dilation=3)
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=1),  # 深度可分卷积
            BasicConv(inter_planes, inter_planes,
                      kernel_size=3, stride=1,
                      padding=1, dilation=1),  # padding = 3 for dilation=3
            BasicConv(inter_planes, out_planes, kernel_size=1, stride=1)
        )

        # Branch1: 3x3 dilated conv (dilation=5)
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=1),
            BasicConv(inter_planes, inter_planes,
                      kernel_size=3, stride=1,
                      padding=3, dilation=3),  # padding = 5 for dilation=5
            BasicConv(inter_planes, out_planes, kernel_size=1, stride=1)
        )

        # Branch2: 5x5 -> 3x3 dilated conv (dilation=7)
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicSepConv(inter_planes, kernel_size=5, stride=1, padding=2),  # 5x5卷积
            BasicConv(inter_planes, inter_planes,
                      kernel_size=3, stride=1,
                      padding=5, dilation=5),  # padding = 7 for dilation=7
            BasicConv(inter_planes, out_planes, kernel_size=1, stride=1)
        )

        # 修改后的Shortcut分支（关键修正）
        self.shortcut = nn.Sequential(
            BasicConv(in_planes, out_planes, kernel_size=1, stride=stride),
            # 当stride≠1时，通过池化调整尺寸
            nn.AvgPool2d(kernel_size=3, stride=stride, padding=1) if stride != 1 else nn.Identity()
        )

        # 后续处理层
        self.ConvLinear = BasicConv(3 * out_planes, out_planes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

        # 强制设置模块的stride属性（避免自动计算出错）
        self.stride = stride

    def forward(self, x):
        _, _, h, w = x.shape

        # 处理各分支
        sppf_out = self.sppf(x)
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        short = self.shortcut(x)

        # 打印各分支输出尺寸用于调试
      #  print(f"SPPF out: {sppf_out.shape}")
      #  print(f"Branch0: {x0.shape}, Branch1: {x1.shape}, Branch2: {x2.shape}")
      #  print(f"Shortcut: {short.shape}")

        # 统一调整所有输出到输入尺寸
        def _adjust_size(tensor):
            return F.interpolate(tensor, size=(h, w), mode='bilinear', align_corners=False)

        sppf_out = self.sppf(x)
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        short = self.shortcut(x)

        # 合并RFB分支
        out = torch.cat([x0, x1, x2], dim=1)
        out = self.ConvLinear(out)

        # 最终合并
        out = out * self.scale + short + sppf_out
        return self.relu(out)


# 测试代码
if __name__ == "__main__":
    # 测试标准尺寸
    x = torch.randn(2, 64, 32, 32)
    model = RFB_SPPF(in_planes=64, out_planes=128)
    out = model(x)
    print(f"Standard test output shape: {out.shape}")  # 预期: torch.Size([2, 128, 32, 32])

    # 测试非标准尺寸
    x = torch.randn(2, 64, 20, 20)
    out = model(x)
    print(f"Non-standard test output shape: {out.shape}")  # 预期: torch.Size([2, 128, 20, 20])