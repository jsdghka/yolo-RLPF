import torch
import torch.nn as nn

__all__ = ['CSPPA']

class SimAM(nn.Module):
    """无参注意力模块（SimAM）"""
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        x_minus_mu = x - x.mean(dim=[2, 3], keepdim=True)
        y = x_minus_mu.pow(2) / (4 * (x_minus_mu.pow(2).sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * torch.sigmoid(y)

class Bottleneck_CSPPA(nn.Module):
    """带SimAM注意力的Bottleneck"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(c_, c2, 3, 1, 1, groups=g, bias=False)
        self.simam = SimAM()  # 嵌入注意力
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.simam(self.cv2(self.cv1(x))) if self.add else self.simam(self.cv2(self.cv1(x)))

class CSPPA(nn.Module):
    """完整的跨阶段部分注意力模块"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1, bias=False)
        self.m = nn.ModuleList(Bottleneck_CSPPA(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))  # 拆分为两部分
        y.extend(m(y[-1]) for m in self.m)  # 处理第二部分并扩展
        return self.cv2(torch.cat(y, 1))    # 拼接所有特征

