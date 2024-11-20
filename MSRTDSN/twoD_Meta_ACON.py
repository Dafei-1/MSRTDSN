import torch
from torch import nn

class AconC(nn.Module):
    r""" ACON activation (activate or not).
    # AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    # according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, height, width):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, height, width))  # 可学习参数 p1
        self.p2 = nn.Parameter(torch.randn(1, height, width))  # 可学习参数 p2
        self.beta = nn.Parameter(torch.ones(1, height, width))  # 可学习参数 beta

    def forward(self, x):
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(self.beta * (self.p1 * x - self.p2 * x)) + self.p2 * x


class MetaAconC(nn.Module):
    r""" ACON activation (activate or not).
    # MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is generated by a small network
    # according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, height, width, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(height, max(r, height // r), kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(max(r, height // r), track_running_stats=True)
        self.fc2 = nn.Conv2d(max(r, height // r), height, kernel_size=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm2d(height, track_running_stats=True)
        self.p1 = nn.Parameter(torch.randn(1, height, width))  # 可学习参数 p1
        self.p2 = nn.Parameter(torch.randn(1, height, width))  # 可学习参数 p2

    def forward(self, x):
        beta = torch.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(x.mean(dim=(2, 3), keepdims=True))))))
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(beta * (self.p1 * x - self.p2 * x)) + self.p2 * x