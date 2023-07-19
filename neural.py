import torch.nn.functional as F
import torch.nn as nn
import torch
import math

def conv3D(in_ch, out_ch=0, size=3, stride=1):
    padding = size // 2

    if out_ch <= 0:
        out_ch = in_ch

    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, size, stride=stride, padding=padding, bias=True),
        nn.LeakyReLU(inplace=True)
    )

class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.layers = nn.Sequential(
            conv3D(ch),
            conv3D(ch)
        )

    def forward(self, x):
        return self.layers(x) + x
        
class BlurPool(nn.Module):
    def __init__(self, ch, stride=2) -> None:
        super().__init__()

        self.f = None
        self.ch = ch
        self.set_dilation(1)
        self.stride = stride
        

    def __repr__(self):
        return f"BlurPool(stride={self.stride}, radius={self.f.size(0) // 2})"

    def set_dilation(self, dilation):
        if dilation == 1:
            f = torch.FloatTensor([1., 4., 6., 4., 1.])
            self.pad = 2
        elif dilation == 2:
            f = torch.FloatTensor([1., 2.5, 4., 5, 6., 5, 4., 2.5, 1.])
            self.pad = 4
        
        if self.f is not None:
            f = f.to(device=self.f.device, dtype=self.f.dtype)

        # Direct convolution is faster than separable one
        f = f.view(1, 1, -1, 1, 1) * f.view(1, 1, 1, -1, 1) * f.view(1, 1, 1, 1, -1)
        f = (f / torch.sum(f)).repeat(self.ch, 1, 1, 1, 1)
        self.f = nn.Parameter(f, requires_grad=False)

    def forward(self, x):
        ch = x.size(1)

        y = F.pad(x, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), mode="replicate")
        y = F.conv3d(y, self.f, groups=ch, stride=self.stride)

        return y


class DISANet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            conv3D(1, 8),
            ResBlock3D(8),
            conv3D(8, 16),
            BlurPool(16),
            ResBlock3D(16),
            conv3D(16, 32),
            BlurPool(32),
            ResBlock3D(32),
            nn.Conv3d(32, 16, 1, bias=False)
        )

    def forward(self, x):
        return self.layers(x)