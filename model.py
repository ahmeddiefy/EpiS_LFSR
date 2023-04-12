import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Net(nn.Module):
    def __init__(self, angRes, factor):
        super(Net, self).__init__()
        channels = 64
        n_group = 5
        n_block = 5
        self.angRes = angRes
        self.factor = factor
        self.init_conv = nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False)
        self.disentg = CascadeDisentgGroup(n_group, n_block, angRes, channels)
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * factor ** 2, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(factor),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x_upscale = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)
        x = rearrange(x,"b 1 (u h) (v w) -> b 1 (h u) (w v)", u=self.angRes, v=self.angRes)
        buffer = self.init_conv(x)
        buffer = self.disentg(buffer)
        buffer_SAI = rearrange(buffer,"b c (h u) (w v) -> b c (u h) (v w)", u=self.angRes, v=self.angRes)
        out = self.upsample(buffer_SAI) + x_upscale
        return out


class CascadeDisentgGroup(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels):
        super(CascadeDisentgGroup, self).__init__()
        self.n_group = n_group
        Groups = []
        for i in range(n_group):
            Groups.append(DisentgGroup(n_block, angRes, channels))
        self.Group = nn.Sequential(*Groups)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_group):
            buffer = self.Group[i](buffer)
        return self.conv(buffer) + x


class DisentgGroup(nn.Module):
    def __init__(self, n_block, angRes, channels):
        super(DisentgGroup, self).__init__()
        self.n_block = n_block
        Blocks = []
        for i in range(n_block):
            Blocks.append(DisentgBlock(angRes, channels))
        self.Block = nn.Sequential(*Blocks)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_block):
            buffer = self.Block[i](buffer)
        return self.conv(buffer) + x


class DisentgBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(DisentgBlock, self).__init__()

        self.SpaConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
			nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
        )
               
        self.EPIConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=[1, angRes+2], stride=[1, 1], padding=[0, (angRes+2)//2], bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=[angRes+2, 1], stride=[1, 1], padding=[(angRes+2)//2, 0], bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=[1, angRes+2], stride=[1, 1], padding=[0, (angRes+2)//2], bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=[angRes+2, 1], stride=[1, 1], padding=[(angRes+2)//2, 0], bias=False),
        )
        
    def forward(self, x):
        feaEPI = self.EPIConv(x)
        buffer = feaEPI + x
        feaSpa = self.SpaConv(buffer)
        return feaSpa + buffer

if __name__ == "__main__":
    angRes = 5
    factor = 4
    net = Net(angRes=angRes, factor=factor).cuda()
    from thop import profile
    input = torch.randn(1, 1, 32*angRes, 32*angRes).cuda()
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))
