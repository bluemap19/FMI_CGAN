import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg16, VGG16_Weights


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size1, in_size2, out_size, dropout=0.0):
        super().__init__()
        # 上采样层：输入in_size -> 输出out_size
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_size1, in_size1, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(in_size1),
            nn.ReLU(inplace=True)
        )
        # 跳跃连接处理层：输入out_size*2 -> 输出out_size
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_size1+in_size2, out_size, 3, padding=1),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True)
        )
        if dropout:
            self.conv_block.add_module("dropout", nn.Dropout(dropout))

    def forward(self, x, skip):
        x = self.up(x)  # 上采样
        x = torch.cat([x, skip], dim=1)  # 拼接跳跃连接
        return self.conv_block(x)  # 卷积融合特征


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 1024, dropout=0.5)
        self.down6 = UNetDown(1024, 1024, dropout=0.5)
        self.down7 = UNetDown(1024, 1024, dropout=0.5)
        self.down8 = UNetDown(1024, 1024, normalize=False, dropout=0.5)

        # 上采样部分修正输入通道
        self.up1 = UNetUp(1024, 1024, 1024, dropout=0.5)  # 输出1024
        self.up2 = UNetUp(1024, 1024, 1024, dropout=0.5)  # 输入1024
        self.up3 = UNetUp(1024, 1024, 512, dropout=0.5)
        self.up4 = UNetUp(512, 512, 256, dropout=0.5)
        self.up5 = UNetUp(256, 256, 128)  # 输入512 -> 输出256
        self.up6 = UNetUp(128, 128, 64)  # 输入256 -> 输出128
        self.up7 = UNetUp(64, 64, 64)  # 输入128 -> 输出64

        # 最终输出层优化
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, out_channels, 3, padding=1),  # 输入64通道
            nn.Tanh(),
        )


    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)



if __name__ == '__main__':
    gen = GeneratorUNet(in_channels=6, out_channels=2)
    print(gen)
    v1 = torch.rand((5, 6, 256, 256))
    v2 = torch.rand((5, 2, 256, 256))
    v1_gen = gen.forward(v1)
    print(v1_gen.shape)
