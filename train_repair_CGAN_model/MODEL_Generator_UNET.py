import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg16, VGG16_Weights


class ResidualBlock(nn.Module):
    """残差块，增强特征传播能力"""

    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += residual
        return self.relu(out)


class AttentionGate(nn.Module):
    """注意力门机制，增强重要特征"""

    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # 计算注意力权重
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)

        # 应用注意力
        value = self.value(x).view(batch_size, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        return self.gamma * out + x


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, use_residual=False):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        if use_residual:
            layers.append(ResidualBlock(out_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size1, in_size2, out_size, dropout=0.0, use_attention=False):
        super().__init__()
        # 上采样层：使用双线性插值+卷积减少棋盘效应
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_size1, in_size1, 3, padding=1, bias=False),
            nn.InstanceNorm2d(in_size1),
            nn.ReLU(inplace=True)
        )

        # 注意力门（可选）
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionGate(in_size2)

        # 跳跃连接处理层
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_size1 + in_size2, out_size, 3, padding=1),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
            ResidualBlock(out_size)  # 添加残差连接
        )
        if dropout:
            self.conv_block.add_module("dropout", nn.Dropout(dropout))

    def forward(self, x, skip):
        x = self.up(x)

        # 应用注意力机制
        if self.use_attention:
            skip = self.attention(skip)

        # 调整skip connection尺寸（如果需要）
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        # 初始卷积层（不使用下采样）
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 下采样路径（减少下采样次数）
        self.down1 = UNetDown(64, 128, use_residual=True)  # 128x128
        self.down2 = UNetDown(128, 256, use_residual=True)  # 64x64
        self.down3 = UNetDown(256, 512, dropout=0.5, use_residual=True)  # 32x32
        self.down4 = UNetDown(512, 512, dropout=0.5)  # 16x16
        self.down5 = UNetDown(512, 512, dropout=0.5)  # 8x8

        # 瓶颈层（添加自注意力）
        self.bottleneck = nn.Sequential(
            ResidualBlock(512),
            AttentionGate(512),
            ResidualBlock(512)
        )

        # 上采样路径（添加注意力机制）
        self.up1 = UNetUp(512, 512, 512, dropout=0.5, use_attention=True)  # 16x16
        self.up2 = UNetUp(512, 512, 512, dropout=0.5, use_attention=True)  # 32x32
        self.up3 = UNetUp(512, 256, 256, dropout=0.5)  # 64x64
        self.up4 = UNetUp(256, 128, 128)  # 128x128
        self.up5 = UNetUp(128, 64, 64)  # 256x256

        # 最终输出层
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, 7, padding=3),
            nn.Tanh(),
        )

        # VGG感知损失层（可选）
        self.vgg = None
        if out_channels == 3:  # 仅当输出为RGB时使用
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg.eval()

    def forward(self, x):
        # 初始卷积
        x0 = self.init_conv(x)

        # 下采样路径
        d1 = self.down1(x0)  # 128x128
        d2 = self.down2(d1)  # 64x64
        d3 = self.down3(d2)  # 32x32
        d4 = self.down4(d3)  # 16x16
        d5 = self.down5(d4)  # 8x8

        # 瓶颈层
        b = self.bottleneck(d5)

        # 上采样路径
        u1 = self.up1(b, d4)  # 16x16
        u2 = self.up2(u1, d3)  # 32x32
        u3 = self.up3(u2, d2)  # 64x64
        u4 = self.up4(u3, d1)  # 128x128
        u5 = self.up5(u4, x0)  # 256x256

        # print(d1.shape, d2.shape, d3.shape, d4.shape, d5.shape, b.shape, u1.shape, u2.shape, u3.shape, u4.shape, u5.shape)

        # 最终输出
        return self.final(u5)



if __name__ == '__main__':
    # 测试用例
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 创建改进后的生成器
    gen = GeneratorUNet(in_channels=2, out_channels=2).to(device)
    print(f"模型参数量: {sum(p.numel() for p in gen.parameters()) / 1e6:.2f}M")

    # 测试输入输出
    v1 = torch.randn((32, 2, 256, 256)).to(device)
    output = gen(v1)
    print(f"输入形状: {v1.shape}, 输出形状: {output.shape}")



