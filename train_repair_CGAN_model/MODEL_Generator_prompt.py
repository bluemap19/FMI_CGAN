import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """改进的残差块，增强特征传播能力"""

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
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的缩放因子

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out = self.gamma * out + residual  # 缩放残差连接
        return self.relu(out)


class DownBlock(nn.Module):
    """下采样块 - 使用步长为2的卷积"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(out_channels)
        )

    def forward(self, x):
        return self.model(x)


class UpBlock(nn.Module):
    """上采样块 - 使用PixelShuffle"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 确保通道数可被4整除
        assert in_channels % 4 == 0, "in_channels must be divisible by 4"

        # 减少通道数以适应PixelShuffle
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.InstanceNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )

        # PixelShuffle上采样
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels // 4, (in_channels // 4) * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )

        # 跳跃连接处理
        self.conv_after_upsample = nn.Sequential(
            nn.Conv2d((in_channels // 4) + skip_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )

    def forward(self, x, skip):
        # 上采样路径
        x = self.conv_before_upsample(x)
        x = self.upsample(x)

        # 调整尺寸（如果需要）
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)

        # 拼接特征
        x = torch.cat([x, skip], dim=1)
        return self.conv_after_upsample(x)


class GeneratorUNetImproved(nn.Module):
    """改进的U-Net生成器（无注意力机制）"""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # 初始卷积层
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 下采样路径
        self.down1 = DownBlock(64, 128)  # 128x128
        self.down2 = DownBlock(128, 256)  # 64x64
        self.down3 = DownBlock(256, 512)  # 32x32
        self.down4 = DownBlock(512, 512)  # 16x16
        self.down5 = DownBlock(512, 512)  # 8x8

        # 瓶颈层（简化）
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512),
            ResidualBlock(512)
        )

        # 上采样路径
        self.up1 = UpBlock(512, 512, 512)  # 16x16
        self.up2 = UpBlock(512, 512, 512)  # 32x32
        self.up3 = UpBlock(512, 256, 256)  # 64x64
        self.up4 = UpBlock(256, 128, 128)  # 128x128
        self.up5 = UpBlock(128, 64, 64)  # 256x256

        # 最终输出层 - 使用Sigmoid激活
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, 7, padding=3),
            nn.Sigmoid(),  # 输出范围[0,1]
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, ResidualBlock):
                nn.init.constant_(m.gamma, 0.1)  # 初始化残差缩放因子

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

        # 最终输出
        return self.final(u5)


if __name__ == '__main__':
    # 测试设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 使用小尺寸图像测试
    gen = GeneratorUNetImproved(in_channels=2, out_channels=2).to(device)
    input_tensor = torch.randn((1, 2, 64, 64)).to(device)
    output = gen(input_tensor)
    print("输出范围:", output.min().item(), output.max().item())

    # # 创建改进后的生成器
    # gen = GeneratorUNetImproved(in_channels=2, out_channels=2).to(device)
    #
    # # 计算参数量
    # total_params = sum(p.numel() for p in gen.parameters())
    # trainable_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    # print(f"总参数量: {total_params / 1e6:.2f}M")
    # print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    #
    # # 测试输入输出
    # input_tensor = torch.randn((3, 2, 256, 256)).to(device)
    # output = gen(input_tensor)
    # print(f"输入形状: {input_tensor.shape}, 输出形状: {output.shape}")
    #
    # # 测试前向传播速度
    # import time
    #
    # start_time = time.time()
    # for _ in range(10):
    #     _ = gen(input_tensor)
    # elapsed_time = time.time() - start_time
    # print(f"10次前向传播耗时: {elapsed_time:.4f}秒")
    # print(f"平均每次前向传播耗时: {elapsed_time / 10:.4f}秒")
    #
    # # 测试梯度反向传播
    # loss = torch.mean(output)
    # loss.backward()
    # print("梯度反向传播测试成功")