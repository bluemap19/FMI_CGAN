import torch
from torch import nn
from torch.nn import functional as F

class SelfAttentionBlock(nn.Module):
    """中心区域自注意力机制"""

    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.center_mask = self.create_center_mask()

    def create_center_mask(self, size=16, radius=0.7):
        """创建中心区域掩码"""
        y, x = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size))
        mask = (x ** 2 + y ** 2) < radius ** 2
        return mask

    def forward(self, x):
        B, C, H, W = x.size()
        center_mask = self.center_mask.to(x.device)

        # 提取中心区域特征
        center_feat = x[:, :, center_mask].view(B, C, -1)

        # 计算注意力
        q = self.query(center_feat)
        k = self.key(center_feat)
        v = self.value(center_feat)

        attn = torch.softmax(torch.bmm(q.permute(0, 2, 1), k), dim=-1)
        center_out = torch.bmm(v, attn.permute(0, 2, 1))

        # 更新中心特征
        output = x.clone()
        output[:, :, center_mask] = self.gamma * center_out.view(-1) + x[:, :, center_mask]
        return output


class ResBlock(nn.Module):
    """带空洞卷积的残差块，增强中心感受野"""

    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + residual)


class CenterConvBlock(nn.Module):
    """中心区域专用卷积，强化中心细节"""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.center_conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.edge_conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        # 创建中心掩码
        _, _, H, W = x.shape
        y_coord, x_coord = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
        center_mask = (x_coord ** 2 + y_coord ** 2) < 0.6

        # 分别处理中心和边缘
        center = x * center_mask.float().to(x.device)
        edge = x * (1 - center_mask.float().to(x.device))

        center_out = self.center_conv(center)
        edge_out = self.edge_conv(edge)

        return center_out + edge_out


class RefineBlock(nn.Module):
    """输出层特征细化，多尺度融合"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(in_ch, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(in_ch, 64, 7, padding=3)

        self.combine = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, out_ch, 3, padding=1)
        )

    def forward(self, x):
        feat1 = F.relu(self.conv1(x))
        feat2 = F.relu(self.conv2(x))
        feat3 = F.relu(self.conv3(x))

        combined = torch.cat([feat1, feat2, feat3], dim=1)
        return self.combine(combined)



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
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        # 上采样层：输入in_size -> 输出out_size
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True)
        )
        # 跳跃连接处理层：输入out_size*2 -> 输出out_size
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_size * 2, out_size, 3, padding=1),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True)
        )
        if dropout:
            self.conv_block.add_module("dropout", nn.Dropout(dropout))

    def forward(self, x, skip):
        x = self.up(x)  # 上采样
        x = torch.cat([x, skip], dim=1)  # 拼接跳跃连接
        return self.conv_block(x)  # 卷积融合特征


class CenterEnhancedGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # 第一阶段：深度可控的下采样
        self.down1 = UNetDown(in_channels, 64, normalize=False)  # 256->128
        self.down2 = UNetDown(64, 128)  # 128->64
        self.down3 = UNetDown(128, 256)  # 64->32
        self.down4 = UNetDown(256, 512)  # 32->16

        # 中心特征增强模块（解决扩散问题核心）
        self.center_block = nn.Sequential(
            ResBlock(512, dilation=2),  # 保持分辨率，扩大感受野
            SelfAttentionBlock(512),  # 中心区域自注意力
            ResBlock(512, dilation=3),
        )

        # 第二阶段：逐步上采样
        self.up4 = UNetUp(512, 256)  # 16->32
        self.up3 = UNetUp(256, 128)  # 32->64
        self.up2 = UNetUp(128, 64)  # 64->128
        self.up1 = UNetUp(64, 32)  # 128->256

        # 中心重建增强模块
        self.center_reconstruct = nn.Sequential(
            CenterConvBlock(32, 64, kernel_size=3),  # 中心特定卷积
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )

        # 最终输出层（多尺度特征融合）
        self.final = nn.Sequential(
            RefineBlock(64, out_channels),  # 特征细化层
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)  # 128
        d2 = self.down2(d1)  # 64
        d3 = self.down3(d2)  # 32
        d4 = self.down4(d3)  # 16

        # 中心特征增强
        center_feat = self.center_block(d4)

        # Decoder with skip connections
        u4 = self.up4(center_feat, d3)  # 32
        u3 = self.up3(u4, d2)  # 64
        u2 = self.up2(u3, d1)  # 128
        u1 = self.up1(u2, x)  # 256 (原始尺寸)

        # 中心区域重建
        center_output = self.center_reconstruct(u1)

        # 最终输出（融合全局和中心特征）
        return self.final(center_output)



if __name__ == '__main__':
    gen = CenterEnhancedGenerator(in_channels=6, out_channels=2)
    print(gen)
    v1 = torch.rand((5, 6, 256, 256))
    v2 = torch.rand((5, 2, 256, 256))
    v1_gen = gen.forward(v1)
    print(v1_gen.shape)