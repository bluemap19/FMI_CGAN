import torch
import torch.nn.functional as F
from torch import nn, rand


class GeoSpatialSSIM(nn.Module):
    """
    地质图像优化的SSIM损失函数
    改进点：
    1. 自动适应动态范围
    2. 地质特征敏感权重
    3. 多通道差异化处理
    4. 奇数窗口大小保证
    """

    def __init__(self, window_size=11, channel=1, sigma=1.5,
                 spatial_weights=None, val_range=None):
        super().__init__()
        # 确保窗口为奇数
        window_size = window_size if window_size % 2 == 1 else window_size + 1

        self.window_size = window_size
        self.channel = channel
        self.sigma = sigma
        self.val_range = val_range

        # 地质特征敏感权重 (可自定义)
        self.spatial_weights = spatial_weights

        # 创建高斯窗口
        self.register_buffer("window", self._create_window(window_size, channel, sigma))

    def _create_window(self, window_size, channel, sigma):
        """创建地质优化高斯窗"""
        # 计算真实中心点 (考虑浮点中心)
        half_size = (window_size - 1) / 2.0
        x = torch.arange(window_size) - half_size
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()

        # 创建2D窗口
        window_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        window_2d = window_2d / window_2d.sum()

        # 扩展为4D张量 [channel, 1, H, W]
        return window_2d.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)

    def forward(self, img1, img2, mask=None):
        """
        计算地质SSIM损失

        参数:
        img1: 生成图像 [B,C,H,W]
        img2: 目标图像 [B,C,H,W]
        mask: 可选区域掩码 [B,1,H,W]

        返回:
        ssim_loss: 平均SSIM损失
        """
        # 自动确定动态范围
        if self.val_range is None:
            max_val = torch.max(torch.max(img1), torch.max(img2))
            min_val = torch.min(torch.min(img1), torch.min(img2))
            L = max_val - min_val
        else:
            L = self.val_range

        # 填充大小
        padding = self.window_size // 2

        # 计算局部统计量
        mu1 = F.conv2d(img1, self.window, padding=padding, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=padding, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=padding, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=padding, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=padding, groups=self.channel) - mu1_mu2

        # SSIM常数
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        # 地质特征增强常数
        C3 = C2 * 0.5  # 增强纹理对比度

        # SSIM计算
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2 + C3))

        # 应用地质空间权重
        if self.spatial_weights is not None:
            ssim_map = ssim_map * self.spatial_weights

        # 应用掩码
        if mask is not None:
            ssim_map = ssim_map * mask

        return 1 - ssim_map.mean()


class GeoSpatialMSSSIM(nn.Module):
    """
    地质图像多尺度SSIM损失
    改进点：
    1. 地质特征优化权重
    2. 自适应尺度选择
    3. 通道重要性加权
    """

    def __init__(self, channel=1, window_size=11, sigma=1.5,
                 val_range=None, scales=5, channel_weights=None):
        super().__init__()
        self.scales = scales
        self.channel = channel
        self.channel_weights = channel_weights or torch.ones(channel)
        self.ssim_modules = nn.ModuleList([
            GeoSpatialSSIM(window_size, channel, sigma, val_range=val_range)
            for _ in range(scales)
        ])

        # 地质特征优化权重 (低频权重更高)
        self.scale_weights = torch.linspace(0.8, 0.2, scales)
        self.scale_weights = self.scale_weights / self.scale_weights.sum()

    def forward(self, img1, img2):
        """
        计算多尺度地质SSIM损失

        参数:
        img1: 生成图像 [B,C,H,W]
        img2: 目标图像 [B,C,H,W]

        返回:
        msssim_loss: 多尺度SSIM损失
        """
        total_loss = 0
        current_img1 = img1
        current_img2 = img2

        for i in range(self.scales):
            # 计算当前尺度损失
            scale_loss = self.ssim_modules[i](current_img1, current_img2)

            # 通道加权
            if self.channel_weights is not None:
                scale_loss = scale_loss * self.channel_weights[i % self.channel]

            # 尺度加权
            total_loss += self.scale_weights[i] * scale_loss

            # 下采样准备下一尺度
            if i < self.scales - 1:
                current_img1 = F.avg_pool2d(current_img1, kernel_size=2)
                current_img2 = F.avg_pool2d(current_img2, kernel_size=2)

        return total_loss


class GeologicalSSIM(nn.Module):
    """
    地质图像专用SSIM损失
    结合地质特征增强
    """

    def __init__(self, window_size=11, channel=1,
                 texture_weight=0.7, structure_weight=0.3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.texture_weight = texture_weight
        self.structure_weight = structure_weight

        # 创建高斯窗口
        self.register_buffer("window", self._create_window(window_size, channel))

    def _create_window(self, window_size, channel):
        """创建地质优化高斯窗"""
        # 确保窗口为奇数
        if window_size % 2 == 0:
            window_size += 1

        # 计算真实中心点
        half_size = (window_size - 1) / 2.0
        x = torch.arange(window_size) - half_size
        gauss = torch.exp(-x.pow(2) / 4.5)  # sigma=1.5
        gauss = gauss / gauss.sum()

        # 创建2D窗口
        window_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        window_2d = window_2d / window_2d.sum()

        # 扩展为4D张量
        return window_2d.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)

    def forward(self, img1, img2):
        # 自动确定动态范围
        max_val = torch.max(torch.max(img1), torch.max(img2))
        min_val = torch.min(torch.min(img1), torch.min(img2))
        L = max_val - min_val

        padding = self.window_size // 2

        # 计算局部统计量
        mu1 = F.conv2d(img1, self.window, padding=padding, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=padding, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=padding, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=padding, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=padding, groups=self.channel) - mu1_mu2

        # 地质特征增强常数
        C1 = (0.02 * L) ** 2  # 降低亮度敏感度
        C2 = (0.08 * L) ** 2  # 提高对比度敏感度

        # 纹理分量 (对比度+结构)
        texture_num = 2 * sigma12 + C2
        texture_den = sigma1_sq + sigma2_sq + C2
        texture_map = texture_num / texture_den

        # 结构分量 (亮度)
        structure_num = 2 * mu1_mu2 + C1
        structure_den = mu1_sq + mu2_sq + C1
        structure_map = structure_num / structure_den

        # 地质优化组合
        ssim_map = (self.texture_weight * texture_map +
                    self.structure_weight * structure_map)

        return 1 - ssim_map.mean()

if __name__ == '__main__':
    criterion = GeologicalSSIM(window_size=15, channel=2)

    v1 = rand((5, 2, 256, 256))
    v2 = rand((5, 2, 256, 256))
    loss_1 = criterion.forward(v1, v2)

    print(loss_1)