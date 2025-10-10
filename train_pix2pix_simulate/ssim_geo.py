import torch
import torch.nn.functional as F
from torch import nn, rand


class GeologicalSSIM(nn.Module):
    """
    地质图像专用SSIM损失
    结合地质特征增强
    """

    def __init__(self, window_size=11, channel=1,
                 texture_weight=0.5, structure_weight=0.5):
        super().__init__()
        self.window_size = window_size  # 滑动窗口大小
        self.channel = channel  # 输入图像的通道数
        self.texture_weight = texture_weight  # 纹理分量权重
        self.structure_weight = structure_weight  # 结构分量权重

        # 创建高斯窗口并注册为模型缓冲区
        self.register_buffer("window", self._create_window(window_size, channel))

    def _create_window(self, window_size, channel):
        """创建地质优化高斯窗"""
        # 确保窗口为奇数（便于中心对称）
        if window_size % 2 == 0:
            window_size += 1

        # 计算真实中心点（确保对称性）
        half_size = (window_size - 1) / 2.0
        # 创建从中心向两侧延伸的坐标序列
        x = torch.arange(window_size) - half_size
        # 计算一维高斯分布（σ=1.5）
        gauss = torch.exp(-x.pow(2) / 4.5)  # 4.5 = 2*σ² (σ=1.5)
        gauss = gauss / gauss.sum()  # 归一化

        # 创建2D窗口（通过外积）
        window_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        window_2d = window_2d / window_2d.sum()  # 归一化

        # 扩展为4D张量 [channel, 1, H, W]
        return window_2d.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)

    def forward(self, img1, img2):
        # 自动确定动态范围（图像的最大值-最小值）
        max_val = torch.max(torch.max(img1), torch.max(img2))
        min_val = torch.min(torch.min(img1), torch.min(img2))
        L = max_val - min_val  # 动态范围

        padding = self.window_size // 2  # 卷积填充量

        # 计算局部均值（使用高斯卷积）
        mu1 = F.conv2d(img1, self.window, padding=padding, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=padding, groups=self.channel)

        # 均值平方
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2  # 均值乘积

        # 计算局部方差和协方差
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=padding, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=padding, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=padding, groups=self.channel) - mu1_mu2

        # 地质特征增强常数
        C1 = (0.02 * L) ** 2  # 降低亮度敏感度（地质图像亮度变化大）
        C2 = (0.08 * L) ** 2  # 提高对比度敏感度（强调地质纹理）

        # 纹理分量 (对比度+结构)
        texture_num = 2 * sigma12 + C2
        texture_den = sigma1_sq + sigma2_sq + C2
        texture_map = texture_num / texture_den

        # 结构分量 (亮度)
        structure_num = 2 * mu1_mu2 + C1
        structure_den = mu1_sq + mu2_sq + C1
        structure_map = structure_num / structure_den

        # 地质优化组合（加权求和）
        ssim_map = (self.texture_weight * texture_map +
                    self.structure_weight * structure_map)

        # 返回1-SSIM作为损失（越小表示越相似）
        return 1 - ssim_map.mean()
# class GeologicalSSIM(nn.Module):
#     """
#     地质图像专用SSIM损失
#     结合地质特征增强
#     """
#
#     def __init__(self, window_size=11, channel=1,
#                  texture_weight=0.5, structure_weight=0.5):
#         super().__init__()
#         self.window_size = window_size
#         self.channel = channel
#         self.texture_weight = texture_weight
#         self.structure_weight = structure_weight
#
#         # 创建高斯窗口
#         self.register_buffer("window", self._create_window(window_size, channel))
#
#     def _create_window(self, window_size, channel):
#         """创建地质优化高斯窗"""
#         # 确保窗口为奇数
#         if window_size % 2 == 0:
#             window_size += 1
#
#         # 计算真实中心点
#         half_size = (window_size - 1) / 2.0
#         x = torch.arange(window_size) - half_size
#         gauss = torch.exp(-x.pow(2) / 4.5)  # sigma=1.5
#         gauss = gauss / gauss.sum()
#
#         # 创建2D窗口
#         window_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
#         window_2d = window_2d / window_2d.sum()
#
#         # 扩展为4D张量
#         return window_2d.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)
#
#     def forward(self, img1, img2):
#         # 自动确定动态范围
#         max_val = torch.max(torch.max(img1), torch.max(img2))
#         min_val = torch.min(torch.min(img1), torch.min(img2))
#         L = max_val - min_val
#
#         padding = self.window_size // 2
#
#         # 计算局部统计量
#         mu1 = F.conv2d(img1, self.window, padding=padding, groups=self.channel)
#         mu2 = F.conv2d(img2, self.window, padding=padding, groups=self.channel)
#
#         mu1_sq = mu1.pow(2)
#         mu2_sq = mu2.pow(2)
#         mu1_mu2 = mu1 * mu2
#
#         sigma1_sq = F.conv2d(img1 * img1, self.window, padding=padding, groups=self.channel) - mu1_sq
#         sigma2_sq = F.conv2d(img2 * img2, self.window, padding=padding, groups=self.channel) - mu2_sq
#         sigma12 = F.conv2d(img1 * img2, self.window, padding=padding, groups=self.channel) - mu1_mu2
#
#         # 地质特征增强常数
#         C1 = (0.02 * L) ** 2  # 降低亮度敏感度
#         C2 = (0.08 * L) ** 2  # 提高对比度敏感度
#
#         # 纹理分量 (对比度+结构)
#         texture_num = 2 * sigma12 + C2
#         texture_den = sigma1_sq + sigma2_sq + C2
#         texture_map = texture_num / texture_den
#
#         # 结构分量 (亮度)
#         structure_num = 2 * mu1_mu2 + C1
#         structure_den = mu1_sq + mu2_sq + C1
#         structure_map = structure_num / structure_den
#
#         # 地质优化组合
#         ssim_map = (self.texture_weight * texture_map +
#                     self.structure_weight * structure_map)
#
#         return 1 - ssim_map.mean()

if __name__ == '__main__':
    criterion = GeologicalSSIM(window_size=15, channel=2)

    v1 = rand((5, 2, 256, 256))
    v2 = rand((5, 2, 256, 256))
    loss_1 = criterion.forward(v1, v2)

    print(loss_1)