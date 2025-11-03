import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from FMI_Para_calucate.loss_cal import MultiTaskLoss


class ViTAttention(nn.Module):
    """
    Vision Transformer自注意力机制

    原理:
    自注意力机制允许模型在处理每个位置时关注输入序列中的所有位置，
    从而捕捉长距离依赖关系。这是Transformer架构的核心组件。

    数学公式:
    Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

    其中:
    - Q: 查询矩阵 (Query)
    - K: 键矩阵 (Key)
    - V: 值矩阵 (Value)
    - d_k: 键向量的维度，用于缩放点积结果

    工作流程:
    1. 将输入线性投影为Q、K、V三个矩阵
    2. 计算Q和K的点积并缩放
    3. 应用softmax获取注意力权重
    4. 使用注意力权重加权求和V矩阵
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        """
        初始化自注意力机制

        参数:
        dim: 输入特征维度
        heads: 注意力头的数量（多头注意力）
        dim_head: 每个注意力头的维度
        dropout: Dropout比率，用于防止过拟合
        """
        super().__init__()
        # 计算内部维度：头数 × 每个头的维度
        inner_dim = dim_head * heads

        # 判断是否需要投影输出（当头数为1且头维度等于输入维度时不需要）
        project_out = not (heads == 1 and dim_head == dim)

        # 设置参数
        self.heads = heads  # 注意力头数量
        self.scale = dim_head ** -0.5  # 缩放因子，用于稳定softmax计算

        # 注意力计算组件
        self.attend = nn.Softmax(dim=-1)  # 在最后一个维度应用softmax
        self.dropout = nn.Dropout(dropout)  # Dropout层

        # 将输入投影到Q、K、V矩阵的线性层
        # 一次投影生成3个矩阵，因此输出维度是inner_dim的3倍
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # 输出投影层（如果需要）
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 将多头输出投影回原始维度
            nn.Dropout(dropout)  # 输出Dropout
        ) if project_out else nn.Identity()  # 如果不需要投影，使用恒等映射

    def forward(self, x):
        """
        前向传播

        参数:
        x: 输入张量 [batch_size, num_patches, dim]

        返回:
        应用自注意力后的输出 [batch_size, num_patches, dim]
        """
        # 1. 通过线性层生成Q、K、V矩阵
        # qkv形状: [batch_size, num_patches, inner_dim * 3]
        qkv = self.to_qkv(x)

        # 2. 将qkv分割为Q、K、V三个部分
        # 每个部分的形状: [batch_size, num_patches, inner_dim]
        qkv = qkv.chunk(3, dim=-1)

        # 3. 重排列维度，为多头注意力做准备
        # 从 [batch_size, num_patches, inner_dim]
        # 变为 [batch_size, heads, num_patches, dim_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 4. 计算Q和K的点积并缩放
        # dots形状: [batch_size, heads, num_patches, num_patches]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 5. 计算注意力权重
        attn = self.attend(dots)  # softmax归一化
        attn = self.dropout(attn)  # 应用Dropout

        # 6. 使用注意力权重加权求和V矩阵
        # out形状: [batch_size, heads, num_patches, dim_head]
        out = torch.matmul(attn, v)

        # 7. 重排列维度，合并多头输出
        # 从 [batch_size, heads, num_patches, dim_head]
        # 变为 [batch_size, num_patches, heads * dim_head]
        out = rearrange(out, 'b h n d -> b n (h d)')

        # 8. 应用输出投影（如果需要）
        return self.to_out(out)

# class ViTAttention(nn.Module):
#     """Vision Transformer自注意力机制"""
#
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         project_out = not (heads == 1 and dim_head == dim)
#
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#
#         self.attend = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)
#
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
#
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
#
#     def forward(self, x):
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
#
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#
#         attn = self.attend(dots)
#         attn = self.dropout(attn)
#
#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)


class ViTBlock(nn.Module):
    """
    Vision Transformer块

    原理:
    Transformer块由多头自注意力机制和前馈神经网络组成，
    使用残差连接和层归一化来促进梯度流动和稳定训练。

    结构:
    LayerNorm -> MultiHeadAttention -> ResidualConnection ->
    LayerNorm -> FeedForward -> ResidualConnection

    残差连接帮助缓解深度网络中的梯度消失问题，
    层归一化提高训练稳定性。
    """
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=512, dropout=0.):
        """
        初始化Transformer块

        参数:
        dim: 输入特征维度
        heads: 注意力头数量
        dim_head: 每个注意力头的维度
        mlp_dim: 前馈网络隐藏层维度
        dropout: Dropout比率
        """
        super().__init__()
        # 第一个层归一化（注意力前）
        self.norm1 = nn.LayerNorm(dim)

        # 自注意力机制
        self.attn = ViTAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)

        # 第二个层归一化（前馈网络前）
        self.norm2 = nn.LayerNorm(dim)

        # 前馈网络（多层感知机）
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),  # 扩展维度
            nn.GELU(),  # GELU激活函数（比ReLU更平滑）
            nn.Dropout(dropout),  # Dropout
            nn.Linear(mlp_dim, dim),  # 投影回原始维度
            nn.Dropout(dropout)  # Dropout
        )

    def forward(self, x):
        """
        前向传播
        参数:
        x: 输入张量 [batch_size, num_patches, dim]
        返回:
        经过Transformer块处理后的输出 [batch_size, num_patches, dim]
        """
        # 1. 第一个残差连接：层归一化 -> 自注意力 -> 残差连接
        # 公式: x = x + Attention(LayerNorm(x))
        x = x + self.attn(self.norm1(x))

        # 2. 第二个残差连接：层归一化 -> 前馈网络 -> 残差连接
        # 公式: x = x + FFN(LayerNorm(x))
        x = x + self.ff(self.norm2(x))

        return x


# class ViTBlock(nn.Module):
#     """Vision Transformer块"""
#
#     def __init__(self, dim, heads=8, dim_head=64, mlp_dim=512, dropout=0.):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = ViTAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
#         self.norm2 = nn.LayerNorm(dim)
#         self.ff = nn.Sequential(
#             nn.Linear(dim, mlp_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(mlp_dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.ff(self.norm2(x))
#         return x

class ViT(nn.Module):
    """
    Vision Transformer模块，用于特征图上的自注意力

    原理:
    将2D特征图转换为1D序列，添加位置编码，通过多个Transformer块处理，
    最后还原为2D特征图格式。

    与传统ViT的区别:
    传统ViT将图像分割为patch并直接分类，而这里将ViT用作特征提取器，
    处理中间特征图并保持空间结构。

    工作流程:
    1. 将特征图分割为patch并展平为序列
    2. 线性投影到高维空间
    3. 添加位置编码
    4. 通过多个Transformer块
    5. 还原为特征图格式
    """
    def __init__(self, in_channels, patch_size=8, dim=512, depth=3, heads=8, dim_head=64, mlp_dim=1024, dropout=0.1):
        """
        初始化ViT模块

        参数:
        in_channels: 输入特征图的通道数
        patch_size: 将特征图分割为patch的大小
        dim: Transformer的隐藏维度
        depth: Transformer块的层数
        heads: 注意力头数量
        dim_head: 每个注意力头的维度
        mlp_dim: 前馈网络隐藏层维度
        dropout: Dropout比率
        """
        super().__init__()
        self.patch_size = patch_size  # patch大小
        self.dim = dim  # 隐藏维度

        # 将特征图转换为patch序列的模块
        self.to_patch_embedding = nn.Sequential(
            # 重排列操作: 将特征图分割为patch并展平
            # 从 [batch, channels, height, width]
            # 变为 [batch, num_patches, patch_size * patch_size * channels]
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),

            # 线性投影: 将每个patch投影到高维空间
            nn.Linear(patch_size * patch_size * in_channels, dim),
        )

        # 位置编码（动态创建，以适应不同大小的输入）
        self.pos_embedding = None

        # 创建多个Transformer块
        self.transformer = nn.Sequential(*[
            ViTBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
        ])

        # 将序列还原为特征图的模块
        self.to_feature_map = nn.Sequential(
            # 线性投影: 将每个patch投影回原始维度
            nn.Linear(dim, patch_size * patch_size * in_channels),
        )

    def forward(self, x):
        """
        前向传播
        参数: x: 输入特征图 [batch_size, channels, height, width]
        返回: 处理后的特征图 [batch_size, channels, height, width]
        """
        # 保存原始尺寸
        b, c, h, w = x.shape

        # 1. 将特征图转换为patch序列
        # 输出形状: [batch_size, num_patches, dim]
        x = self.to_patch_embedding(x)

        # 2. 动态创建位置编码
        num_patches = x.shape[1]  # 计算patch数量
        if self.pos_embedding is None or self.pos_embedding.shape[1] != num_patches:
            # 创建与当前patch数量匹配的位置编码
            # 形状: [1, num_patches, dim]
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim)).to(x.device)

        # 3. 添加位置编码（为每个patch添加位置信息）
        x += self.pos_embedding

        # 4. 通过Transformer块序列
        x = self.transformer(x)

        # 5. 还原为特征图格式
        x = self.to_feature_map(x)  # 形状: [batch_size, num_patches, patch_size*patch_size*channels]

        # 重排列操作: 将序列还原为特征图
        # 从 [batch_size, num_patches, patch_size*patch_size*channels]
        # 变为 [batch_size, channels, height, width]
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                      h=h // self.patch_size, w=w // self.patch_size,
                      p1=self.patch_size, p2=self.patch_size, c=c)

        return x

# class ViT(nn.Module):
#     """Vision Transformer模块，用于特征图上的自注意力"""
#     def __init__(self, in_channels, patch_size=8, dim=512, depth=3, heads=8, dim_head=64, mlp_dim=1024, dropout=0.1):
#         super().__init__()
#         self.patch_size = patch_size
#         self.dim = dim
#
#         # 将特征图转换为patch序列
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
#             nn.Linear(patch_size * patch_size * in_channels, dim),
#         )
#
#         # 位置编码
#         # self.pos_embedding = nn.Parameter(torch.randn(1, (64 // patch_size) ** 2, dim))
#         self.pos_embedding = None  # 将在forward中动态创建
#
#         # Transformer层
#         self.transformer = nn.Sequential(*[
#             ViTBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
#         ])
#
#         # 还原到特征图
#         self.to_feature_map = nn.Sequential(nn.Linear(dim, patch_size * patch_size * in_channels),)
#
#     def forward(self, x):
#         # 保存原始尺寸
#         b, c, h, w = x.shape
#
#         # 转换为patch序列
#         x = self.to_patch_embedding(x)
#
#         # 动态创建位置编码，确保与patch数量匹配
#         num_patches = x.shape[1]
#         if self.pos_embedding is None or self.pos_embedding.shape[1] != num_patches:
#             # 动态创建位置编码
#             self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim)).to(x.device)
#
#         # 添加位置编码
#         x += self.pos_embedding
#
#         # 通过Transformer
#         x = self.transformer(x)
#
#         # 还原为特征图
#         x = self.to_feature_map(x)
#         x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
#                       h=h // self.patch_size, w=w // self.patch_size,
#                       p1=self.patch_size, p2=self.patch_size, c=c)
#
#         return x


class ResidualBlock(nn.Module):
    """改进的残差块，增强特征传播能力"""

    def __init__(self, channels, use_vit_attention=False, vit_patch_size=8):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

        # 添加ViT注意力机制
        self.use_vit_attention = use_vit_attention
        if use_vit_attention:
            self.vit_attention = ViT(channels, patch_size=vit_patch_size, dim=channels)

    def forward(self, x):
        residual = x
        out = self.conv_block(x)

        # 应用ViT注意力
        if self.use_vit_attention:
            out = self.vit_attention(out)

        out += residual
        return self.relu(out)


class DownBlock(nn.Module):
    """下采样块 - 使用步长为2的卷积"""

    def __init__(self, in_channels, out_channels, use_vit_attention=False, vit_patch_size=8):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(out_channels, use_vit_attention, vit_patch_size)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UpBlock(nn.Module):
    """上采样块 - 使用PixelShuffle"""

    def __init__(self, in_channels, skip_channels, out_channels, use_vit_attention=False, vit_patch_size=8):
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
            ResidualBlock(out_channels, use_vit_attention, vit_patch_size)
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


class ParameterRegressionHead(nn.Module):
    """
        简单高效的孔洞缝参数回归头

        设计原则:
        1. 简单性: 减少不必要的层和复杂性
        2. 高效性: 使用最少的参数实现功能
        3. 稳定性: 确保输出合理的参数范围
        4. 可解释性: 结构清晰，易于理解和调试
        """

    def __init__(self, in_channels=512, num_params=7, hidden_dim=[256, 128]):
        """
        初始化简单参数回归头

        参数:
        in_channels: 输入特征通道数
        num_params: 参数数量 (7个孔洞缝参数)
        hidden_dim: 隐藏层维度 (平衡表达能力和效率)
        """
        super().__init__()
        self.num_params = num_params

        # 全局特征提取 - 使用自适应平均池化获取全局信息
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 简单的两层MLP - 平衡表达能力和效率
        self.mlp = nn.Sequential(
            # 第一层: 特征变换
            nn.Linear(in_channels, hidden_dim[0]),
            nn.BatchNorm1d(hidden_dim[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # 轻微正则化防止过拟合

            # 第一层: 特征变换
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.BatchNorm1d(hidden_dim[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # 轻微正则化防止过拟合

            # 第二层: 参数回归
            nn.Linear(hidden_dim[1], num_params),

            # 输出激活: 确保参数在合理范围内
            nn.Softplus()  # 输出正值，适合物理参数
        )

        # 参数初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """简单的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He初始化，适合ReLU激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)  # 小的正偏置避免死神经元
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        参数:
        x: 输入特征图 [B, C, H, W]
        返回:
        params: 参数预测 [B, 7]
        """
        # 全局特征提取
        x = self.global_pool(x)  # [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # [B, C]

        # 通过简单MLP回归参数
        params = self.mlp(x)  # [B, 7]

        return params


class GeneratorUNetImproved(nn.Module):
    """改进的U-Net生成器，支持ViT注意力和参数回归"""

    def __init__(self, in_channels=2, out_channels=2, num_params=7,
                 use_vit_attention=False, vit_patch_size=8):
        """
        参数:
        in_channels: 输入通道数 (默认为2: 动态+静态图像)
        out_channels: 输出通道数 (默认为2: 裂缝+孔洞掩码)
        num_params: 参数数量 (默认为7个孔洞缝参数)
        use_vit_attention: 是否使用ViT注意力机制
        vit_patch_size: ViT的patch大小
        """
        super().__init__()
        self.use_vit_attention = use_vit_attention
        self.vit_patch_size = vit_patch_size

        # 初始卷积层
        self.branch_init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 下采样路径
        self.branch_down1 = DownBlock(64, 128)  # 128x128
        self.branch_down2 = DownBlock(128, 256)  # 64x64
        self.branch_down3 = DownBlock(256, 512)  # 32x32
        self.branch_down4 = DownBlock(512, 512, use_vit_attention, vit_patch_size)  # 16x16
        self.branch_down5 = DownBlock(512, 512, use_vit_attention, vit_patch_size)  # 8x8

        bottleneck_layers = []
        # 添加ViT注意力到瓶颈层
        if use_vit_attention:
            # 使用更深的ViT结构
            bottleneck_layers.append(ViT(512, patch_size=vit_patch_size, dim=512, depth=4))
        else:
            # 使用常规卷积作为备选
            bottleneck_layers.extend([
                nn.Conv2d(512, 512, 3, padding=1),
                nn.InstanceNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.InstanceNorm2d(512),
                nn.ReLU(inplace=True)
            ])

        self.branch_bottleneck = nn.Sequential(*bottleneck_layers)

        # 参数回归分支（在瓶颈层后添加）
        self.branch_regression = ParameterRegressionHead(
            in_channels = 512,
            num_params = num_params,
            hidden_dim = [256, 128],
        )

        # 上采样路径
        self.segment_up1 = UpBlock(512, 512, 512, use_vit_attention, vit_patch_size)  # 16x16
        self.segment_up2 = UpBlock(512, 512, 512, use_vit_attention, vit_patch_size)  # 32x32
        self.segment_up3 = UpBlock(512, 256, 256)  # 64x64
        self.segment_up4 = UpBlock(256, 128, 128)  # 128x128
        self.segment_up5 = UpBlock(128, 64, 64)  # 256x256

        # 最终输出层
        self.segment_final = nn.Sequential(
            nn.Conv2d(64, out_channels, 7, padding=3),
            nn.Sigmoid(),  # 输出范围[0,1]
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """ 初始化权重 """
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

    def freeze_backbone(self):
        """ 冻结主干网络(下采样部分＋参数回归部分) """
        for name, param in self.named_parameters():
            if 'branch' in name:
                param.requires_grad = False
        print("主干网络已冻结")

    def freeze_segbone(self):
        """ 冻结分割(上采样部分) """
        for name, param in self.named_parameters():
            if 'segment_' in name:
                param.requires_grad = False
        print("分割网络已冻结")

    def freeze_unet(self):
        """冻结UNet，保留孔洞缝参数计算头"""
        for name, param in self.named_parameters():
            if 'branch_regression' not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        """解冻主干网络"""
        for param in self.parameters():
            param.requires_grad = True
        print("主干网络已解冻")

    def forward(self, x):
        """
        前向传播
        返回:
        - segmentation: 分割结果 [B, 2, H, W]
        - parameters: 参数预测 [B, 7]
        """
        # 初始卷积
        x0 = self.branch_init_conv(x)

        # 下采样路径
        d1 = self.branch_down1(x0)  # 128x128
        d2 = self.branch_down2(d1)  # 64x64
        d3 = self.branch_down3(d2)  # 32x32
        d4 = self.branch_down4(d3)  # 16x16
        d5 = self.branch_down5(d4)  # 8x8

        # 瓶颈层
        b = self.branch_bottleneck(d5)

        # 参数提取（在瓶颈层特征上）
        params = self.branch_regression(b)

        # 上采样路径
        u1 = self.segment_up1(b, d4)  # 16x16
        u2 = self.segment_up2(u1, d3)  # 32x32
        u3 = self.segment_up3(u2, d2)  # 64x64
        u4 = self.segment_up4(u3, d1)  # 128x128
        u5 = self.segment_up5(u4, x0)  # 256x256

        # 最终分割输出
        segmentation = self.segment_final(u5)

        # 返回结果
        return {
            'fmi_mask': segmentation,  # 分割掩码
            'fmi_params': params  # 孔洞缝参数
        }


if __name__ == '__main__':
    # 测试设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    image_length = 160
    VIT_batch_size = image_length//32
    data_num = 7

    # 测试不同配置的模型
    configs = [
        {'use_vit_attention': False, 'name': '基础模型+参数分支'},
        {'use_vit_attention': True, 'name': 'ViT注意力+参数分支'},
        {'use_vit_attention': False, 'name': '纯分割模型'},
    ]

    for config in configs:
        print(f"\n=== 测试配置: {config['name']} ===")

        # 创建模型
        gen = GeneratorUNetImproved(
            in_channels=2,
            out_channels=2,
            num_params=7,
            use_vit_attention=config['use_vit_attention'],
            vit_patch_size=VIT_batch_size,
        ).to(device)

        # 计算参数量
        total_params = sum(p.numel() for p in gen.parameters())
        trainable_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
        print(f"总参数量: {total_params / 1e6:.2f}M")
        print(f"可训练参数量: {trainable_params / 1e6:.2f}M")

        # 测试输入输出
        input_tensor = torch.randn((data_num, 2, image_length, image_length)).to(device)
        output = gen(input_tensor)

        print(f"分割输出形状: {output['fmi_mask'].shape}")
        print(f"参数输出形状: {output['fmi_params'].shape}")
        print(f"参数范围: [{output['fmi_params'].min().item():.4f}, {output['fmi_params'].max().item():.4f}]")

        # 测试多任务损失（如果使用参数分支）
        # 创建模拟目标
        target_seg = torch.randn_like(output['fmi_mask']).sigmoid()  # 模拟分割目标
        target_params = torch.randn((input_tensor.shape[0], 7)).to(device)  # 模拟参数目标
        test_origin = {'fmi_mask': target_seg, 'fmi_params': target_params}

        # 计算损失
        criterion = MultiTaskLoss(seg_weight=1.0, param_weight=0.1)
        loss_dict = criterion(output, test_origin)

        print(f"总损失: {loss_dict['total_loss'].item():.4f}")
        print(f"分割损失: {loss_dict['seg_loss'].item():.4f}")
        print(f"参数损失: {loss_dict['param_loss'].item():.4f}")

        # 测试梯度反向传播
        loss = loss_dict['total_loss']

        loss.backward()
        print("梯度反向传播测试成功")

        # 测试冻结和解冻功能
        print("测试冻结功能...")
        gen.freeze_backbone()

        # 检查参数是否冻结
        for name, param in gen.named_parameters():
            if 'branch' in name:
                assert not param.requires_grad, f"参数 {name} 应该被冻结"
            else:
                assert param.requires_grad, f"参数 {name} 应该可训练"

        print("测试解冻功能...")
        gen.unfreeze_all()

        # 检查参数是否解冻
        for param in gen.parameters():
            assert param.requires_grad, "所有参数应该可训练"

        print("冻结和解冻功能测试成功")
