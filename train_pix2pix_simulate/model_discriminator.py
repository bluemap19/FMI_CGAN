##############################
#        Discriminator
##############################
from torch import nn, cat, rand, ones, zeros
from torchvision.models import vgg16, VGG16_Weights
from torch.cuda.amp import autocast

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        # print(img_A.shape, img_B.shape)
        img_input = cat((img_A, img_B), 1)
        return self.model(img_input)


class ModelVGGDiscriminator(nn.Module):
    def __init__(self):
        super(ModelVGGDiscriminator, self).__init__()

        # 神经网络感知模块
        self.style_net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
        for param in self.style_net.parameters():
            param.requires_grad = False

    def _content_match_loss(self, real, fake):
        """计算内容结构一致性损失"""
        real_features = self.style_net(real)
        fake_features = self.style_net(fake)
        return nn.functional.l1_loss(real_features, fake_features)

    def forward(self, img_A, img_B):
        content_loss = self._content_match_loss(img_A, img_B)

        return content_loss

class EnhancedDiscriminator(nn.Module):
    def __init__(self, in_channels=8, use_attention=True, spectral_norm=True):
        super(EnhancedDiscriminator, self).__init__()

        # 多尺度判别器结构 [7](@ref)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.discriminators = nn.ModuleList()

        # 构建3个不同分辨率的判别器
        for i in range(3):
            self.discriminators.append(
                self._make_discriminator(in_channels, spectral_norm, depth=4)
            )

    def _make_discriminator(self, in_channels, spectral_norm, depth=4):
        layers = []
        in_filters = in_channels
        out_filters = 32

        for i in range(depth):
            conv = nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)
            if spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            layers.append(conv)
            layers.append(nn.LeakyReLU(0.2))
            in_filters = out_filters
            out_filters = min(512, out_filters * 2)

        layers.append(nn.Conv2d(in_filters, 1, 3, padding=1))
        return nn.Sequential(*layers)


    def forward(self, img_A, img_B):
        # 拼接条件图像与生成图像
        fake_input = cat((img_A, img_B), 1)

        losses = []

        # 多尺度判别 [7](@ref)
        for i, disc in enumerate(self.discriminators):
            # 获取不同分辨率（256*256、128*128、64*64）下的disc(fake_input)特征分布
            if i > 0:
                fake_input = self.downsample(fake_input)
            pred = disc(fake_input)
            losses.append(pred)

        return losses


class SwitchableNorm2d(nn.Module):
    """可切换归一化层，自动适应不同特征分布"""

    def __init__(self, num_features):  # 添加num_features参数
        super().__init__()
        self.inn = nn.InstanceNorm2d(num_features)
        self.bn = nn.BatchNorm2d(num_features)
        self.ln = LayerNorm2d(num_features)
        self.weight = nn.Parameter(ones(3))
        self.eps = 1e-5

    def forward(self, x):
        inn = self.inn(x)
        bn = self.bn(x)
        ln = self.ln(x)

        weights = nn.functional.softmax(self.weight, 0)
        return weights[0] * inn + weights[1] * bn + weights[2] * ln


class EnhancedDiscriminator2(nn.Module):
    def __init__(self, in_channels=3, n_scales=3, use_spectral_norm=True,
                 use_switchable_norm=True, use_adaptive_attn=True):
        super().__init__()
        self.n_scales = n_scales
        self.discriminators = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # 初始化多尺度判别器
        base_channels = 32
        for i in range(n_scales):
            scale_factor = 2 ** i
            disc = ScaleDiscriminator(
                in_channels,
                base_channels * (2 ** i),
                scale_factor,
                use_spectral_norm,
                use_switchable_norm,
                use_adaptive_attn
            )
            self.discriminators.append(disc)

            if i < n_scales - 1:
                self.downsamples.append(
                    nn.Sequential(
                        nn.AvgPool2d(3, stride=2, padding=1),
                        nn.Conv2d(in_channels, in_channels, 3, padding=1)
                    )
                )

        # 特征融合模块
        self.fusion = FeatureFusionModule(base_channels * (2 ** n_scales), n_scales)

        # 多任务输出头
        self.adv_head = nn.Linear(512, 1)
        self.cls_head = nn.Linear(512, 10)  # 示例：10个类别
        self.reg_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 预测图像质量评分
        )

        # 高级归一化 - 仅在需要时创建
        self.sn = use_spectral_norm
        self.use_switchable_norm = use_switchable_norm

    def forward(self, x, y=None):
        features = []
        outputs = []

        # 多尺度处理
        for i in range(self.n_scales):
            if i > 0:
                x = self.downsamples[i - 1](x)

            feat, out = self.discriminators[i](x)
            features.append(feat)
            outputs.append(out)

            print(f'feature {i}', feat.shape)

        # 特征融合
        fused_feat = self.fusion(features)

        # 多任务预测
        with autocast():
            adv_out = self.adv_head(fused_feat)
            cls_out = self.cls_head(fused_feat) if y is not None else None
            reg_out = self.reg_head(fused_feat)

        return {
            'adv_outputs': outputs,
            'fused_feat': fused_feat,
            'adv_final': adv_out,
            'cls_output': cls_out,
            'quality_score': reg_out
        }


class ScaleDiscriminator(nn.Module):
    def __init__(self, in_channels, base_channels, scale_factor,
                 spectral_norm, switchable_norm, adaptive_attn):
        super().__init__()
        self.scale_factor = scale_factor
        self.blocks = nn.ModuleList()

        # 残差块序列
        for i in range(4):
            in_ch = in_channels if i == 0 else base_channels * (2 ** (i - 1))
            out_ch = base_channels * (2 ** i)

            block = ResidualBlock(
                in_ch, out_ch,
                downsample=(i > 0),
                spectral_norm=spectral_norm,
                switchable_norm=switchable_norm,
                use_attn=(i == 2) and adaptive_attn
            )
            self.blocks.append(block)

        # 动态注意力
        self.attn = DynamicAttentionGate(out_ch) if adaptive_attn else None

        # 特征压缩
        self.compression = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_ch, 128)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        if self.attn:
            x = self.attn(x)

        feat = x
        out = self.compression(x)
        return feat, out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample,
                 spectral_norm, switchable_norm, use_attn):
        super().__init__()
        stride = 2 if downsample else 1

        # 主分支
        conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        if spectral_norm:
            conv1 = nn.utils.spectral_norm(conv1)
            conv2 = nn.utils.spectral_norm(conv2)

        # 归一化层 - 根据参数选择
        if switchable_norm:
            norm1 = SwitchableNorm2d(out_channels)
            norm2 = SwitchableNorm2d(out_channels)
        else:
            norm1 = nn.InstanceNorm2d(out_channels)
            norm2 = nn.InstanceNorm2d(out_channels)

        self.conv = nn.Sequential(
            conv1,
            nn.LeakyReLU(0.2),
            norm1,
            conv2,
            nn.LeakyReLU(0.2),
            norm2
        )

        # 捷径分支
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or downsample:
            sc_layers = [nn.AvgPool2d(2)] if downsample else []
            sc_layers.append(nn.Conv2d(in_channels, out_channels, 1))
            self.shortcut = nn.Sequential(*sc_layers)

        # 动态注意力
        self.attn = DynamicAttentionGate(out_channels) if use_attn else None

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv(x)

        if self.attn:
            x = self.attn(x)

        return x + residual


class DynamicAttentionGate(nn.Module):
    """动态通道-空间注意力机制"""

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, 7, padding=3),
            nn.Sigmoid()
        )

        self.context_gate = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_att(x)

        # 空间注意力
        spatial_att = self.spatial_att(x)

        # 动态加权
        global_feat = nn.functional.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        context_weight = self.context_gate(global_feat).unsqueeze(-1).unsqueeze(-1)

        # 组合注意力
        att = context_weight * channel_att + (1 - context_weight) * spatial_att
        return x * att


class FeatureFusionModule(nn.Module):
    """多尺度特征融合模块"""

    def __init__(self, base_channels, n_scales):
        super().__init__()
        self.transformers = nn.ModuleList()

        for i in range(n_scales):
            self.transformers.append(
                nn.Sequential(
                    nn.Conv2d(base_channels // (2 ** i), 512, 1),
                    nn.ReLU()
                )
            )

        # 归一化层 - 根据参数选择
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(512 * n_scales, 512, 3, padding=1),
            nn.ReLU()
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, features):
        transformed = []
        for i, feat in enumerate(features):
            t = self.transformers[i](feat)
            print(f'layer {i} feature shape trans as:', feat.shape, t.shape)
            t = nn.functional.interpolate(t, scale_factor=2 ** i, mode='bilinear', align_corners=False)
            transformed.append(t)

        fused = cat(transformed, dim=1)
        fused = self.fusion_conv(fused)
        return self.global_pool(fused).view(fused.size(0), -1)


class LayerNorm2d(nn.Module):
    """2D层归一化"""

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(zeros(1, num_features, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.weight + self.bias



if __name__ == '__main__':
    v1 = rand((5, 2, 256, 256))
    v2 = rand((5, 8, 256, 256))

    # dis_sim = Discriminator(in_channels=10)
    # print(dis_sim)
    # v1_dis_sim = dis_sim.forward(v1, v2)
    # print(v1_dis_sim.shape)

    dis_enh = EnhancedDiscriminator(in_channels=10)
    loss_1 = dis_enh.forward(v1, v2)
    print(loss_1[0].shape, loss_1[1].shape, loss_1[2].shape)

    # dis_MVGG = ModelVGGDiscriminator()
    # v1_dis_MVGG = dis_MVGG.forward(v1, v2)
    # print(v1_dis_MVGG)

    # dis_enh2 = EnhancedDiscriminator2(in_channels=2)
    # dis_enh2_1 = dis_enh2.forward(v1, v2)
    # # 'adv_outputs': outputs,
    # # 'fused_feat': fused_feat,
    # # 'adv_final': adv_out,
    # # 'cls_output': cls_out,
    # # 'quality_score': reg_out
    # print(dis_enh2_1[0].shape, dis_enh2_1[1].shape, dis_enh2_1[2].shape, dis_enh2_1[3].shape)

