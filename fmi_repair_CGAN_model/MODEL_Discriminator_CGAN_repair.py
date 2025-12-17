from torch import nn, cat


class EnhancedDiscriminator(nn.Module):
    def __init__(self, in_channels=8, spectral_norm=True):
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