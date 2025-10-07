import torch.nn as nn
import torch.nn.functional as F
import torch


class FMI_Generator(nn.Module):
    def __init__(self, channels_in=1, channels_out=2):
        super(FMI_Generator, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 3, stride=1, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 3, stride=1, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        self.model = nn.Sequential(
            *downsample(channels_in, 64, normalize=False),
            *downsample(64, 128),
            *downsample(128, 256),
            *downsample(256, 512),
            *downsample(512, 1024),
            nn.Conv2d(1024, 1024, 1),
            *upsample(1024, 512),
            *upsample(512, 256),
            *upsample(256, 128),
            *upsample(128, 64),
            # in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1
            nn.Conv2d(64, channels_out, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    FG = FMI_Generator(channels_in=3, channels_out=2).to(device)
    print(FG)
    t_r = torch.randn((12, 3, 256, 256)).to(device)

    print(FG(t_r).shape)