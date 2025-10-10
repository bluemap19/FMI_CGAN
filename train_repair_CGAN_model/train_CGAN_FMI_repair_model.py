import argparse
import os
import numpy as np
import time
import datetime
import sys
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from train_pix2pix_simulate.datasets import ImageDataset_FMI
from train_pix2pix_simulate.model_discriminator import EnhancedDiscriminator
from train_pix2pix_simulate.models_generator import weights_init_normal, GeneratorUNet
from train_pix2pix_simulate.ssim_geo import GeologicalSSIM
from train_pix2pix_simulate.ssim_loss import MSSSIM

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
# parser.add_argument("--dataset_path", type=str, default="FMI_GAN", help="name of the dataset")

parser.add_argument("--img_size_x", type=int, default=256, help="size of image height")
parser.add_argument("--img_size_y", type=int, default=256, help="size of image width")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--dataset_path_val", type=str, default=r"F:\DeepLData\target_stage1_small_big_mix", help="path of the valide dataset")
parser.add_argument("--dataset_path", type=str, default=r"F:\DeepLData\target_stage1_small_big_mix", help="path of the train dataset")

parser.add_argument("--channels_in", type=int, default=1, help="number of image channels")
parser.add_argument("--channels_out", type=int, default=1, help="number of image channels")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=50, help="epoch from which to start lr decay")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between model_fmi checkpoints")
parser.add_argument("--dataset_name", type=str, default='FMI_CGAN_repair', help="folder to save model")
# parser.add_argument("--netG", type=str, default=r'/root/autodl-tmp/FMI_GAN/pix2pix/saved_models/pic2pic/generator_104.pth', help="path model Gen")
# parser.add_argument("--netD", type=str, default=r'/root/autodl-tmp/FMI_GAN/pix2pix/saved_models/pic2pic/discriminator_104.pth', help="path model Discrimi")
parser.add_argument("--netG", type=str, default=r'', help="path model Gen")
parser.add_argument("--netD", type=str, default=r'', help="path model Discrimi")
opt = parser.parse_args()
print(opt)

# 替换交叉熵损失为余弦相似度损失
def cosine_similarity_loss(feat1, feat2):
    """计算特征间的余弦相似度损失"""
    feat1_norm = torch.nn.functional.normalize(feat1, dim=1)
    feat2_norm = torch.nn.functional.normalize(feat2, dim=1)
    return 1 - (feat1_norm * feat2_norm).sum(dim=1).mean()


if __name__ == '__main__':
    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    # Loss functions
    criterion_MSE = torch.nn.MSELoss()
    criterion_ssim = GeologicalSSIM(window_size=27, channel=2)

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.img_size_x // 2 ** 4, opt.img_size_y // 2 ** 4)

    # Initialize generator and discriminator
    generator = GeneratorUNet(in_channels=opt.channels_in, out_channels=opt.channels_out)
    discriminator = EnhancedDiscriminator(in_channels=opt.channels_out+opt.channels_in)

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_MSE.cuda()
        criterion_ssim.cuda()
        # criterion_cross_entropy.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
        discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
    if opt.netG != '':
        print('from model continue to train.........')
        generator.load_state_dict(torch.load(opt.netG), strict=True)
    else:
        print('train a new model .......... ')
        generator.apply(weights_init_normal)
    if opt.netD != '':
        discriminator.load_state_dict(torch.load(opt.netD), strict=True)


    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # 添加梯度裁剪
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
    # 添加学习率衰减
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.9)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.9)

    dataloader = ImageDataset_FMI(opt.dataset_path, x_l=opt.img_size_x, y_l=opt.img_size_x)
    dataloader = DataLoader(dataloader, shuffle=True, batch_size=opt.batch_size, drop_last=True, pin_memory=True, num_workers=opt.n_cpu)

    dataloader_val = ImageDataset_FMI(opt.dataset_path_val, x_l=opt.img_size_x, y_l=opt.img_size_x)
    dataloader_val = DataLoader(dataloader_val, shuffle=True, batch_size=opt.batch_size, drop_last=True, pin_memory=True, num_workers=opt.n_cpu)

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # 在训练过程中，保存模型的生成效果
    def sample_images(imgs, input, batches_done):
        gen_data = generator(input)

        # Save sample
        imgs = torch.cat((imgs.data, input.data, gen_data.data), 1)
        imgs = imgs.reshape((imgs.shape[0] * imgs.shape[1], 1, imgs.shape[-2], imgs.shape[-1]))
        # print(imgs.shape)

        save_image(imgs, "images/%s/%d.png" % (opt.dataset_name, batches_done), nrow=12, padding=1, normalize=True)
        model_path = "saved_models/{}/model_{}_{}.pth".format(opt.dataset_name, 'ele_gen', batches_done)
        torch.save(generator.state_dict(), model_path)
        model_path = "saved_models/{}/model_{}_{}.pth".format(opt.dataset_name, 'ele_dis', batches_done)
        torch.save(discriminator.state_dict(), model_path)


    log_train = []
    log_val = []

    # ----------
    #  Training
    # ----------

    prev_time = time.time()
    best_val_loss = 100


    # 方法1：展平特征
    def flatten_features(features):
        """将特征展平为 (batch, features) 格式"""
        return features.view(features.size(0), -1)


    for epoch in range(opt.epoch, opt.n_epochs):
        # 进行模型的训练工作
        for i, batch in enumerate(dataloader):
            # Model inputs
            real_input = Variable(batch["B"].type(Tensor))
            real_output = Variable(batch["A"].type(Tensor))

            # 创建张量并移动到正确的设备
            device = real_input.device
            valid_16 = torch.tensor(np.ones((real_input.size(0), 1, 16, 16)), dtype=torch.float32, device=device, requires_grad=False)
            fake_16 = torch.tensor(np.zeros((real_input.size(0), 1, 16, 16)), dtype=torch.float32, device=device, requires_grad=False)
            valid_8 = torch.tensor(np.ones((real_input.size(0), 1, 8, 8)), dtype=torch.float32, device=device, requires_grad=False)
            fake_8  = torch.tensor(np.zeros((real_input.size(0), 1, 8, 8)), dtype=torch.float32, device=device, requires_grad=False)
            valid_4 = torch.tensor(np.ones((real_input.size(0), 1, 4, 4)), dtype=torch.float32, device=device, requires_grad=False)
            fake_4  = torch.tensor(np.zeros((real_input.size(0), 1, 4, 4)), dtype=torch.float32, device=device, requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()

            fake_gen = generator(real_input)                                            ### torch.Size([16, 2, 256, 256]) torch.Size([16, 8, 256, 256]) torch.Size([16, 2, 256, 256]) torch.float32 torch.float32 torch.float32
            pred_fake = discriminator(fake_gen, real_input)                             ### [torch.Size([16, 1, 16, 16]), torch.Size([16, 1, 8, 8]), torch.Size([16, 1, 4, 4])]
            pred_real = discriminator(real_output, real_input)                          ### [torch.Size([16, 1, 16, 16]), torch.Size([16, 1, 8, 8]), torch.Size([16, 1, 4, 4])]

            # 生成器希望判别器认为生成的图像是真实的
            # 使用 valid_ 目标（全1）计算损失
            loss_fake_DIS_16 = criterion_MSE(pred_fake[0], valid_16)
            loss_fake_DIS_8 = criterion_MSE(pred_fake[1], valid_8)
            loss_fake_DIS_4 = criterion_MSE(pred_fake[2], valid_4)
            # print(loss_fake_DIS_16.item(), loss_fake_DIS_8.item(), loss_fake_DIS_4.item())
            loss_fake_DIS = (loss_fake_DIS_16 + 2*loss_fake_DIS_8 + 4*loss_fake_DIS_4) / 7

            # 重建损失
            loss_ssim_GEN = criterion_ssim(fake_gen, real_output)
            loss_mse_GEN = criterion_MSE(fake_gen, real_output)
            # print(loss_ssim_GEN, loss_mse_GEN, loss_l1_GEN)

            # 总损失
            loss_G = 0.1 * loss_fake_DIS + 0.8 * loss_ssim_GEN + 0.2 * loss_mse_GEN
            # loss_G = 0.1 * loss_fake_DIS + 0.99 * loss_mse_GEN

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real loss
            pred_fake = discriminator(fake_gen.detach(), real_input)                    ### torch.Size([16, 1, 16, 16]) torch.Size([16, 1, 8, 8]) torch.Size([16, 1, 4, 4])
            # Fake loss
            pred_real = discriminator(real_output, real_input)

            # 真实图像损失
            loss_real_DIS_16 = criterion_MSE(pred_real[0], valid_16)
            loss_real_DIS_8 = criterion_MSE(pred_real[1], valid_8)
            loss_real_DIS_4 = criterion_MSE(pred_real[2], valid_4)

            # 生成图像损失
            loss_fake_DIS_16 = criterion_MSE(pred_fake[0].detach(), fake_16)
            loss_fake_DIS_8 = criterion_MSE(pred_fake[1].detach(), fake_8)
            loss_fake_DIS_4 = criterion_MSE(pred_fake[2].detach(), fake_4)
            # print(loss_real_DIS_16, loss_real_DIS_8, loss_real_DIS_4, loss_fake_DIS_16, loss_fake_DIS_8, loss_fake_DIS_4)

            # 总判别器损失
            loss_DIS_MSE = (loss_real_DIS_16 + 2*loss_real_DIS_8 + 4*loss_real_DIS_4 + loss_fake_DIS_16 + 2*loss_fake_DIS_8 + 4*loss_fake_DIS_4) / 14

            loss_cross_entropy_fake_real_16 = cosine_similarity_loss(flatten_features(pred_fake[0]), flatten_features(pred_real[0]))
            loss_cross_entropy_fake_real_8 = cosine_similarity_loss(flatten_features(pred_fake[1]), flatten_features(pred_real[1]))
            loss_cross_entropy_fake_real_4 = cosine_similarity_loss(flatten_features(pred_fake[2]), flatten_features(pred_real[2]))
            loss_DIS_cross_entropy = (loss_cross_entropy_fake_real_16 + 2*loss_cross_entropy_fake_real_8 + 4*loss_cross_entropy_fake_real_4) / 7
            print(loss_cross_entropy_fake_real_16, loss_cross_entropy_fake_real_8, loss_cross_entropy_fake_real_4)

            loss_DIS_ALL = 0.6 * loss_DIS_MSE + 0.4 * loss_DIS_cross_entropy

            loss_DIS_ALL.backward()
            optimizer_D.step()

            # 本轮学习结束，学习率衰减
            scheduler_G.step()
            scheduler_D.step()

            # --------------
            #  Log Progress
            # --------------
            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # loss_G = 0.01 * loss_fake_DIS + 0.5 * loss_ssim_GEN + 0.3 * loss_mse_GEN
            # loss_fake_DIS = loss_fake_DIS_16 + loss_fake_DIS_8 + loss_fake_DIS_4
            # loss_DIS_MSE = (loss_real_DIS_16 + loss_real_DIS_8 + loss_real_DIS_4 + loss_fake_DIS_16 + loss_fake_DIS_8 + loss_fake_DIS_4) / 6
            # loss_DIS_cross_entropy = (loss_cross_entropy_fake_real_16 + loss_cross_entropy_fake_real_8 + loss_cross_entropy_fake_real_4) / 3
            log_train.append([epoch, opt.n_epochs, i, len(dataloader),
                              loss_fake_DIS_16.item(), loss_fake_DIS_8.item(), loss_fake_DIS_4.item(),
                              loss_fake_DIS.item(), loss_ssim_GEN.item(), loss_mse_GEN.item(),
                              loss_real_DIS_16.item(), loss_real_DIS_8.item(), loss_real_DIS_4.item(),
                              loss_cross_entropy_fake_real_16.item(), loss_cross_entropy_fake_real_8.item(), loss_cross_entropy_fake_real_4.item(),
                              loss_DIS_MSE.item(), loss_DIS_cross_entropy.item(),
                              optimizer_G.state_dict()['param_groups'][0]['lr'], optimizer_D.state_dict()['param_groups'][0]['lr']])

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [G  mse: %f, ssim: %f, all:%f] [D mse: %f, cross entropy:%f, all:%f] [Gen lr:%f, Dis lr:%f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_mse_GEN.item(),
                    loss_ssim_GEN.item(),
                    loss_G.item(),
                    loss_DIS_MSE.item(),
                    loss_DIS_cross_entropy.item(),
                    loss_DIS_ALL.item(),
                    optimizer_G.state_dict()['param_groups'][0]['lr'],
                    optimizer_D.state_dict()['param_groups'][0]['lr'],
                    time_left,
                )
            )

            # # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(real_output, real_input, batches_done)
            # break

        with torch.no_grad():
            # 进行模型的测试
            generator.eval()
            val_mse_losses = []
            val_ssim_losses = []
            with torch.no_grad():
                for i, batch in enumerate(dataloader_val):
                    real_input = Variable(batch["B"].type(Tensor))
                    real_output = Variable(batch["A"].type(Tensor))

                    fake_gen = generator(real_input)
                    loss_mse = criterion_MSE(fake_gen, real_output)
                    loss_ssim = criterion_ssim(fake_gen, real_output)
                    val_mse_losses.append(loss_mse.item())
                    val_ssim_losses.append(loss_ssim.item())

            avg_mse_loss = sum(val_mse_losses) / len(val_mse_losses)
            avg_ssim_loss = sum(val_ssim_losses) / len(val_ssim_losses)
            avg_val_loss = 0.4*avg_ssim_loss + 0.3*avg_mse_loss
            print(f"\nValidation Loss: {avg_val_loss:.4f}")

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(generator.state_dict(), f"saved_models/{opt.dataset_name}/best_generator.pth")
                torch.save(discriminator.state_dict(), f"saved_models/{opt.dataset_name}/best_discriminator.pth")

            generator.train()

    np.savetxt('Train_simulate_model_log_{}.txt'.format(epoch), np.array(log_train), delimiter='\t', comments='',newline='\n', fmt='%.4f')
    np.savetxt('Valide_simulate_model_log_{}.txt'.format(epoch), np.array(log_val), delimiter='\t', comments='',newline='\n', fmt='%.4f')
