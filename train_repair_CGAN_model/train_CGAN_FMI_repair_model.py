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

from src_ele.pic_opeeration import show_Pic
from train_pix2pix_simulate.ssim_geo import GeologicalSSIM
from train_repair_CGAN_model.Dataloader_FMI_add_empty_stripe import dataloader_padding_striped
from train_repair_CGAN_model.MODEL_Discriminator_CGAN_repair import EnhancedDiscriminator
from train_repair_CGAN_model.MODEL_Generator_UNET import GeneratorUNet

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
parser.add_argument("--img_length", type=int, default=128, help="size of image height")
parser.add_argument("--FMI_padding_length", type=int, default=16, help="size of image height")
parser.add_argument("--batch_size", type=int, default=52, help="size of the batches")
parser.add_argument("--num_workers", type=int, default=6, help="number of cpu threads to use during batch generation")
parser.add_argument("--dataset_path_val", type=str, default=r"F:\DeepLData\target_stage1_small_big_mix", help="path of the valide dataset")
parser.add_argument("--dataset_path", type=str, default=r"F:\DeepLData\target_stage1_small_big_mix", help="path of the train dataset")

parser.add_argument("--channels_in", type=int, default=1, help="number of image channels")
parser.add_argument("--channels_out", type=int, default=1, help="number of image channels")
parser.add_argument("--lr", type=float, default=0.005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.6, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between model_fmi checkpoints")
parser.add_argument("--dataset_name", type=str, default='FMI_CGAN_repair', help="folder to save model")
parser.add_argument("--netG", type=str, default=r'D:\GitHub\FMI_CGAN\train_repair_CGAN_model\saved_models\FMI_CGAN_repair\4\model_ele_gen_22800.pth', help="path model Gen")
parser.add_argument("--netD", type=str, default=r'D:\GitHub\FMI_CGAN\train_repair_CGAN_model\saved_models\FMI_CGAN_repair\4\model_ele_dis_22800.pth', help="path model Discrimi")
# parser.add_argument("--netG", type=str, default=r'', help="path model Gen")
# parser.add_argument("--netD", type=str, default=r'', help="path model Discrimi")
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
    criterion_MSE = torch.nn.MSELoss()          # MSE loss
    criterion_ssim = GeologicalSSIM(window_size=27, channel=opt.channels_out)       # SSIM loss

    # Initialize generator and discriminator
    generator = GeneratorUNet(in_channels=opt.channels_in*2, out_channels=opt.channels_out)
    discriminator = EnhancedDiscriminator(in_channels=opt.channels_out+opt.channels_in)

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_MSE.cuda()
        criterion_ssim.cuda()
        # criterion_cross_entropy.cuda()

    if opt.netG != '':
        print('from model continue to train.........')
        generator.load_state_dict(torch.load(opt.netG), strict=True)
    else:
        print('train a new netG model .......... ')
    if opt.netD != '':
        discriminator.load_state_dict(torch.load(opt.netD), strict=True)
    else:
        print('train a new netD model .......... ')

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # 添加梯度裁剪
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

    # 添加学习率衰减
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.9)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.9)

    dataloader = dataloader_padding_striped(opt.dataset_path, len_pic=opt.img_length, padding=opt.FMI_padding_length)
    dataloader = DataLoader(dataloader, shuffle=True, batch_size=opt.batch_size, drop_last=True, pin_memory=False, num_workers=opt.num_workers)

    dataloader_val = dataloader_padding_striped(opt.dataset_path_val, len_pic=opt.img_length, padding=opt.FMI_padding_length)
    dataloader_val = DataLoader(dataloader_val, shuffle=True, batch_size=opt.batch_size, drop_last=True, pin_memory=True, num_workers=opt.num_workers//2)

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # 在训练过程中，保存模型的生成效果
    def sample_images(imgs_org, input, mask, batches_done):
        data_input = torch.cat((input, mask), dim=1)
        gen_data = generator(data_input)

        # Save sample
        imgs = torch.cat((imgs_org.data, mask.data, input.data, gen_data.data), 1)
        imgs = imgs.reshape((imgs.shape[0] * imgs.shape[1], 1, imgs.shape[-2], imgs.shape[-1]))

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
            # mask, real_all, real_input, real_deduction
            real_all = Variable(batch["real_all"].type(Tensor))                         # 完整的图像
            real_input = Variable(batch["real_input"].type(Tensor))                     # 图像被遮掩后的部分，也是图像修复模型的输入
            real_target = Variable(batch["real_deduction"].type(Tensor))             # 图像被遮掩的部分，也是模型要进行修复的部分
            mask = Variable(batch["mask"].type(Tensor))                                 # 图像的空白条带掩码，其中的0是图像保存的部分，其中的1是空白条带部分
            # print(real_all.shape, real_input.shape, real_target.shape, mask.shape)

            # 创建张量并移动到正确的设备
            device = real_all.device
            valid_16 = torch.tensor(np.ones((real_all.size(0), 1, opt.img_length//16, opt.img_length//16)), dtype=torch.float32, device=device, requires_grad=False)
            fake_16 = torch.tensor(np.zeros((real_all.size(0), 1, opt.img_length//16, opt.img_length//16)), dtype=torch.float32, device=device, requires_grad=False)
            valid_8 = torch.tensor(np.ones((real_all.size(0), 1, opt.img_length//32, opt.img_length//32)), dtype=torch.float32, device=device, requires_grad=False)
            fake_8  = torch.tensor(np.zeros((real_all.size(0), 1, opt.img_length//32, opt.img_length//32)), dtype=torch.float32, device=device, requires_grad=False)
            valid_4 = torch.tensor(np.ones((real_all.size(0), 1, opt.img_length//64, opt.img_length//64)), dtype=torch.float32, device=device, requires_grad=False)
            fake_4  = torch.tensor(np.zeros((real_all.size(0), 1, opt.img_length//64, opt.img_length//64)), dtype=torch.float32, device=device, requires_grad=False)

            # ------------------
            #  Train Generators 训练生成器，主要目的是生成图像 进行图像的修复 以及 欺骗分辨器 使生成的图像能够以真乱假
            # ------------------
            optimizer_G.zero_grad()

            data_input = torch.cat((real_input, mask), dim=1)
            model_generate_all = generator(data_input)
            generate_target = mask * model_generate_all

            # # 1. 从计算图中分离张量，避免影响梯度计算
            # with torch.no_grad():
            #     # 选择第一个样本进行展示
            #     real_all_np = real_all[0, 0].detach().cpu().numpy()
            #     generate_deduction_np = generate_deduction[0, 0].detach().cpu().numpy()
            #     real_input_np = real_input[0, 0].detach().cpu().numpy()
            #     mask_np = mask[0, 0].detach().cpu().numpy()
            #     model_generate_all_np = model_generate_all[0, 0].detach().cpu().numpy()
            #     show_Pic([real_all_np, real_input_np, generate_deduction_np, mask_np, model_generate_all_np])

            valid_generator_input_part = discriminator(model_generate_all.detach(), real_input)
            valid_generator_target_part = discriminator(model_generate_all.detach(), real_target)

            # 生成器希望判别器认为生成的图像是真实的
            # 使用 valid_ 目标（全1）计算损失
            loss_input_DIS_16 = criterion_MSE(valid_generator_input_part[0], valid_16)
            loss_input_DIS_8 = criterion_MSE(valid_generator_input_part[1], valid_8)
            loss_input_DIS_4 = criterion_MSE(valid_generator_input_part[2], valid_4)
            loss_input_DIS = (loss_input_DIS_16 + 2*loss_input_DIS_8 + 4*loss_input_DIS_4) / 7
            loss_target_DIS_16 = criterion_MSE(valid_generator_target_part[0], valid_16)
            loss_target_DIS_8 = criterion_MSE(valid_generator_target_part[1], valid_8)
            loss_target_DIS_4 = criterion_MSE(valid_generator_target_part[2], valid_4)
            loss_target_DIS = (loss_target_DIS_16 + 2*loss_target_DIS_8 + 4*loss_target_DIS_4) / 7

            # 重建MSE损失--->总的
            loss_ssim_GEN_all = criterion_ssim(model_generate_all, real_all)
            loss_mse_GEN_all = criterion_MSE(model_generate_all, real_all)

            # 重建SSIM损失--->注重强调待修复部分
            loss_ssim_GEN_deduction = criterion_ssim(generate_target, real_target)
            loss_mse_GEN_deduction = criterion_MSE(generate_target, real_target)

            # 重建 交叉熵cross_entropy 损失--->总的
            loss_cross_entropy_fake_real_all = cosine_similarity_loss(flatten_features(model_generate_all), flatten_features(real_all))
            loss_cross_entropy_fake_target = cosine_similarity_loss(flatten_features(real_target), flatten_features(generate_target))
            loss_cross_entropy = (7*loss_cross_entropy_fake_real_all + 3*loss_cross_entropy_fake_target)/10

            # 总损失
            # loss_G = 0.05 * (loss_input_DIS+loss_target_DIS)/2 + 0.3 * (loss_ssim_GEN_all+loss_mse_GEN_all)/2 + 0.3*(loss_ssim_GEN_deduction+loss_mse_GEN_deduction)/2 + 0.3*loss_cross_entropy
            # loss_G = 0.02*(loss_input_DIS+loss_target_DIS)/2 + 0.4*(loss_mse_GEN_all+loss_mse_GEN_deduction)/2 + 0.2*loss_ssim_GEN_all + 0.3*loss_cross_entropy
            # loss_G = (
            #         0.02 * (loss_input_DIS + loss_target_DIS) / 2 +  # 判别器损失权重降低
            #         0.20 * loss_mse_GEN_all +  # 整体MSE权重增加
            #         0.20 * loss_mse_GEN_deduction +  # 修复区域MSE权重增加
            #         0.20 * loss_ssim_GEN_all +  # 整体SSIM权重
            #         0.30 * loss_cross_entropy   # 整体交叉熵特征相似度
            # )
            # loss_G = 0.02*(loss_input_DIS+loss_target_DIS)/2 + 0.2*loss_mse_GEN_all + 0.3+loss_mse_GEN_deduction + 0.2*loss_ssim_GEN_all + 0.2*loss_cross_entropy_fake_real_all + 0.1*loss_cross_entropy_fake_target
            loss_G = (
                    0.02 * (loss_input_DIS + loss_target_DIS) / 2 +  # 判别器损失权重降低
                    0.20 * loss_mse_GEN_all +  # 整体MSE权重增加
                    0.30 * loss_mse_GEN_deduction +  # 修复区域MSE权重增加
                    0.25 * loss_ssim_GEN_all +  # 整体SSIM权重
                    0.15 * loss_cross_entropy_fake_real_all +  # 整体交叉熵特征相似度
                    0.08 * loss_cross_entropy_fake_target  # 修复区域交叉熵特征相似度
            )

            loss_G.backward()
            optimizer_G.step()


            # ---------------------
            #  Train Discriminator 分辨器，主要目的是分辨图像
            # ---------------------
            optimizer_D.zero_grad()

            # Fake loss
            fake1 = discriminator(model_generate_all.detach(), real_input)
            fake2 = discriminator(model_generate_all.detach(), real_target)
            fake3 = discriminator(model_generate_all.detach(), model_generate_all.detach())
            # 生成图像损失
            fake1_16 = criterion_MSE(fake1[0].detach(), fake_16)
            fake1_8 = criterion_MSE(fake1[1].detach(), fake_8)
            fake1_4 = criterion_MSE(fake1[2].detach(), fake_4)
            fake2_16 = criterion_MSE(fake2[0].detach(), fake_16)
            fake2_8 = criterion_MSE(fake2[1].detach(), fake_8)
            fake2_4 = criterion_MSE(fake2[2].detach(), fake_4)
            fake3_16 = criterion_MSE(fake3[0].detach(), fake_16)
            fake3_8 = criterion_MSE(fake3[1].detach(), fake_8)
            fake3_4 = criterion_MSE(fake3[2].detach(), fake_4)
            fake_loss = (fake1_16 + fake2_16 + fake3_16 + 2*fake1_8 + 2*fake2_8 + 2*fake3_8 + 4*fake1_4 + 4*fake2_4 + 4*fake3_4)/21

            # Real loss
            real1 = discriminator(real_all, real_input)
            real2 = discriminator(real_all, real_target)
            real3 = discriminator(real_all, real_all)
            # 真实图像损失
            real1_16 = criterion_MSE(real1[0], valid_16)
            real1_8 = criterion_MSE(real1[1], valid_8)
            real1_4 = criterion_MSE(real1[2], valid_4)
            real2_16 = criterion_MSE(real2[0], valid_16)
            real2_8 = criterion_MSE(real2[1], valid_8)
            real2_4 = criterion_MSE(real2[2], valid_4)
            real3_16 = criterion_MSE(real3[0], valid_16)
            real3_8 = criterion_MSE(real3[1], valid_8)
            real3_4 = criterion_MSE(real3[2], valid_4)
            real_loss = (real1_16 + real2_16 +real3_16 + 2*real1_8 + 2*real2_8 + 2*real3_8 + 4*real1_4 + 4*real2_4 + 4*real3_4)/21

            # 总判别器损失
            loss_DIS_ALL = (fake_loss + real_loss)/2
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

            # loss_G = 0.05 * (loss_input_DIS+loss_target_DIS)/2 + 0.3 * (loss_ssim_GEN_all+loss_mse_GEN_all)/2 + 0.3*loss_cross_entropy + 0.3*(loss_ssim_GEN_deduction+loss_mse_GEN_deduction)/2
            # fake_loss = (fake1_16 + fake2_16 + fake3_16 + 2*fake1_8 + 2*fake2_8 + 2*fake3_8 + 4*fake1_4 + 4*fake2_4 + 4*fake3_4)/21
            # real_loss = (real1_16 + real2_16 +real3_16 + 2*real1_8 + 2*real2_8 + 2*real3_8 + 4*real1_4 + 4*real2_4 + 4*real3_4)/21
            # loss_DIS_ALL = fake_loss + real_loss
            log_train.append([epoch, opt.n_epochs, i, len(dataloader),
                              loss_ssim_GEN_all.item(), loss_mse_GEN_all.item(),
                              loss_ssim_GEN_deduction.item(), loss_mse_GEN_deduction.item(),
                              loss_cross_entropy.item(), loss_G.item(),
                              fake1_16.item(), fake2_16.item(), fake3_16.item(),
                              fake1_8.item(), fake2_8.item(), fake3_8.item(),
                              fake1_4.item(), fake2_4.item(), fake3_4.item(),
                              real1_16.item(), real2_16.item(), real3_16.item(),
                              real1_8.item(), real2_8.item(), real3_8.item(),
                              real1_4.item(), real2_4.item(), real3_4.item(),
                              fake_loss.item(), real_loss.item(),
                              optimizer_G.state_dict()['param_groups'][0]['lr'], optimizer_D.state_dict()['param_groups'][0]['lr']])

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [G  SSIM: %f, MSE: %f, all:%f] [D fake valid loss: %f, real valid loss:%f, loss all:%f] [Gen lr:%f, Dis lr:%f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_ssim_GEN_deduction.item(),
                    loss_mse_GEN_deduction.item(),
                    loss_G.item(),
                    fake_loss.item(),
                    real_loss.item(),
                    loss_DIS_ALL.item(),
                    optimizer_G.state_dict()['param_groups'][0]['lr'],
                    optimizer_D.state_dict()['param_groups'][0]['lr'],
                    time_left,
                )
            )

            # # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(real_all, real_input, mask, batches_done)

        # with (torch.no_grad()):
        #     # 进行模型的测试
        #     generator.eval()
        #     val_mse_losses = []
        #     val_ssim_losses = []
        #     val_cross_entropy_losses = []
        #     with torch.no_grad():
        #         for i, batch in enumerate(dataloader_val):
        #             real_all = Variable(batch["real_all"].type(Tensor))                         # 完整的图像
        #             real_input = Variable(batch["real_input"].type(Tensor))                     # 图像被遮掩后的部分，也是图像修复模型的输入
        #             real_target = Variable(batch["real_deduction"].type(Tensor))             # 图像被遮掩的部分，也是模型要进行修复的部分
        #             mask = Variable(batch["mask"].type(Tensor))                                 # 图像的空白条带掩码，其中的0是图像保存的部分，其中的1是空白条带部分
        #
        #             data_input = torch.cat((real_input, mask), dim=1)
        #             model_generate_all = generator(data_input)
        #             generate_target = mask * model_generate_all
        #
        #             # 重建MSE损失--->总的
        #             loss_ssim_GEN_all = criterion_ssim(model_generate_all, real_all)
        #             loss_mse_GEN_all = criterion_MSE(model_generate_all, real_all)
        #
        #             # 重建SSIM损失--->待修复部分
        #             loss_ssim_GEN_deduction = criterion_ssim(generate_target, real_target)
        #             loss_mse_GEN_deduction = criterion_MSE(generate_target, real_target)
        #
        #             # 重建 交叉熵cross_entropy 损失--->总的
        #             loss_cross_entropy_fake_real_all = cosine_similarity_loss(flatten_features(model_generate_all), flatten_features(real_all))
        #             loss_cross_entropy_fake_target = cosine_similarity_loss(flatten_features(real_target), flatten_features(generate_target))
        #             loss_cross_entropy = (7 * loss_cross_entropy_fake_real_all + 3 * loss_cross_entropy_fake_target) / 10
        #             loss_G = 0.2*loss_mse_GEN_all + 0.3+loss_mse_GEN_deduction +0.2*loss_ssim_GEN_all + 0.2*loss_cross_entropy_fake_real_all + 0.1*loss_cross_entropy_fake_target
        #
        #             val_mse_losses.append((loss_mse_GEN_all.item()+loss_mse_GEN_deduction.item())/2)
        #             val_ssim_losses.append((loss_ssim_GEN_all.item()+loss_ssim_GEN_deduction.item())/2)
        #             val_cross_entropy_losses.append(loss_cross_entropy.item())
        #             log_val.append([epoch, opt.n_epochs, i, len(dataloader),
        #                               loss_ssim_GEN_all.item(), loss_mse_GEN_all.item(),
        #                               loss_ssim_GEN_deduction.item(), loss_mse_GEN_deduction.item(),
        #                               loss_cross_entropy.item(), loss_G.item(),
        #                               optimizer_G.state_dict()['param_groups'][0]['lr'],
        #                               optimizer_D.state_dict()['param_groups'][0]['lr']])
        #
        #     avg_mse_loss = sum(val_mse_losses) / len(val_mse_losses)
        #     avg_ssim_loss = sum(val_ssim_losses) / len(val_ssim_losses)
        #     avg_cs_loss = sum(val_cross_entropy_losses) / len(val_cross_entropy_losses)
        #     avg_val_loss = 0.2*avg_ssim_loss + 0.3*avg_mse_loss + 0.2*avg_cs_loss
        #     print(f"\nValidation Loss: {avg_val_loss:.4f}")
        #
        #     # 保存最佳模型
        #     if loss_G < best_val_loss:
        #         best_val_loss = loss_G
        #         torch.save(generator.state_dict(), f"saved_models/{opt.dataset_name}/best_generator.pth")
        #         torch.save(discriminator.state_dict(), f"saved_models/{opt.dataset_name}/best_discriminator.pth")
        #
        #     generator.train()

        np.savetxt('Train_simulate_model_log_{}.txt'.format(epoch), np.array(log_train), delimiter='\t', comments='',newline='\n', fmt='%.4f')
        np.savetxt('Valide_simulate_model_log_{}.txt'.format(epoch), np.array(log_val), delimiter='\t', comments='',newline='\n', fmt='%.4f')
