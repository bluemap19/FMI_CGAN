import argparse
from math import sqrt
import cv2
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from model_fmi.netG_FMI import FMI_Generator
from src_ele.pic_opeeration import show_Pic
from use_gan_model_create_fmi.dataloader_use_gan_model import ImageDataset_FMI_SIMULATE_LAYER

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_size_x", type=int, default=128, help="size of image height")
    parser.add_argument("--img_size_y", type=int, default=128, help="size of image width")
    parser.add_argument("--channels_in", type=int, default=2, help="number of image channels")
    parser.add_argument("--channels_out", type=int, default=2, help="number of image channels")

    # parser.add_argument("--dataset_path", type=str, default=r"D:\GitHubProj\dino\fracture_simulation\background_simulate\fracture_moni_3.png",
    # parser.add_argument("--dataset_path", type=str, default=r"D:\GitHubProj\dino\fracture_simulation\background_simulate\fracture_moni_7.png",
    parser.add_argument("--dataset_path", type=str, default=r"C:\Users\Administrator\PycharmProjects\FMI_GAN_Create\fracture_mask_simulate\simu_4\dyna_simu.png",
    # parser.add_argument("--dataset_path", type=str, default=r"D:\GitHubProj\dino\fracture_simulation\background_simulate\ZG8-2X_0_6179.0_6184.0_mask_33.png",
                        help="path of the dataset")
    parser.add_argument("--netG", type=str, default=r'D:\Data\pix2pix_exp\context_encoder_model_1\model_ele_gen_700.pth', help="netG path to load")
    parser.add_argument("--netD", type=str, default='', help="netD path to load")

    parser.add_argument("--batch_size", type=int, default=3, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=1, help="num of cpu to process input data")
    parser.add_argument("--mask_path", type=str, default=r"D:\Data\ele_img_big_small_mix_GAN_test", help="path of the dataset")

    parser.add_argument("--win_len", type=int, default=400, help="windows length to walk through the full layer mask")
    parser.add_argument("--step", type=int, default=10, help="windows step to walk through the full layer mask")
    opt = parser.parse_args()
    print(opt)

    cuda = True if torch.cuda.is_available() else False

    # Initialize generator and discriminator
    generator = FMI_Generator(channels_in=opt.channels_in, channels_out=opt.channels_out)

    if len(opt.netG) > 1:
        if cuda:
            generator.load_state_dict(torch.load(opt.netG), strict=True)
        else:
            generator.load_state_dict(torch.load(opt.netG, map_location=torch.device('cpu')), strict=True)
    else:
        print('no available netG path:{}'.format(opt.netG))

    if cuda:
        generator = generator.cuda()


    dataloader = ImageDataset_FMI_SIMULATE_LAYER(opt.dataset_path, x_l=opt.img_size_x, y_l=opt.img_size_x, win_len=opt.win_len, step=opt.step)
    print('split layer to :{} windows to simulate'.format(dataloader.length))
    dataloader = DataLoader(dataloader, shuffle=False, batch_size=opt.batch_size, drop_last=False, pin_memory=True,
                            num_workers=opt.n_cpu)


    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    prev_time = time.time()

    img_gan = np.array([])
    with torch.no_grad():
        for i, mask in enumerate(dataloader):
            # Model inputs
            mask = Variable(mask.type(Tensor))

            gen_parts = generator(mask).cpu().detach().numpy()
            mask = mask.cpu().detach().numpy()

            if len(img_gan) == 0:
                img_gan = gen_parts
            else:
                img_gan = np.append(img_gan, gen_parts, axis=0)

            # for j in range(gen_parts.shape[0]):
            #     show_Pic([mask[j, 0, :, :], 1-gen_parts[j, 0, :, :], 1-gen_parts[j, 1, :, :]], pic_order='13')
            # exit(0)

    # combine fracture to a layer FMI image
    img_dyna_gan_full = np.zeros((2200, 250))
    img_stat_gan_full = np.zeros((2200, 250))
    img_mask_gan_full = np.zeros((2200, 250))
    for i in range(img_gan.shape[0]):
        dyna_t = img_gan[i, 0, :, :]
        stat_t = img_gan[i, 1, :, :]

        dyna_t = cv2.resize(dyna_t, (250, opt.win_len))
        stat_t = cv2.resize(stat_t, (250, opt.win_len))
        mask = np.ones_like(dyna_t)
        # print(dyna_t.shape)

        img_dyna_gan_full[i*opt.step:i*opt.step+opt.win_len, :] += dyna_t
        img_stat_gan_full[i * opt.step:i * opt.step + opt.win_len, :] += stat_t
        img_mask_gan_full[i * opt.step:i * opt.step + opt.win_len, :] += mask

    img_dyna_gan_full[:(img_gan.shape[0] - 1) * opt.step + opt.win_len] /= img_mask_gan_full[:(img_gan.shape[0] - 1) * opt.step + opt.win_len]
    img_stat_gan_full[:(img_gan.shape[0] - 1) * opt.step + opt.win_len] /= img_mask_gan_full[:(img_gan.shape[0] - 1) * opt.step + opt.win_len]

    show_Pic([1-img_dyna_gan_full, 1-img_stat_gan_full, img_mask_gan_full], pic_order='13')
    cv2.imwrite('dyna_simu_1.png', (img_dyna_gan_full*255).astype(np.uint8))
    cv2.imwrite('stat_simu_1.png', (img_stat_gan_full*255).astype(np.uint8))
    np.savetxt('dyna_simu_1.txt', (img_dyna_gan_full*255).astype(np.uint8), comments='', delimiter='\t', fmt='%d',
               header='ZG8-2X\n6179\n6184\nIMAGE.DYNA_SIMU')
    np.savetxt('stat_simu_1.txt', (img_stat_gan_full*255).astype(np.uint8), comments='', delimiter='\t', fmt='%d',
               header='ZG8-2X\n6179\n6184\nIMAGE.STAT_SIMU')

