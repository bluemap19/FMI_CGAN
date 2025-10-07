import argparse

from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import cv2
import numpy as np
import time

from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from src_ele.ele_data_process import dynamic_enhancement
from train_pix2pix_simulate.pix2pix import GeneratorUNet
from src_ele.pic_opeeration import show_Pic
from use_gan_model_create_fmi.dataloader_use_gan_model import ImageDataset_FMI_SIMULATE_LAYER



def correct_image_with_rxo(img_ele, rxo, normalize_rxo=False):
    """
    使用rxo曲线校正图像

    参数:
    img_ele: 输入图像 (形状 [2500, 256])
    rxo: 校正曲线 (形状 [200,])
    normalize_rxo: 是否将rxo归一化到0-1范围 (默认True)

    返回:
    corrected_img: 校正后的图像

    处理步骤:
    1. 验证rxo范围并归一化
    2. 将rxo插值到图像长度(2500)
    3. 对图像每行应用校正
    4. 返回校正后图像
    """
    # 0. 验证输入
    if not isinstance(img_ele, np.ndarray) or img_ele.ndim != 2:
        raise ValueError("img_ele必须是二维numpy数组")

    if not isinstance(rxo, np.ndarray) or rxo.ndim != 1:
        raise ValueError("rxo必须是一维numpy数组")

    # 1. 验证并归一化rxo
    if normalize_rxo:
        # 归一化到0-1范围
        rxo_min = np.min(rxo)
        rxo_max = np.max(rxo)

        if rxo_max - rxo_min > 1e-6:  # 避免除以零
            rxo = (rxo - rxo_min) / (rxo_max - rxo_min)
        else:
            rxo = np.ones_like(rxo)  # 如果所有值相同，设为1

    # 2. 将rxo插值到图像长度
    original_rxo_length = len(rxo)
    target_length = img_ele.shape[0]

    # 创建原始rxo的x坐标 (0到1均匀分布)
    x_original = np.linspace(0, 1, original_rxo_length)

    # 创建目标x坐标
    x_target = np.linspace(0, 1, target_length)

    # 使用三次样条插值
    interpolator = CubicSpline(x_original, rxo)
    rxo_stretched = interpolator(x_target)
    # 确保插值结果在0-1范围内
    rxo_stretched = np.clip(rxo_stretched, 0, 1)

    # # 3. 对图像每行应用校正
    # # 将rxo_stretched转换为列向量 (2500, 1)
    # rxo_column = rxo_stretched[:, np.newaxis]
    # # # 应用校正: 每行乘以对应的rxo值
    # target_img = img_ele * rxo_column + img_ele

    # 图像的左右两侧靠拢缩放，rxo_stretched大于0.5，则扩大，小于0.5则缩小
    target_img = np.zeros_like(img_ele)
    for i in range(img_ele.shape[0]):
        scale_ratio = rxo_stretched[i]
        if scale_ratio < 0.5:
            scale_ratio *= 2
            # target_img[i] = img_ele[i] * (np.power(scale_ratio, 1.5) + 0.01)
            target_img[i] = img_ele[i] * (scale_ratio + 0.01)
        else:
            scale_ratio = (1-scale_ratio)*2
            # target_img[i] = 1-((1-img_ele[i]) * (np.power(scale_ratio, 1.5) + 0.01))
            target_img[i] = 1-((1-img_ele[i]) * (scale_ratio + 0.01))

    target_img = np.clip(target_img, 0, 1)
    return target_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_size_x", type=int, default=256, help="size of image height")
    parser.add_argument("--img_size_y", type=int, default=256, help="size of image width")
    parser.add_argument("--channels_in", type=int, default=8, help="number of image channels")
    parser.add_argument("--channels_out", type=int, default=2, help="number of image channels")

    # parser.add_argument("--dataset_path", type=str, default=r"D:\PycharmProject\FMI_GAN_Create\fracture_mask_simulate\simu-result\mask_dyna_1_100_500.png",
    # parser.add_argument("--dataset_path", type=str, default=r"F:\FMI_SIMULATION\simu_cracks\9_background_mask.png",
    parser.add_argument("--dataset_path", type=str, default=r"F:\DeepLData\FMI_SIMULATION\simu_cracks_2\9_background_mask.png",
                        help="path of the dataset")
    parser.add_argument("--style_path", type=str, default=r'D:\DeepLData\target_stage1_small_big_mix\guan17-11_194_3677.0025_3678.2525_dyna.png',
                        help="path of the dataset")
    parser.add_argument("--netG", type=str, default=r'D:\GitHub\FMI_GAN_Create\train_pix2pix_simulate\SSIM80%\best_generator.pth', help="netG path to load")
    parser.add_argument("--netD", type=str, default='', help="netD path to load")

    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=4, help="num of cpu to process input data")

    parser.add_argument("--win_len", type=int, default=250, help="windows length to walk through the full layer mask")
    parser.add_argument("--step", type=int, default=10, help="windows step to walk through the full layer mask")
    opt = parser.parse_args()
    print(opt)

    cuda = True if torch.cuda.is_available() else False

    # Initialize generator and discriminator
    generator = GeneratorUNet(in_channels=opt.channels_in, out_channels=opt.channels_out)

    if len(opt.netG) > 1:
        if cuda:
            # generator.load_state_dict(torch.load(opt.netG), strict=True)
            # 使用 weights_only=True 以避免安全警告
            generator.load_state_dict(
                torch.load(opt.netG, weights_only=True, map_location="cuda"),
                strict=True
            )
        else:
            # generator.load_state_dict(torch.load(opt.netG, map_location=torch.device('cpu')), strict=True)
            generator.load_state_dict(
                torch.load(opt.netG, weights_only=True, map_location=torch.device('cpu')),
                strict=True
            )
    else:
        print('no available netG path:{}'.format(opt.netG))

    if cuda:
        generator = generator.cuda()


    dataloader_base = ImageDataset_FMI_SIMULATE_LAYER(path=opt.dataset_path, style_dyna_pic_path=opt.style_path,
                                                 x_l=opt.img_size_x, y_l=opt.img_size_x, win_len=opt.win_len, step=opt.step)
    print('split layer to :{} windows to simulate'.format(dataloader_base.length))
    dataloader = DataLoader(dataloader_base, shuffle=False, batch_size=opt.batch_size, drop_last=False, pin_memory=True,
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
            #     show_Pic([mask[j, 0, :, :], 1-gen_parts[j, 1, :, :], 1-gen_parts[j, 0, :, :]], pic_order='13')

    # combine fracture to a layer FMI image
    img_dyna_gan_full = np.zeros(dataloader_base.get_base_shape())
    img_stat_gan_full = np.zeros(dataloader_base.get_base_shape())
    img_mask_gan_full = np.zeros(dataloader_base.get_base_shape())
    for i in range(img_gan.shape[0]):
        dyna_t = img_gan[i, 0, :, :]
        stat_t = img_gan[i, 1, :, :]

        dyna_t = cv2.resize(dyna_t, (dataloader_base.shape_base[1], opt.win_len))
        stat_t = cv2.resize(stat_t, (dataloader_base.shape_base[1], opt.win_len))
        mask = np.ones_like(dyna_t)
        # print(dyna_t.shape)

        img_dyna_gan_full[i * opt.step:i * opt.step + opt.win_len, :] += dyna_t
        img_stat_gan_full[i * opt.step:i * opt.step + opt.win_len, :] += stat_t
        img_mask_gan_full[i * opt.step:i * opt.step + opt.win_len, :] += mask

    img_dyna_gan_full /= (img_mask_gan_full + 0.01)
    img_stat_gan_full /= (img_mask_gan_full + 0.01)
    img_dyna_gan_full = np.nan_to_num(img_dyna_gan_full, nan=0.0, posinf=255, neginf=0)
    img_stat_gan_full = np.nan_to_num(img_stat_gan_full, nan=0.0, posinf=255, neginf=0)
    # show_Pic([img_dyna_gan_full, img_stat_gan_full, img_mask_gan_full], pic_order='13', figuresize=(6, 9))

    img_stat_gan_full = correct_image_with_rxo(img_stat_gan_full, dataloader_base.random_rxo, normalize_rxo=False)

    img_dyna_gan_full = img_dyna_gan_full * 255
    img_stat_gan_full = img_stat_gan_full * 255
    img_dyna_gan_full = dynamic_enhancement(img_stat_gan_full.astype(np.uint8), windows=200, step=1)

    # show_Pic([
    #           img_dyna_gan_full[100:300, :], img_stat_gan_full[100:300, :],
    #           img_dyna_gan_full[400:500, :], img_stat_gan_full[400:500, :],
    #           img_dyna_gan_full[-300:-100, :], img_stat_gan_full[-300:-100, :],
    #           img_dyna_gan_full[-900:-600, :], img_stat_gan_full[-900:-700, :], ], pic_order='42', figure=(4, 8))

    path_o = opt.dataset_path.replace('simu_cracks', 'simu_FMI')
    cv2.imwrite(path_o.replace('background_mask.png', 'fmi_dyna.png'), (img_dyna_gan_full).astype(np.uint8))
    cv2.imwrite(path_o.replace('background_mask.png', 'fmi_stat.png'), (img_stat_gan_full).astype(np.uint8))
    np.savetxt(path_o.replace('background_mask.png', 'fmi_dyna.txt'), (img_dyna_gan_full).astype(np.uint8), comments='', delimiter='\t', fmt='%d',
               header='simu_1\n100\n104\nIMAGE.DYNA_SIMU\n\n\n\n')
    np.savetxt(path_o.replace('background_mask.png', 'fmi_stat.txt'), (img_stat_gan_full).astype(np.uint8), comments='', delimiter='\t', fmt='%d',
               header='simu_1\n100\n104\nIMAGE.STAT_SIMU\n\n\n\n')

