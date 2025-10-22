import argparse
from scipy.interpolate import CubicSpline
import cv2
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from src_ele.ele_data_process import dynamic_enhancement
from train_pix2pix_simulate.pix2pix import GeneratorUNet
from src_ele.pic_opeeration import show_Pic
from use_gan_model_create_fmi.dataloader_use_gan_model import ImageDataset_FMI_SPLIT_NO_REPEAT


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

    # 图像的左右两侧靠拢缩放，rxo_stretched大于0.5，则扩大，小于0.5则缩小
    target_img = np.zeros_like(img_ele)
    for i in range(img_ele.shape[0]):
        scale_ratio = rxo_stretched[i]
        if scale_ratio < 0.5:
            scale_ratio *= 2
            target_img[i] = img_ele[i] * (scale_ratio + 0.01)
        else:
            scale_ratio = (1-scale_ratio)*2
            target_img[i] = 1-((1-img_ele[i]) * (scale_ratio + 0.01))

    target_img = np.clip(target_img, 0, 1)
    return target_img


def use_pix2pix_gen_FMI(opt):
    cuda = True if torch.cuda.is_available() else False

    # Initialize generator and discriminator
    generator = GeneratorUNet(in_channels=opt.channels_in, out_channels=opt.channels_out)

    if len(opt.netG) > 1:
        if cuda:
            # 使用 weights_only=True 以避免安全警告
            generator.load_state_dict(
                torch.load(opt.netG, weights_only=True, map_location="cuda"),
                strict=True
            )
        else:
            generator.load_state_dict(
                torch.load(opt.netG, weights_only=True, map_location=torch.device('cpu')),
                strict=True
            )
    else:
        print('no available netG path:{}'.format(opt.netG))

    if cuda:
        generator = generator.cuda()

    dataloader_base = ImageDataset_FMI_SPLIT_NO_REPEAT(path=opt.dataset_path, x_l=opt.img_size_x, y_l=opt.img_size_x, win_len=opt.win_len)
    print('split layer to :{} windows to simulate'.format(dataloader_base.length))
    dataloader = DataLoader(dataloader_base, shuffle=False, batch_size=opt.batch_size, drop_last=False, pin_memory=False, num_workers=opt.n_cpu)

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

    dataloader_base.combine_pic_list_to_full_fmi(img_gan)

    path_o = opt.dataset_path.replace('simu_cracks', 'simu_FMI')
    print(path_o)
    # cv2.imwrite(path_o.replace('background_mask', 'fmi_dyna'), (img_dyna_gan_full).astype(np.uint8))
    # cv2.imwrite(path_o.replace('background_mask', 'fmi_stat'), (img_stat_gan_full).astype(np.uint8))
    # np.savetxt(path_o.replace('.png', '_dyna.txt'), (img_dyna_gan_full).astype(np.uint8), comments='', delimiter='\t', fmt='%d',
    #            header='simu_1\n100\n104\nIMAGE.DYNA_SIMU\n\n\n\n')
    # np.savetxt(path_o.replace('.png', '_stat.txt'), (img_stat_gan_full).astype(np.uint8), comments='', delimiter='\t', fmt='%d',
    #            header='simu_1\n100\n104\nIMAGE.STAT_SIMU\n\n\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_size_x", type=int, default=256, help="size of image height")
    parser.add_argument("--img_size_y", type=int, default=256, help="size of image width")
    parser.add_argument("--channels_in", type=int, default=4, help="number of image channels")
    parser.add_argument("--channels_out", type=int, default=2, help="number of image channels")
    parser.add_argument("--dataset_path", type=str, default=r"F:\DeepLData\FMI_SIMULATION\simu_cracks_2\9_background_mask.png", help="path of the dataset")
    parser.add_argument("--netG", type=str, default=r'D:\GitHub\FMI_CGAN\train_pix2pix_simulate\saved_models\pic2pic\3\best_generator.pth', help="netG path to load")
    parser.add_argument("--netD", type=str, default=r'D:\GitHub\FMI_CGAN\train_pix2pix_simulate\saved_models\pic2pic\3\best_discriminator.pth', help="netD path to load")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=1, help="num of cpu to process input data")
    parser.add_argument("--win_len", type=int, default=250, help="windows length to walk through the full layer mask")
    opt = parser.parse_args()
    print(opt)

    use_pix2pix_gen_FMI(opt)

