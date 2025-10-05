import argparse
import cv2
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from train_pix2pix_simulate.pix2pix import GeneratorUNet
from src_ele.dir_operation import traverseFolder
from use_gan_model_create_fmi.dataloader_use_gan_model import ImageDataset_FMI_SIMULATE_LAYER


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_size_x", type=int, default=256, help="size of image height")
    parser.add_argument("--img_size_y", type=int, default=256, help="size of image width")
    parser.add_argument("--channels_in", type=int, default=2, help="number of image channels")
    parser.add_argument("--channels_out", type=int, default=2, help="number of image channels")

    # parser.add_argument("--dataset_path", type=str, default=r"D:\Data\image_simulated\background\p-t",
    parser.add_argument("--dataset_path", type=str, default=r"D:\Data\image_simulated\p-t-2",
                        help="path of the dataset")
    parser.add_argument("--netG", type=str, default=r'D:\Data\pix2pix_exp\model_pix2pix_unet\generator_78.pth', help="netG path to load")
    parser.add_argument("--netD", type=str, default='', help="netD path to load")

    parser.add_argument("--batch_size", type=int, default=6, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=2, help="num of cpu to process input data")

    parser.add_argument("--win_len", type=int, default=400, help="windows length to walk through the full layer mask")
    parser.add_argument("--step", type=int, default=20, help="windows step to walk through the full layer mask")
    opt = parser.parse_args()
    print(opt)

    cuda = True if torch.cuda.is_available() else False

    # Initialize generator and discriminator
    generator = GeneratorUNet(in_channels=opt.channels_in, out_channels=opt.channels_out)

    if len(opt.netG) > 1:
        if cuda:
            generator.load_state_dict(torch.load(opt.netG), strict=True)
        else:
            generator.load_state_dict(torch.load(opt.netG, map_location=torch.device('cpu')), strict=True)
    else:
        print('no available netG path:{}'.format(opt.netG))

    if cuda:
        generator = generator.cuda()

    path_list = traverseFolder(opt.dataset_path)
    # print(path_list)
    for i in range(len(path_list)):
        path_t = ''
        if path_list[i].__contains__('dyna'):
            path_t = path_list[i]
        else:
            continue
        index = path_t.split('_')[-3]
        print(path_t, index)

        dataloader = ImageDataset_FMI_SIMULATE_LAYER(path_t, x_l=opt.img_size_x, y_l=opt.img_size_x, win_len=opt.win_len, step=opt.step)
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
                #     show_Pic([mask[j, 0, :, :], 1-gen_parts[j, 1, :, :], 1-gen_parts[j, 0, :, :]], pic_order='13')

        # combine fracture to a layer FMI image
        img_dyna_gan_full = np.zeros((2000, 250))
        img_stat_gan_full = np.zeros((2000, 250))
        img_mask_gan_full = np.zeros((2000, 250))

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

        # show_Pic([1-img_dyna_gan_full, 1-img_stat_gan_full, img_mask_gan_full], pic_order='13')
        cv2.imwrite('D:\Data\image_simulated\simu-t-2-N\dyna-simu_{}_100_104.png'.format(index), (img_dyna_gan_full*255).astype(np.uint8)[:1600, :])
        cv2.imwrite('D:\Data\image_simulated\simu-t-2-N\stat-simu_{}_100_104.png'.format(index), (img_stat_gan_full*255).astype(np.uint8)[:1600, :])
        np.savetxt('D:\Data\image_simulated\simu-t-2-N\dyna-simu_{}_100_104.txt'.format(index), (img_dyna_gan_full*255).astype(np.uint8)[:1600, :],
                   comments='', delimiter='\t', fmt='%d', header='simu_1\n100\n104\nIMAGE.DYNA_SIMU\n\n\n\n\n')
        np.savetxt('D:\Data\image_simulated\simu-t-2-N\stat-simu_{}_100_104.txt'.format(index), (img_stat_gan_full*255).astype(np.uint8)[:1600, :],
                   comments='', delimiter='\t', fmt='%d', header='simu_1\n100\n104\nIMAGE.STAT_SIMU\n\n\n\n\n')
