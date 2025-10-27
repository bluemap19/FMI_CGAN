import argparse

from src_ele.dir_operation import get_all_file_paths
from use_cgan_model_create_fmi.use_pix2pix_gen_FMI_new import use_pix2pix_gen_FMI

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--img_size_x", type=int, default=256, help="size of image height")
    parser.add_argument("--img_size_y", type=int, default=256, help="size of image width")
    parser.add_argument("--channels_in", type=int, default=4, help="number of image channels")
    parser.add_argument("--channels_out", type=int, default=2, help="number of image channels")
    parser.add_argument("--dataset_path", type=str, default=r"", help="path of the dataset")
    parser.add_argument("--netG", type=str, default=r'D:\GitHub\FMI_CGAN\train_pix2pix_simulate\saved_models\pic2pic\3\best_generator.pth', help="netG path to load")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=1, help="num of cpu to process input data")
    parser.add_argument("--win_len", type=int, default=250, help="windows length to walk through the full layer mask")
    opt = parser.parse_args()
    print(opt)


    path_target_folder = r'F:\DeepLData\FMI_SIMULATION\simu_cracks'
    list_all_file = get_all_file_paths(path_target_folder)
    print(list_all_file)
    for file_name in list_all_file:
        if (file_name.endswith('.png')) and (file_name.__contains__('_background_mask.')):
            opt.dataset_path = file_name
            use_pix2pix_gen_FMI(opt)