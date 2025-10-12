import argparse
import os

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from src_ele.pic_opeeration import show_Pic
from train_repair_CGAN_model.Dataloader_FMI_add_empty_stripe import dataloader_FMI_logging
from train_repair_CGAN_model.MODEL_Generator_UNET import GeneratorUNet


def repair_images_with_model(model_path, input_path, output_path, device='cuda', len_windows=128, padding_length=16, step_windows=10, batch_size=32):
    """
    使用训练好的图像修复模型修复图像

    参数:
        model_path (str): 训练好的生成器模型路径
        input_path (str): 包含待修复图像的路径
        output_path (str): 保存修复后图像的路径
        device (str): 使用的设备（'cuda'或'cpu'）
    """
    # 加载模型
    generator = load_generator_model(model_path, device)

    dataloader_o = dataloader_FMI_logging(input_path, len_windows=256, step_windows=step_windows, pic_length_target=len_windows, padding=padding_length)
    dataloader = DataLoader(dataloader_o, shuffle=False, batch_size=batch_size, drop_last=False, pin_memory=False, num_workers=0)


    data_generate_list = []
    for i, batch in enumerate(dataloader):
        # 将数据移动到与模型相同的设备
        real_all = batch["img_origin"].to(device)  # 完整的图像
        real_input = batch["img_masked"].to(device)  # 图像被遮掩后的部分，也是图像修复模型的输入
        mask = batch["img_mask"].to(device)  # 图像的空白条带掩码

        print(real_all.shape, real_input.shape, mask.shape)

        # 拼接输入
        data_input = torch.cat((real_input, mask), dim=1)
        model_generate_all = generator(data_input)

        # # 1. 从计算图中分离张量，避免影响梯度计算
        # with torch.no_grad():
        #     # 选择第一个样本进行展示
        #     real_all_np = real_all[0, 0].detach().cpu().numpy()
        #     real_input_np = real_input[0, 0].detach().cpu().numpy()
        #     mask_np = mask[0, 0].detach().cpu().numpy()
        #     model_generate_all_np = model_generate_all[0, 0].detach().cpu().numpy()
        #     show_Pic([real_all_np, real_input_np, mask_np, model_generate_all_np])
        if i == 0:
            data_generate_list = model_generate_all.detach().cpu().numpy()
        else:
            model_generate_all = model_generate_all.detach().cpu().numpy()
            data_generate_list = np.concatenate((data_generate_list, model_generate_all), axis=0)
            print(data_generate_list.shape)

    pic_result, pic_result_target = dataloader_o.combine_pic_list(data_generate_list, path_target_folder=output_path)
    print(pic_result.shape, pic_result_target.shape)
    print(f"所有图像修复完成！结果保存在: {output_path}")


def load_generator_model(model_path, device='cuda'):
    """
    加载训练好的生成器模型

    参数:
        model_path (str): 模型文件路径
        img_length (int): 图像尺寸
        device (str): 使用的设备

    返回:
        GeneratorUNet: 加载的生成器模型
    """
    # 初始化模型
    generator = GeneratorUNet(in_channels=2, out_channels=1)  # 输入通道为2（图像+掩码），输出通道为1

    # 加载模型权重
    if os.path.exists(model_path):
        # 根据设备选择加载方式
        if device == 'cuda':
            generator.load_state_dict(torch.load(model_path))
        else:
            generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        generator.to(device)
        generator.eval()  # 设置为评估模式
        print(f"成功加载模型: {model_path}")
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    return generator


if __name__ == '__main__':
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='使用训练好的CGAN模型修复图像')
    parser.add_argument('--model_path', type=str, default=r'D:\GitHub\FMI_CGAN\train_repair_CGAN_model\saved_models\FMI_CGAN_repair\4\model_ele_gen_22800.pth', help='生成器模型路径')
    parser.add_argument('--input_path', type=str, default=r'F:\DeepLData\FMI_SIMULATION\simu_FMI\0_fmi_dyna.png', help='待修复图像目录')
    parser.add_argument('--output_path', type=str, default=r'F:\DeepLData', help='修复后图像保存目录')
    parser.add_argument('--img_length', type=int, default=128, help='图像尺寸')
    parser.add_argument('--padding', type=int, default=16, help='FMI填充尺寸')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='使用的设备')

    args = parser.parse_args()

    # 检查设备可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        args.device = 'cpu'
    else:
        print('使用GPU加速进行图像修复')

    # 运行修复过程
    repair_images_with_model(
        model_path=args.model_path,
        input_path=args.input_path,
        output_path=args.output_path,
        device=args.device,
        len_windows=args.img_length,
        step_windows=20,
        padding_length=args.padding,
    )