import argparse
import os

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from src_ele.pic_opeeration import show_Pic
from train_repair_CGAN_model.Dataloader_FMI_add_empty_stripe import dataloader_FMI_logging, dataloader_FMI_logging_no_repeat
from train_repair_CGAN_model.MODEL_Generator_UNET import GeneratorUNet
from train_repair_CGAN_model.MODEL_Generator_prompt2 import GeneratorUNetImproved


def repair_images_with_model(model_path, input_path, output_path, save_charter='', device='cuda', padding_length=16, len_windows=256, step_windows=10, pic_length_target=128, mask_config={'ratio_empty':0.4, 'num_belt':6}):
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

    # (path=r'0_fmi_dyna.png', padding=16, len_windows=256, step_windows=10, pic_length_target=128, mask_config={'ratio_empty':0.2, 'num_belt':6, 'rotate_rb':True})
    # dataloader_o = dataloader_FMI_logging(input_path, padding=padding_length, len_windows=len_windows, step_windows=step_windows, pic_length_target=pic_length_target, mask_config=mask_config)
    dataloader_o = dataloader_FMI_logging_no_repeat(input_path, padding=padding_length, len_windows=len_windows, pic_length_target=pic_length_target, mask_config=mask_config)
    dataloader = DataLoader(dataloader_o, shuffle=False, batch_size=32, drop_last=False, pin_memory=False, num_workers=0)

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

    pic_result, pic_result_target = dataloader_o.combine_pic_list(data_generate_list, path_target_folder=output_path, str_charter=save_charter)
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
    # generator = GeneratorUNet(in_channels=2, out_channels=1)  # 输入通道为2（图像+掩码），输出通道为1
    generator = GeneratorUNetImproved(in_channels=2, out_channels=1)  # 输入通道为2（图像+掩码），输出通道为1

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
    parser.add_argument('--model_path', type=str, default=r'D:\GitHub\FMI_CGAN\train_repair_CGAN_model\saved_models\FMI_CGAN_repair\6\model_ele_gen_2800.pth', help='生成器模型路径')
    parser.add_argument('--input_path', type=str, default=r'F:\DeepLData\FMI_SIMULATION\simu_FMI\0_fmi_dyna.png', help='待修复图像目录')
    # parser.add_argument('--input_path', type=str, default=r'F:\DeepLData\FMI_SIMULATION\simu_cracks\0_background_mask.png', help='待修复图像目录')
    parser.add_argument('--output_path', type=str, default=r'F:\DeepLData\pic_repair_paper_effect', help='修复后图像保存目录')
    parser.add_argument('--saved_charter', type=str, default=r'', help='修复后图像保存目录')
    parser.add_argument('--length_windows', type=int, default=250, help='图像尺寸，这个可以随意修改')
    parser.add_argument('--length_padding', type=int, default=16, help='FMI填充尺寸，用来进行辅助修复，这个可以随意修改')
    parser.add_argument('--step_windows', type=int, default=10, help='模型窗口遍历时使用的步长大小，这个要小于length_windows')
    parser.add_argument('--length_target', type=int, default=128, help='模型输入数据尺寸，这个不能修改，只能根据模型参数来进行设置')
    parser.add_argument('--ratio_empty', type=float, default=0.4, help='图像的空白率')
    parser.add_argument('--num_belt', type=int, default=6, help='图像空白带条数')
    parser.add_argument('--rotate_rb', type=bool, default=True, help='图像空白带是否进行旋转，模拟更加真实的空白带水平')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='使用的设备')
    args = parser.parse_args()

    # 检查设备可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        args.device = 'cpu'
    else:
        print('使用GPU加速进行图像修复')

    if args.saved_charter == '':
        args.saved_charter = args.input_path.split('/')[-1].split('\\')[-1].split('.')[0]
    path_save = args.output_path + '\\' + args.saved_charter
    if not os.path.exists(path_save):
        os.mkdir(path_save)
        print(path_save)

    # 运行修复过程
    repair_images_with_model(
        model_path=args.model_path,
        input_path=args.input_path,
        output_path=path_save,
        save_charter= args.saved_charter,
        device=args.device,
        padding_length=args.length_padding,
        len_windows=args.length_windows,
        step_windows=args.step_windows,
        pic_length_target=args.length_target,
        mask_config={'ratio_empty': args.ratio_empty, 'num_belt': args.num_belt},
    )