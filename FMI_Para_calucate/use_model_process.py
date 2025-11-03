import os
import sys
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# 导入您的模型和数据集
from FMI_Para_calucate.UNet_FMI_paras_mask import GeneratorUNetImproved
from FMI_Para_calucate.dataloader_fmi_uses import dataloader_fmi_seg
from FMI_Para_calucate.loss_cal import MultiTaskLoss
from src_ele.file_operation import fmi_data_save


def setup_logging(log_dir):
    """
        设置日志系统 - 简化版本
        只返回Logger对象，不保存到文件

        参数:
        log_dir: 可选，为了兼容性保留
        phase: 可选，为了兼容性保留
    """
    # 创建日志器
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 清除现有的处理器，避免重复
    if logger.handlers:
        logger.handlers.clear()

    os.makedirs(log_dir, exist_ok=True)

    # 只创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # 添加处理器到日志器
    logger.addHandler(console_handler)

    return logger


def load_model(checkpoint_path, device, in_channels=2, out_channels=2, num_params=7, use_vit_attention=False, vit_patch_size=8):
    """加载训练好的模型"""
    # 初始化模型结构
    model = GeneratorUNetImproved(
        in_channels=in_channels,
        out_channels=out_channels,
        num_params=num_params,
        use_vit_attention=use_vit_attention,
        vit_patch_size=vit_patch_size
    ).to(device)

    # 加载模型权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"模型加载成功: {checkpoint_path}")
    return model


def process_directory(model, dataloader, criterion, device='cuda'):
    # 模型使用模式
    model.to(device)
    model.eval()

    # 进度条
    progress_bar = tqdm(dataloader, desc=f'处理FMI数据')
    result_mask = np.array([])
    result_paras = np.array([])
    all_losses = {
        'total_loss': [],
        'seg_loss': [],
        'param_loss': [],
        'dice_loss': [],
        'focal_loss': []
    }

    for batch_idx, batch in enumerate(progress_bar):
        # 准备数据
        inputs = {
            'fmi_image': batch['fmi_image'].to(device),
            'fmi_mask': batch['fmi_mask'].to(device),
            'fmi_params': batch['fmi_params'].to(device)
        }

        # 前向传播
        predictions = model(inputs['fmi_image'])

        # 计算损失
        loss_dict = criterion(predictions, inputs)
        loss_t = loss_dict['total_loss']
        loss_s = loss_dict['seg_loss']
        loss_p = loss_dict['param_loss']
        loss_d = loss_dict['dice_loss']
        loss_f = loss_dict['focal_loss']

        # 记录损失
        all_losses['total_loss'].append(loss_t.item())
        all_losses['seg_loss'].append(loss_s.item())
        all_losses['param_loss'].append(loss_p.item())
        all_losses['dice_loss'].append(loss_d.item())
        all_losses['focal_loss'].append(loss_f.item())

        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss_t.item():.6f}',
        })

        if result_mask.size == 0:
            result_mask = predictions['fmi_mask'].cpu().detach().numpy()
            result_paras = predictions['fmi_params'].cpu().detach().numpy()
        else:
            result_mask = np.concatenate((result_mask, predictions['fmi_mask'].cpu().detach().numpy()), axis=0)
            result_paras = np.concatenate((result_paras, predictions['fmi_params'].cpu().detach().numpy()), axis=0)

    return result_mask, result_paras


def main(args):
    args.vit_patch_size = args.image_length//32

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用设备: {device}")

    # 设置日志
    logger = setup_logging(args.path_output_dir)
    logger.info("开始电成像数据推理")
    logger.info(f"输入文件: {args.path_input_dyna}")
    logger.info(f"输出目录: {args.path_output_dir}")
    logger.info(f"模型路径: {args.checkpoint_path}")

    # 加载模型
    try:
        model = load_model(
            checkpoint_path=args.checkpoint_path,
            device=device,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            num_params=args.num_params,
            use_vit_attention=args.use_vit_attention,
            vit_patch_size=args.vit_patch_size,
        )
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return

    dataloader_o = dataloader_fmi_seg(path_dyna=args.path_input_dyna, windows_length=args.windows_length, windows_step=args.windows_step, out_shape=args.image_length, normalization_method='minmax')
    dataloader_u = DataLoader(dataloader_o, batch_size=32, shuffle=False, num_workers=1)

    # 初始化损失函数
    criterion = MultiTaskLoss(
        phase=1,  # 从阶段1开始
        seg_weight=args.seg_weight,
        param_weight=args.param_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
    )
    criterion.set_phase(3)

    # 处理目录中的所有FMI数据
    result_mask, result_paras = process_directory(
        model=model,
        dataloader=dataloader_u,
        criterion=criterion,
        device=device,
    )

    mask_cracks, mask_holes, model_paras_pred, fmi_dyna, fmi_stat, mask_cracks_origin, mask_holes_origin = dataloader_o.combine_model_data(result_mask, result_paras)

    file_name = args.path_input_dyna.split('/')[-1].split('\\')[-1]
    crack_path = os.path.join(args.path_output_dir, file_name.replace('.png', '_crack_mask.png'))
    hole_path = os.path.join(args.path_output_dir, file_name.replace('.png', '_hole_mask.png'))
    paras_path = os.path.join(args.path_output_dir, file_name.replace('.png', '_params.csv'))
    cracks_txt_path = os.path.join(args.path_output_dir, file_name.replace('dyna.png', '_cracks_mask.txt'))
    holes_txt_path = os.path.join(args.path_output_dir, file_name.replace('dyna.png', '_holes_mask.txt'))
    cracks_origin_txt_path = os.path.join(args.path_output_dir, file_name.replace('dyna.png', '_cracks_mask.txt'))
    holes_origin_txt_path = os.path.join(args.path_output_dir, file_name.replace('dyna.png', '_holes_mask.txt'))
    dyna_txt_path = os.path.join(args.path_output_dir, file_name.replace('dyna.png', '_dyna_origin.txt'))
    stat_txt_path = os.path.join(args.path_output_dir, file_name.replace('dyna.png', '_stat_origin.txt'))

    cv2.imwrite(crack_path, mask_cracks.astype(np.uint8))
    cv2.imwrite(hole_path, mask_holes.astype(np.uint8))
    model_paras_pred.to_csv(paras_path, float_format='%.6f', sep=',', index=False)

    fmi_data_save(mask_cracks, depth_config=None, path_save=cracks_txt_path)
    fmi_data_save(mask_holes, depth_config=None, path_save=holes_txt_path)
    fmi_data_save(mask_cracks_origin, depth_config=None, path_save=cracks_origin_txt_path)
    fmi_data_save(mask_holes_origin, depth_config=None, path_save=holes_origin_txt_path)
    fmi_data_save(fmi_dyna, depth_config=None, path_save=dyna_txt_path)
    fmi_data_save(fmi_stat, depth_config=None, path_save=stat_txt_path)

if __name__ == '__main__':

    """主函数"""
    # 参数解析
    parser = argparse.ArgumentParser(description='电成像多任务UNet模型推理')

    # 模型参数
    # parser.add_argument('--checkpoint_path', type=str, default=r'D:\GitHub\FMI_CGAN\FMI_Para_calucate\windows400-2\checkpoints\checkpoint_epoch_0020.pth', help='模型检查点路径')
    # parser.add_argument('--checkpoint_path', type=str, default=r'D:\GitHub\FMI_CGAN\FMI_Para_calucate\windows400-parashead-prompt\checkpoints\checkpoint_epoch_0020.pth', help='模型检查点路径')
    parser.add_argument('--checkpoint_path', type=str, default=r'D:\GitHub\FMI_CGAN\FMI_Para_calucate\windows400-4step-trained\checkpoints\checkpoint_epoch_0013.pth', help='模型检查点路径')
    parser.add_argument('--in_channels', type=int, default=2, help='输入通道数（动态+静态图像）')
    parser.add_argument('--out_channels', type=int, default=2, help='输出通道数（裂缝+孔洞掩码）')
    parser.add_argument('--num_params', type=int, default=7, help='参数数量')
    # parser.add_argument('--use_vit_attention', type=bool, default=False, help='使用ViT注意力机制')
    parser.add_argument('--use_vit_attention', type=bool, default=True, help='使用ViT注意力机制')
    parser.add_argument('--vit_patch_size', type=int, default=0, help='ViT的patch大小,这个要根据image_length计算，image_length//32')

    # 数据参数
    parser.add_argument('--path_input_dyna', type=str, default=r'F:\DeepLData\FMI_SIMULATION\simu_FMI\1_fmi_dyna.png', help='输入数据目录')
    # parser.add_argument('--path_input_dyna', type=str, default=r'F:\DeepLData\FMI_SIMULATION\simu_FMI\1_fmi_dyna.png', help='输入数据目录')
    parser.add_argument('--path_output_dir', type=str, default=r'F:\DeepLData\FMI_SIMULATION\simu_FMI_exteact', help='输出结果目录')
    # parser.add_argument('--image_length', type=int, default=128, help='输入图像尺寸')
    parser.add_argument('--image_length', type=int, default=160, help='输入图像尺寸')
    parser.add_argument('--windows_length', type=int, default=400, help='输入图像尺寸')
    parser.add_argument('--windows_step', type=int, default=10, help='输入图像尺寸')

    # loss函数参数
    parser.add_argument('--seg_weight', type=float, default=0.9, help='')
    parser.add_argument('--param_weight', type=float, default=0.1, help='')
    parser.add_argument('--dice_weight', type=float, default=0.5, help='')
    parser.add_argument('--focal_weight', type=float, default=0.5, help='')

    # 设备参数
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='推理设备')
    args = parser.parse_args()

    file_name = args.path_input_dyna.split('/')[-1].split('\\')[-1]
    crack_path = os.path.join(args.path_output_dir, file_name.replace('.png', '_crack_mask.png'))
    hole_path = os.path.join(args.path_output_dir, file_name.replace('.png', '_hole_mask.png'))
    paras_path = os.path.join(args.path_output_dir, file_name.replace('.png', '_params.csv'))

    main(args)