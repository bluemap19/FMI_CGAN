import json
import os
import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datetime import datetime
from tqdm import tqdm

from FMI_Para_calucate.UNet_FMI_paras_mask import GeneratorUNetImproved
from FMI_Para_calucate.dataloader_fmi_and_para_list import dataloader_FMI_logging
from FMI_Para_calucate.loss_cal import MultiTaskLoss



def setup_logging(log_dir, phase):
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件名（包含时间戳和阶段）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_phase{phase}_{timestamp}.log")

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, is_best=False):
    """保存模型检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),   # 这两个所需要消耗的空间太大了，只能给注释掉了
        # 'loss': loss,                                     # 这两个所需要消耗的空间太大了，只能给注释掉了
        'timestamp': datetime.now().isoformat()
    }

    # 常规检查点
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:04d}.pth')
    torch.save(checkpoint, checkpoint_path)

    # 最佳模型
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)

    return checkpoint_path


def visualize_segmentation_results(epoch, phase, inputs, predictions, save_dir):
    """可视化分割结果"""
    os.makedirs(save_dir, exist_ok=True)

    # 提取输入图像、预测分割和目标分割
    input_image = inputs['fmi_image']  # [B, 2, H, W]
    target_seg = inputs['fmi_mask']  # [B, 2, H, W]
    pred_seg = predictions['fmi_mask']  # [B, 2, H, W]

    # Save sample
    imgs = torch.cat((input_image.data, target_seg.data, pred_seg.data), 1)
    imgs = imgs.reshape((imgs.shape[0] * imgs.shape[1], 1, imgs.shape[-2], imgs.shape[-1]))

    # 保存图像
    save_path = os.path.join(save_dir, f'segmentation_phase{phase}_epoch{epoch:04d}.png')
    save_image(imgs, save_path, nrow=6, padding=1, normalize=True)

    return save_path



def train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                phase, args, logger, start_epoch=0):
    """训练一个阶段"""
    logger.info(f"开始阶段 {phase} 训练")

    # 根据阶段设置模型状态
    if phase == 1:
        # 阶段1: 只训练参数回归分支
        model.unfreeze_all()
        model.freeze_segbone()
        criterion.set_phase(phase)
    elif phase == 2:
        # 阶段2: 只训练分割网络
        model.unfreeze_all()
        model.freeze_backbone()
        criterion.set_phase(phase)
    elif phase == 3:
        # 阶段3: 整体训练
        model.unfreeze_all()
        criterion.set_phase(phase)
    elif phase == 4:
        # 阶段4：冻结UNET，只训练参数预测头
        model.freeze_unet()
        criterion.set_phase(phase)
    else:
        print('error phase as :{}'.format(phase))
        exit(0)


    # 训练记录
    best_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, start_epoch + args.epochs_per_phase):
        # 训练模式
        model.train()
        epoch_train_loss = 0
        num_batches = 0

        # 进度条
        progress_bar = tqdm(train_loader, desc=f'阶段{phase} 轮次{epoch + 1}/{start_epoch + args.epochs_per_phase}')

        for batch_idx, batch in enumerate(progress_bar):
            # 准备数据
            inputs = {
                'fmi_image': batch['fmi_image'].to(device),
                'fmi_mask': batch['fmi_mask'].to(device),
                'fmi_params': batch['fmi_params'].to(device)
            }

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            predictions = model(inputs['fmi_image'])

            # 计算损失
            loss_dict = criterion(predictions, inputs)
            loss = loss_dict['total_loss']
            loss_params = loss_dict['param_loss']
            loss_seg = loss_dict['seg_loss']

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 更新权重
            optimizer.step()

            # 更新进度条
            progress_bar.set_postfix({
                'loss_total': f'{loss.item():.6f}',
                'loss_seg': f'{loss_seg.item():.6f}',
                'loss_params': f'{loss_params.item():.6f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

            # 记录损失
            epoch_train_loss += loss.item()
            num_batches += 1

        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)

        # 验证模式
        model.eval()
        epoch_val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for val_batch in val_loader:
                # 准备验证数据
                inputs = {
                    'fmi_image': val_batch['fmi_image'].to(device),
                    'fmi_mask': val_batch['fmi_mask'].to(device),
                    'fmi_params': val_batch['fmi_params'].to(device)
                }

                # 前向传播
                predictions = model(inputs['fmi_image'])

                # 计算损失
                loss_dict = criterion(predictions, inputs)
                loss = loss_dict['total_loss']

                # 记录损失
                epoch_val_loss += loss.item()
                num_val_batches += 1

        # 计算平均验证损失和指标
        avg_val_loss = epoch_val_loss / num_val_batches
        val_losses.append(avg_val_loss)

        # 更新学习率
        scheduler.step(avg_val_loss)

        # 记录日志
        logger.info(f"阶段{phase} 轮次{epoch + 1:02d} - "
                    f"训练损失: {avg_train_loss:.6f}, "
                    f"验证损失: {avg_val_loss:.6f}, "
                    f"学习率: {optimizer.param_groups[0]['lr']:.2e}")

        # 保存检查点
        is_best = avg_val_loss < best_loss
        if is_best:
            best_loss = avg_val_loss

        checkpoint_path = save_checkpoint(
            model, optimizer, epoch + 1, avg_val_loss,
            os.path.join(args.checkpoint_dir, f'phase{phase}'),
            is_best=is_best
        )
        logger.info(f"检查点已保存: {checkpoint_path}")

        # 可视化结果（每5个轮次或最佳模型）
        if (epoch + 1) % 5 == 0 or is_best:
            visualize_path = visualize_segmentation_results(
                epoch + 1, phase, inputs, predictions,
                os.path.join(args.visualization_dir, f'phase{phase}')
            )
            logger.info(f"可视化结果已保存: {visualize_path}")

    logger.info(f"阶段 {phase} 训练完成，最佳验证损失: {best_loss:.6f}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_loss': best_loss,
    }


def save_loss_history_csv(loss_history, file_charter='', save_dir="./loss_history"):
    """
    使用Pandas保存为CSV文件（便于数据分析）
    """
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"loss_history_{timestamp}_{file_charter}.csv"
    filepath = os.path.join(save_dir, filename)

    # 转换为DataFrame
    df = pd.DataFrame(loss_history)

    # 保存为CSV
    df.to_csv(filepath, index=False)

    print(f"损失历史已保存到: {filepath}")
    return filepath

def main():
    """主函数"""
    # 参数解析
    parser = argparse.ArgumentParser(description='电成像多任务UNet模型训练')

    # 数据参数
    # parser.add_argument('--train_data_dir', type=str, default=r'F:\DeepLData\FMI_SIMULATION\simu_FMI_2', help='训练数据目录')
    parser.add_argument('--train_data_dir', type=str, default=r'F:\DeepLData\FMI_SIMULATION\simu_FMI_3', help='训练数据目录')
    parser.add_argument('--val_data_dir', type=str, default=r'F:\DeepLData\FMI_SIMULATION\simu_FMI', help='验证数据目录')
    parser.add_argument('--data_iter_windows_length', type=int, default=400, help='数据处理的窗长大小')
    parser.add_argument('--batch_size', type=int, default=52, help='批处理大小')
    parser.add_argument('--image_length', type=int, default=160, help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=1, help='数据加载工作线程数')

    # 模型参数
    parser.add_argument('--in_channels', type=int, default=2, help='输入通道数（动态+静态图像）')
    parser.add_argument('--out_channels', type=int, default=2, help='输出通道数（裂缝+孔洞掩码）')
    parser.add_argument('--num_params', type=int, default=7, help='参数数量')
    parser.add_argument('--use_vit_attention', type=bool, default=True, help='使用ViT注意力机制')
    parser.add_argument('--vit_patch_size', type=int, default=0, help='ViT的patch大小,这个要根据image_length计算，image_length//32')

    # 训练参数
    parser.add_argument('--epochs_per_phase', type=int, default=1, help='每个阶段的训练轮次')
    parser.add_argument('--learning_rate', type=float, default=0.001, help = '初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')

    # 损失函数参数
    parser.add_argument('--seg_weight', type=float, default=0.95, help='分割损失权重')
    parser.add_argument('--param_weight', type=float, default=0.05, help='参数回归损失权重')
    parser.add_argument('--dice_weight', type=float, default=0.6, help='Dice损失权重')
    parser.add_argument('--focal_weight', type=float, default=0.4, help='Focal损失权重')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Focal Loss的alpha参数')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss的gamma参数')

    # 输出参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志保存目录')
    parser.add_argument('--visualization_dir', type=str, default='./visualizations', help='可视化结果保存目录')

    # 设备参数
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='训练设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    args.vit_patch_size = args.image_length//32

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # 创建输出目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.visualization_dir, exist_ok=True)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用设备: {device}")

    # 设置日志
    logger = setup_logging(args.log_dir, 0)  # 阶段0表示整体训练日志

    # 记录参数
    logger.info("训练参数:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")

    # 加载数据集
    logger.info("加载数据集...")

    train_dataset = dataloader_FMI_logging(
        path_folder=args.train_data_dir,
        len_windows=args.data_iter_windows_length,
        step_windows=200,
        out_shape=args.image_length,
        normalize_params=True,
        normalization_method='minmax',
    )

    val_dataset = dataloader_FMI_logging(
        path_folder=args.val_data_dir,
        len_windows=args.data_iter_windows_length,
        step_windows=200,
        out_shape=args.image_length,
        normalize_params=True,
        normalization_method='minmax'
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logger.info(f"训练集: {len(train_dataset)}个样本")
    logger.info(f"验证集: {len(val_dataset)}个样本")

    # 初始化模型
    logger.info("初始化模型...")
    model = GeneratorUNetImproved(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        num_params=args.num_params,
        use_vit_attention=args.use_vit_attention,
        vit_patch_size=args.vit_patch_size
    ).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params / 1e6:.2f}M")
    logger.info(f"可训练参数量: {trainable_params / 1e6:.2f}M")

    # 初始化优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 初始化损失函数
    criterion = MultiTaskLoss(
        phase=1,  # 从阶段1开始
        seg_weight=args.seg_weight,
        param_weight=args.param_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
        alpha=args.focal_alpha,
        gamma=args.focal_gamma
    )

    # 二阶段 循环 训练
    current_epoch = 0
    total_cycle_epoch = 5
    for i in range(total_cycle_epoch):
        if i < int(0.6*total_cycle_epoch):
            # 阶段1: 只训练参数回归
            logger.info("=== 开始阶段1训练 (参数回归) ===")
            phase1_results = train_phase(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, 1, args, logger, current_epoch
            )
            current_epoch += args.epochs_per_phase

            # 阶段2: 只训练分割网络
            logger.info("=== 开始阶段2训练 (分割网络) ===")
            phase2_results = train_phase(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, 2, args, logger, current_epoch
            )
            current_epoch += args.epochs_per_phase

            # 阶段3: 整体训练
            logger.info("=== 开始阶段3训练 (整体训练) ===")
            phase3_results = train_phase(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, 3, args, logger, current_epoch
            )
            current_epoch += args.epochs_per_phase
        else:
            # 阶段3: 整体训练
            logger.info("=== 开始阶段3训练 (整体训练) ===")
            phase3_results = train_phase(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, 3, args, logger, current_epoch
            )
            current_epoch += args.epochs_per_phase

            # 阶段4: 整体训练，但是保留
            logger.info("=== 开始阶段4训练 (整体训练) ===")
            phase4_results = train_phase(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, 4, args, logger, current_epoch
            )
            current_epoch += args.epochs_per_phase
            phase3_results.update(phase4_results)

        # 保存最终模型
        final_checkpoint_path = save_checkpoint(model, optimizer, current_epoch, phase3_results['val_losses'], args.checkpoint_dir, is_best=True)
        logger.info(f"最终模型已保存: {final_checkpoint_path}")

    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存训练结果
    loss_history = criterion.get_loss_history()
    save_loss_history_csv(loss_history, file_charter='cycle-train')

    # 1. 保存训练参数
    args_dict = vars(args)
    args_file = os.path.join(args.log_dir, f'training_args_{timestamp}.json')
    with open(args_file, 'w', encoding='utf-8') as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"训练参数已保存: {args_file}")

    # 2. 保存模型结构
    model_file = os.path.join(args.log_dir, f'model_summary_{timestamp}.txt')
    with open(model_file, 'w', encoding='utf-8') as f:
        f.write(f"模型名称: {model.__class__.__name__}\n")
        f.write(f"总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")
        f.write(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M\n")
        f.write("\n模型结构:\n")
        f.write(str(model))
    logger.info(f"模型结构已保存: {model_file}")


if __name__ == '__main__':
    main()