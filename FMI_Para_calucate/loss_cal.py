import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class DiceLoss(nn.Module):
    """
    Dice损失函数 - 专门处理分割任务中的类别不平衡

    原理:
    Dice系数 = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice损失 = 1 - Dice系数

    优势:
    - 直接优化预测区域和真实区域的重叠度
    - 对小目标敏感，适合裂缝和孔洞分割
    - 对类别不平衡鲁棒
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth  # 平滑项，防止除零

    def forward(self, predictions, targets):
        """
        计算Dice损失

        参数:
        predictions: 模型预测 [B, C, H, W]
        targets: 真实标签 [B, C, H, W]
        """
        # 确保输入在[0,1]范围内
        predictions = torch.clamp(predictions, 0, 1)
        targets = torch.clamp(targets, 0, 1)

        batch_size = predictions.size(0)
        num_classes = predictions.size(1)

        # 计算每个类别的Dice系数
        dice_losses = []
        for class_idx in range(num_classes):
            # 获取当前类别的预测和目标
            pred_class = predictions[:, class_idx]  # [B, H, W]
            target_class = targets[:, class_idx]  # [B, H, W]

            # 计算交集和并集
            intersection = (pred_class * target_class).sum(dim=(1, 2))  # [B]
            union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))  # [B]

            # 计算Dice系数（添加平滑项防止除零）
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B]
            dice_loss = 1 - dice  # [B]

            dice_losses.append(dice_loss.mean())  # 标量

        # 返回所有类别的平均Dice损失
        return torch.stack(dice_losses).mean()


class FocalLoss(nn.Module):
    """
    Focal损失函数 - 解决难易样本不平衡问题

    原理:
    在标准交叉熵损失基础上增加调制因子 (1 - pt)^γ
    其中pt是模型对真实类别的预测概率

    优势:
    - 降低易分类样本的权重，关注难分类样本
    - 特别适合电成像数据中稀疏的裂缝和孔洞
    - 提高模型对难样本的学习能力
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        参数:
        alpha: 平衡正负样本的权重（通常α < 0.5用于平衡类别）
        gamma: 调制因子，γ > 0减少易分类样本的权重
        reduction: 损失归约方式 ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        计算Focal损失

        参数:
        predictions: 模型预测 [B, C, H, W] (经过sigmoid，值在[0,1])
        targets: 真实标签 [B, C, H, W] (值在[0,1])
        """
        # 确保输入在有效范围内
        predictions = torch.clamp(predictions, 1e-6, 1 - 1e-6)  # 防止log(0)
        targets = torch.clamp(targets, 0, 1)

        # 计算二值交叉熵
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')

        # 计算调制因子
        pt = torch.exp(-bce_loss)  # pt = p if y=1, else 1-p

        # 计算Focal损失
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        # 应用归约
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveWeightScheduler:
    """
    自适应权重调度器 - 根据训练进度调整损失权重

    功能:
    - 在训练过程中动态调整不同损失的权重
    - 根据损失收敛情况自动平衡多任务
    - 提供灵活的调度策略
    """

    def __init__(self, initial_weights, strategy='linear'):
        self.initial_weights = initial_weights
        self.strategy = strategy
        self.epoch = 0

    def update_weights(self, epoch, loss_history):
        """根据训练进度更新权重"""
        self.epoch = epoch

        if self.strategy == 'linear':
            # 线性衰减参数回归权重，增加分割权重
            param_weight = max(0.01, self.initial_weights['param'] * (1 - epoch / 100))
            seg_weight = min(2.0, self.initial_weights['seg'] * (1 + epoch / 100))
            return {'param': param_weight, 'seg': seg_weight}

        elif self.strategy == 'adaptive':
            # 根据损失收敛情况自适应调整
            if len(loss_history) > 10:
                recent_seg_loss = np.mean(loss_history[-10:])
                recent_param_loss = np.mean(loss_history[-10:])

                # 如果分割损失收敛慢，增加其权重
                if recent_seg_loss > recent_param_loss * 2:
                    return {'param': 0.05, 'seg': 1.5}

        return self.initial_weights


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数，支持三阶段训练和电成像数据特性

    设计原理：
    1. 三阶段训练：阶段1（参数回归）→ 阶段2（分割）→ 阶段3（整体优化）
    2. 分割损失优化：使用Dice Loss + Focal Loss处理类别不平衡
    3. 参数回归损失：使用Huber Loss减少异常值影响
    4. 自适应权重：根据训练进度动态调整损失权重
    """

    def __init__(self, phase=1, seg_weight=0.1, param_weight=0.9, param_weights=None,
                 dice_weight=0.5, focal_weight=0.5, alpha=0.25, gamma=2.0):
        """
        初始化多任务损失函数

        参数:
        phase: 训练阶段 (1:参数回归, 2:分割训练, 3:整体训练)
        seg_weight: 分割损失总权重
        param_weight: 参数回归损失权重
        dice_weight: Dice损失在分割损失中的权重
        focal_weight: Focal损失在分割损失中的权重
        alpha: Focal Loss的alpha参数，控制正负样本权重
        gamma: Focal Loss的gamma参数，控制难易样本权重
        """
        super().__init__()

        self.phase = phase
        self.seg_weight_o = seg_weight
        self.param_weight_o = param_weight
        self.seg_weight = seg_weight
        self.param_weight = param_weight

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.alpha = alpha
        self.gamma = gamma

        # 分割损失组件：处理类别不平衡
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

        # 参数回归损失：对异常值更鲁棒
        self.param_loss = nn.SmoothL1Loss()

        self.paras_cols = ['crack_length', 'crack_width', 'crack_area', 'crack_density', 'hole_area', 'hole_density', 'hole_area_ratio']

        ## 损失历史记录（统一记录，不按阶段分开）
        self.loss_history = {
            'total_loss': [],
            'param_loss': [],
            'seg_loss': [],
            'dice_loss': [],
            'focal_loss': [],
        }
        for key in self.paras_cols:
            self.loss_history[key] = []

        print(f"多任务损失函数初始化完成 - 阶段{phase}")
        print(f"分割权重: {seg_weight}, 参数权重: {param_weight}")
        print(f"Dice权重: {dice_weight}, Focal权重: {focal_weight}")

    def set_phase(self, phase):
        """
        设置训练阶段

        阶段说明:
        - 阶段1: 只训练参数回归分支（冻结分割网络）
        - 阶段2: 只训练分割网络（冻结参数回归分支）
        - 阶段3: 整体训练（同时优化两个任务）
        """
        assert phase in [1, 2, 3, 4], "训练阶段必须是1、2、3或4"
        self.phase = phase
        print(f"切换到训练阶段 {phase}")

        # 根据阶段调整损失权重
        if phase == 1:
            # 阶段1重点关注参数回归
            self.seg_weight = 0.0
            self.param_weight = 1.0
        elif phase == 2:
            # 阶段2重点关注分割
            self.seg_weight = 1.0
            self.param_weight = 0.0
        elif phase == 3:
            # 阶段3平衡两个任务
            self.seg_weight = self.seg_weight_o
            self.param_weight = self.param_weight_o
        else:
            # 阶段4，只训练孔洞缝参数预测头，冻结所有的UNET网络
            self.seg_weight = 0.0
            self.param_weight = 1.0

    def calculate_segmentation_loss(self, predictions, targets):
        """
        计算分割损失 - 专门处理电成像数据的类别不平衡问题

        电成像数据特点：
        - 裂缝和孔洞通常只占图像的很小部分（<5%）
        - 背景区域占主导地位，容易导致模型忽略小目标
        - 使用组合损失解决这个问题

        损失组合：
        - Dice Loss: 直接优化分割区域的重叠度，对小目标敏感
        - Focal Loss: 调整难易样本权重，关注难分类的像素
        """
        # 输入验证
        assert predictions.shape == targets.shape, "预测和目标形状必须一致"
        assert predictions.min() >= 0 and predictions.max() <= 1, "预测值必须在[0,1]范围内"
        assert targets.min() >= 0 and targets.max() <= 1, "目标值必须在[0,1]范围内"

        # 计算Dice损失（对类别不平衡鲁棒）
        dice_loss = self.dice_loss(predictions, targets)

        # 计算Focal损失（关注难分类样本）
        focal_loss = self.focal_loss(predictions, targets)

        # 组合损失
        seg_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss

        return {
            'seg_loss': seg_loss,
            'dice_loss': dice_loss,
            'focal_loss': focal_loss
        }

    def calculate_parameter_loss(self, pred_params, target_params):
        """
        计算参数回归损失
        电成像参数特点：
        - 7个参数已经通过dataloader进行了minmax 或 zscore 的归一化处理
        - 使用SmoothL1Loss（Huber Loss）对异常值更鲁棒
        """
        # 输入验证
        assert pred_params.shape == target_params.shape, "参数预测和目标形状必须一致"

        # 使用SmoothL1Loss（结合了L1和L2损失的优点）
        param_loss = self.param_loss(pred_params, target_params)

        # 可选：添加梯度裁剪（防止极端情况下的梯度爆炸）
        if param_loss > 10.0:  # 设置一个合理的阈值
            param_loss = torch.clamp(param_loss, max=10.0)
            # print(f"参数损失裁剪: {param_loss.item()}")

        # 计算每一个参数的Loss
        for i, name in enumerate(self.paras_cols):
            individual_loss = self.param_loss(pred_params[:, i], target_params[:, i])
            self.loss_history[name].append(individual_loss.item())

        return {
            'param_loss': param_loss,
        }

    def forward(self, predictions, targets):
        """
        前向传播 - 根据训练阶段计算相应的损失

        参数:
        predictions: 模型输出字典
            - 'segmentation': 分割预测 [B, 2, H, W]
            - 'parameters': 参数预测 [B, 7]
        targets: 目标字典
            - 'segmentation': 分割目标 [B, 2, H, W]
            - 'parameters': 参数目标 [B, 7]
        """
        losses = {}

        seg_loss_dict = self.calculate_segmentation_loss(predictions['fmi_mask'], targets['fmi_mask'])
        losses.update(seg_loss_dict)

        param_loss_dict = self.calculate_parameter_loss(predictions['fmi_params'], targets['fmi_params'])
        losses.update(param_loss_dict)

        # 根据阶段计算总损失
        losses['total_loss'] = (self.seg_weight * losses['seg_loss'] + self.param_weight * losses['param_loss'])

        # 记录损失历史
        self.loss_history['total_loss'].append(losses['total_loss'].item())
        self.loss_history['seg_loss'].append(losses['seg_loss'].item())
        self.loss_history['dice_loss'].append(losses['dice_loss'].item())
        self.loss_history['focal_loss'].append(losses['focal_loss'].item())
        self.loss_history['param_loss'].append(losses['param_loss'].item())

        return losses

    def get_loss_history(self):
        """获取损失历史记录，用于分析和可视化"""
        return self.loss_history





# 使用示例和测试代码
if __name__ == '__main__':
    # 测试设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化损失函数
    criterion = MultiTaskLoss(
        seg_weight=1.0,
        param_weight=0.1,
        dice_weight=0.6,  # 给Dice损失更高权重，更适合小目标
        focal_weight=0.4,
        alpha=0.25,
        gamma=2.0
    )

    # 测试不同训练阶段
    for phase in [1, 2]:
        print(f"\n=== 测试训练阶段 {phase} ===")

        # 测试阶段切换
        criterion.set_phase(phase)
        print(f"已切换到阶段 {phase}")

        # 创建模拟数据
        batch_size = 5
        img_size = 256

        # 模拟参数数据（已归一化）
        pred_params = torch.randn(batch_size, 7).to(device)
        target_params = torch.randn(batch_size, 7).to(device)
        # 模拟分割数据（模拟稀疏目标）
        pred_seg = torch.rand(batch_size, 2, img_size, img_size).to(device)
        # 创建稀疏目标：大部分为0，小部分为1（模拟裂缝和孔洞）
        target_seg = torch.bernoulli(torch.full((batch_size, 2, img_size, img_size), 0.05)).to(device)
        # 构建模型输入、输出字典
        predictions = {'fmi_params':pred_params,  'fmi_mask':pred_seg}
        targets = {'fmi_params':target_params, 'fmi_mask':target_seg}

        # 计算损失
        loss_dict = criterion(predictions, targets)

        # 打印结果
        print(f"总损失: {loss_dict['total_loss'].item():.6f}")
        for key, value in loss_dict.items():
            if key != 'total_loss':
                print(f"\t{key}: {value.item():.6f}")

        # 测试反向传播
        if loss_dict['total_loss'].requires_grad:
            loss_dict['total_loss'].backward()
            print("梯度反向传播测试成功")

    # 测试损失历史记录
    print(f"\n=== 测试损失历史记录 ===")
    history = criterion.get_loss_history()
    for phase, phase_history in history.items():
        print(f"loss类型 {phase} : {phase_history}")

