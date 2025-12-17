import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt

from src_ele.dir_operation import search_files_by_criteria
from src_ele.pic_opeeration import show_Pic


def preprocess_inputs(img1, img2, mask=None):
    """
    预处理输入图像和掩码

    参数:
        img1 (np.ndarray): 图像1 (H, W) 或 (H, W, C)
        img2 (np.ndarray): 图像2 (H, W) 或 (H, W, C)
        mask (np.ndarray): 掩码图像 (H, W) 或 (H, W, C)

    返回:
        tuple: (img1, img2, mask) 转换为 (C, H, W) 格式
    """
    # 确保图像类型为 float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # 处理单通道图像
    if img1.ndim == 2:
        img1 = img1[np.newaxis, :, :]  # (1, H, W)
    elif img1.ndim == 3:
        # 转换通道顺序 (H, W, C) -> (C, H, W)
        img1 = np.transpose(img1, (2, 0, 1))

    if img2.ndim == 2:
        img2 = img2[np.newaxis, :, :]  # (1, H, W)
    elif img2.ndim == 3:
        img2 = np.transpose(img2, (2, 0, 1))

    # 处理掩码
    if mask is not None:
        mask = mask.astype(np.float32)
        if mask.ndim == 2:
            mask = mask[np.newaxis, :, :]  # (1, H, W)
        elif mask.ndim == 3:
            # 如果掩码是多通道的，取第一个通道
            if mask.shape[2] > 1:
                mask = mask[:, :, 0]
            mask = mask[np.newaxis, :, :]  # (1, H, W)
    else:
        # 如果没有掩码，创建一个全1的掩码
        mask = np.ones_like(img1[0])[np.newaxis, :, :]
    if np.max(mask) > 250:
        mask = mask / 250.0

    return img1, img2, mask


def calculate_mse(img1, img2, mask=None):
    """
    计算均方误差 (MSE)

    参数:
        img1 (np.ndarray): 原始图像 (H, W) 或 (H, W, C)
        img2 (np.ndarray): 修复后图像 (H, W) 或 (H, W, C)
        mask (np.ndarray): 掩码图像 (H, W) 或 (H, W, C)

    返回:
        float: MSE值
    """
    # 预处理输入
    img1, img2, mask = preprocess_inputs(img1, img2, mask)

    # 计算掩码区域的MSE
    diff = (img1 - img2) * mask
    squared_diff = diff ** 2
    mse = np.sum(squared_diff) / np.sum(mask)

    return mse

def calculate_mae(img1, img2, mask=None):
    """
    计算均方误差 (MSE)

    参数:
        img1 (np.ndarray): 原始图像 (H, W) 或 (H, W, C)
        img2 (np.ndarray): 修复后图像 (H, W) 或 (H, W, C)
        mask (np.ndarray): 掩码图像 (H, W) 或 (H, W, C)
    返回:
        float: MAE值
    """
    # 预处理输入
    img1, img2, mask = preprocess_inputs(img1, img2, mask)

    # 计算掩码区域的MSE
    diff = (img1 - img2) * mask
    mse = np.sum(diff) / np.sum(mask)

    return mse

def calculate_psnr(img1, img2, mask=None):
    """
    计算峰值信噪比 (PSNR)

    参数:
        img1 (np.ndarray): 原始图像 (H, W) 或 (H, W, C)
        img2 (np.ndarray): 修复后图像 (H, W) 或 (H, W, C)
        mask (np.ndarray): 掩码图像 (H, W) 或 (H, W, C)
        data_range (float): 图像数据的最大范围，默认为1.0（归一化图像）

    返回:
        float: PSNR值 (dB)
    """
    # 计算MSE
    mse = calculate_mse(img1, img2, mask)

    # 避免除以零
    if mse == 0:
        return float('inf')

    # 计算PSNR
    data_range = np.max(img1) - np.min(img1)
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    return psnr


def calculate_ssim(img1, img2, mask=None, data_range=1.0, window_size=11, gaussian_sigma=1.5):
    """
    计算结构相似性指数 (SSIM)

    参数:
        img1 (np.ndarray): 原始图像 (H, W) 或 (H, W, C)
        img2 (np.ndarray): 修复后图像 (H, W) 或 (H, W, C)
        mask (np.ndarray): 掩码图像 (H, W) 或 (H, W, C)
        data_range (float): 图像数据的最大范围，默认为1.0（归一化图像）
        window_size (int): 高斯窗口大小
        gaussian_sigma (float): 高斯核的标准差

    返回:
        float: SSIM值
    """
    # 预处理输入
    img1, img2, mask = preprocess_inputs(img1, img2, mask)

    # 创建高斯窗口
    def create_gaussian_window(window_size, sigma):
        """创建高斯窗口"""
        gauss = np.array([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
        gauss /= gauss.sum()
        window = np.outer(gauss, gauss)
        return window / window.sum()

    window = create_gaussian_window(window_size, gaussian_sigma)

    # 计算局部统计量
    def compute_local_stats(img, window):
        """计算局部均值和方差"""
        mu = signal.convolve2d(img, window, mode='same', boundary='symm')
        sigma_sq = signal.convolve2d(img ** 2, window, mode='same', boundary='symm') - mu ** 2
        return mu, sigma_sq

    # 初始化SSIM值
    ssim_total = 0.0
    num_channels = img1.shape[0]

    # 对每个通道计算SSIM
    for c in range(num_channels):
        channel1 = img1[c]
        channel2 = img2[c]
        channel_mask = mask[0]  # 掩码对所有通道相同

        # 计算局部统计量
        mu1, sigma1_sq = compute_local_stats(channel1, window)
        mu2, sigma2_sq = compute_local_stats(channel2, window)
        sigma12 = signal.convolve2d(channel1 * channel2, window, mode='same', boundary='symm') - mu1 * mu2

        # SSIM常数
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        # 计算SSIM图
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

        # 计算掩码区域的SSIM
        ssim_val = np.sum(ssim_map * channel_mask) / np.sum(channel_mask)
        ssim_total += ssim_val

    # 计算平均SSIM
    return ssim_total / num_channels


def evaluate_repair_quality(original_imgs, repaired_imgs, masks=None):
    """
    综合评估图像修复质量

    参数:
        original_imgs (np.ndarray): 原始图像 (H, W) 或 (H, W, C)
        repaired_imgs (np.ndarray): 修复后图像 (H, W) 或 (H, W, C)
        masks (np.ndarray): 掩码图像 (H, W) 或 (H, W, C)

    返回:
        dict: 包含MSE, PSNR, SSIM的字典
    """
    results = {}

    # 计算MSE
    results['mse'] = calculate_mse(original_imgs, repaired_imgs, masks)

    # 计算MAE
    results['mae'] = calculate_mae(original_imgs, repaired_imgs, masks)

    # 计算PSNR
    results['psnr'] = calculate_psnr(original_imgs, repaired_imgs, masks)

    # 计算SSIM
    results['ssim'] = calculate_ssim(original_imgs, repaired_imgs, masks)

    return results


def generate_test_images(img_size=128, channels=1, mask_ratio=0.3):
    """
    生成测试图像和掩码

    参数:
        img_size (int): 图像尺寸
        channels (int): 图像通道数 (1为灰度, 3为彩色)
        mask_ratio (float): 掩码区域比例

    返回:
        tuple: (原始图像, 修复图像, 掩码)
    """
    # 生成原始图像 (正弦波纹理)
    x = np.linspace(0, 4 * np.pi, img_size)
    y = np.linspace(0, 4 * np.pi, img_size)
    X, Y = np.meshgrid(x, y)

    # 创建不同频率的正弦波
    img1 = np.sin(X) * np.cos(Y)
    img2 = np.sin(2 * X) * np.cos(2 * Y)

    # 归一化到[0,1]
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())

    # 添加噪声模拟修复不完美
    noise = np.random.normal(0, 0.1, (img_size, img_size))
    repaired_img = img1 + noise
    repaired_img = np.clip(repaired_img, 0, 1)

    # 创建掩码 (随机位置)
    mask = np.zeros((img_size, img_size))
    mask_size = int(img_size * mask_ratio)
    start_x = np.random.randint(0, img_size - mask_size)
    start_y = np.random.randint(0, img_size - mask_size)
    mask[start_y:start_y + mask_size, start_x:start_x + mask_size] = 1

    # 处理多通道图像
    if channels > 1:
        # 创建彩色图像
        img1_color = np.stack([img1, img1 * 0.8, img1 * 0.6], axis=-1)
        repaired_img_color = np.stack([repaired_img, repaired_img * 0.8, repaired_img * 0.6], axis=-1)

        # 创建彩色掩码
        mask_color = np.stack([mask, mask, mask], axis=-1)

        return img1_color, repaired_img_color, mask_color
    else:
        return img1, repaired_img, mask


def visualize_results(original, repaired, mask, metrics):
    """
    可视化结果和指标

    参数:
        original (np.ndarray): 原始图像 (H, W) 或 (H, W, C)
        repaired (np.ndarray): 修复图像 (H, W) 或 (H, W, C)
        mask (np.ndarray): 掩码图像 (H, W) 或 (H, W, C)
        metrics (dict): 评估指标
    """
    # 创建对比图像
    diff = np.abs(original - repaired)

    # 创建带掩码的修复图像
    repaired_masked = original.copy()
    if mask.ndim == 2:
        repaired_masked[mask > 0] = repaired[mask > 0]
    elif mask.ndim == 3:
        repaired_masked[mask[:, :, 0] > 0] = repaired[mask[:, :, 0] > 0]

    # 创建绘图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 显示原始图像
    if original.ndim == 2:
        axes[0, 0].imshow(original, cmap='gray')
    else:
        axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 显示修复图像
    if repaired.ndim == 2:
        axes[0, 1].imshow(repaired, cmap='gray')
    else:
        axes[0, 1].imshow(repaired)
    axes[0, 1].set_title('Repaired Image')
    axes[0, 1].axis('off')

    # 显示差异图像
    if diff.ndim == 2:
        diff_plot = axes[0, 2].imshow(diff, cmap='hot')
    else:
        # 对于彩色图像，计算平均差异
        diff_avg = np.mean(diff, axis=2)
        diff_plot = axes[0, 2].imshow(diff_avg, cmap='hot')
    axes[0, 2].set_title('Difference')
    axes[0, 2].axis('off')
    fig.colorbar(diff_plot, ax=axes[0, 2])

    # 显示掩码
    if mask.ndim == 2:
        axes[1, 0].imshow(mask, cmap='gray')
    else:
        axes[1, 0].imshow(mask[:, :, 0], cmap='gray')  # 只显示第一个通道
    axes[1, 0].set_title('Mask')
    axes[1, 0].axis('off')

    # 显示带掩码的修复图像
    if repaired_masked.ndim == 2:
        axes[1, 1].imshow(repaired_masked, cmap='gray')
    else:
        axes[1, 1].imshow(repaired_masked)
    axes[1, 1].set_title('Repaired with Mask')
    axes[1, 1].axis('off')

    # 显示指标
    axes[1, 2].axis('off')
    metrics_text = f"MSE: {metrics['mse']:.4f}\nPSNR: {metrics['psnr']:.2f} dB\nSSIM: {metrics['ssim']:.4f}"
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12)

    plt.tight_layout()
    plt.show()


# 测试代码
if __name__ == '__main__':
    # # 测试灰度图像
    # print("测试灰度图像:")
    # original_gray, repaired_gray, mask_gray = generate_test_images(img_size=128, channels=1, mask_ratio=0.3)
    #
    # # 评估整个图像
    # print("评估整个图像:")
    # full_eval = evaluate_repair_quality(original_gray, repaired_gray)
    # print(f"全图评估: MSE={full_eval['mse']:.4f}, PSNR={full_eval['psnr']:.2f} dB, SSIM={full_eval['ssim']:.4f}")
    #
    # # 评估掩码区域
    # print("\n评估掩码区域:")
    # mask_eval = evaluate_repair_quality(original_gray, repaired_gray, mask_gray)
    # print(f"掩码区域评估: MSE={mask_eval['mse']:.4f}, PSNR={mask_eval['psnr']:.2f} dB, SSIM={mask_eval['ssim']:.4f}")
    #
    # # 可视化结果
    # visualize_results(original_gray, repaired_gray, mask_gray, mask_eval)
    #
    # # 测试彩色图像
    # print("\n测试彩色图像:")
    # original_color, repaired_color, mask_color = generate_test_images(img_size=128, channels=3, mask_ratio=0.3)
    #
    # # 评估整个图像
    # print("评估整个图像:")
    # full_eval_color = evaluate_repair_quality(original_color, repaired_color)
    # print(f"全图评估: MSE={full_eval_color['mse']:.4f}, PSNR={full_eval_color['psnr']:.2f} dB, SSIM={full_eval_color['ssim']:.4f}")
    #
    # # 评估掩码区域
    # print("\n评估掩码区域:")
    # mask_eval_color = evaluate_repair_quality(original_color, repaired_color, mask_color)
    # print(f"掩码区域评估: MSE={mask_eval_color['mse']:.4f}, PSNR={mask_eval_color['psnr']:.2f} dB, SSIM={mask_eval_color['ssim']:.4f}")

    # # 可视化结果
    # visualize_results(original_color, repaired_color, mask_color, mask_eval_color)

    # path_origin = r'F:\DeepLData\pic_repair_paper_effect\temp\0_fmi_dyna_org.png'
    # path_repaired = r'F:\DeepLData\pic_repair_paper_effect\temp\0_fmi_dyna_target_result.png'
    # path_masked = r'F:\DeepLData\pic_repair_paper_effect\temp\0_fmi_dyna_mask.png'
    #
    # image_origin = cv2.imread(path_origin, cv2.IMREAD_GRAYSCALE)
    # image_repaired = cv2.imread(path_repaired, cv2.IMREAD_GRAYSCALE)
    # image_mask = 255 - cv2.imread(path_masked, cv2.IMREAD_GRAYSCALE)
    # # image_mask = None
    #
    # print(image_origin.shape, image_repaired.shape)
    #
    # results_local = evaluate_repair_quality(image_origin, image_repaired, image_mask)
    # results_all = evaluate_repair_quality(image_origin, image_repaired)
    # print(results_local, results_all)

    # path_data = r'F:\DeepLData\pic_repair_paper_effect\0_background_mask'
    path_data = r'F:\DeepLData\pic_repair_paper_effect\0_fmi_dyna'
    path_list_org = search_files_by_criteria(path_data, name_keywords=['_org'], file_extensions=['.png'])
    path_list_mask = search_files_by_criteria(path_data, name_keywords=['_mask'], file_extensions=['.png'])
    path_list_result = search_files_by_criteria(path_data, name_keywords=['_model_result'], file_extensions=['.png'])
    path_list_target_result = search_files_by_criteria(path_data, name_keywords=['_target_result'], file_extensions=['.png'])

    image_origin = cv2.imread(path_list_org[0], cv2.IMREAD_GRAYSCALE)
    image_repaired = cv2.imread(path_list_result[0], cv2.IMREAD_GRAYSCALE)
    image_repaired_target = cv2.imread(path_list_target_result[0], cv2.IMREAD_GRAYSCALE)
    image_mask = 255 - cv2.imread(path_list_mask[0], cv2.IMREAD_GRAYSCALE)
    image_input = image_mask/255 * image_origin

    results_local = evaluate_repair_quality(image_origin, image_repaired, image_mask)
    results_all = evaluate_repair_quality(image_origin, image_repaired)
    print(results_local, results_all)

    windows_length = 256
    step = 200
    NUM_PIC = (image_origin.shape[0] - windows_length)//step + 1
    for i in range(NUM_PIC):
        start = i * step
        end = min(start + windows_length, image_origin.shape[0])
        img_windows_origin = image_origin[start:end]
        img_windows_mask = image_mask[start:end]
        img_input = image_input[start:end]

        img_windows_repaired = image_repaired[start:end]
        img_windows_target_result = image_repaired_target[start:end]

        show_Pic([img_windows_origin, img_windows_mask, img_input.astype(np.uint8), img_windows_repaired, img_windows_target_result], title='Image repair effect', pic_str=['Origin', 'Mask', 'Model_input', 'Model_output', 'origin+Repair*mask'])
