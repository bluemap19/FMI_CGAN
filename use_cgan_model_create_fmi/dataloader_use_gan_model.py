import math
import os
import pandas as pd
from skimage import exposure

from src_plot.plot_logging import visualize_well_logs
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 在导入任何库之前设置
import cv2
from scipy.signal import savgol_filter
from torch.utils.data import Dataset
from pic_preprocess import pic_tailor, random_morphology
from src_ele.pic_opeeration import show_Pic
import numpy as np
import matplotlib.pyplot as plt
import pywt


def wavelet_denoise(values, wavelet='db4', level=3, mode='soft'):
    """
    小波阈值去噪
    参数:
    values: 输入曲线值
    wavelet: 小波基类型
    level: 分解层数
    mode: 阈值模式 ('soft'或'hard')
    返回: 滤波后曲线
    """
    # 小波分解
    coeffs = pywt.wavedec(values, wavelet, level=level)

    # 计算阈值 (通用阈值规则)
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(values)))

    # 阈值处理
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode=mode))

    # 重构信号
    return pywt.waverec(new_coeffs, wavelet)[:len(values)]


def advanced_filter_pipeline(values):
    # 小波去噪去除高频噪声
    denoised = wavelet_denoise(values, wavelet='db4', level=3)
    # Savitzky-Golay优化曲线
    denoised = savgol_filter(denoised, window_length=int(values.shape[0]*0.09), polyorder=3)
    return denoised


def get_randoms_curve_by_normalized(normalized):
    """
    根据归一化曲线生成上下限曲线

    参数:
    normalized: 归一化曲线，形状为(L,)

    返回:
    N_lower: 下限曲线，始终小于normalized
    N_upper: 上限曲线，始终大于normalized
    """
    L = len(normalized)

    # 生成基于原始曲线的平滑噪声[1,4](@ref)
    noise_std = 1.5  # 噪声标准差
    base_noise = np.random.normal(0, noise_std, L)
    # 使用高斯滤波使噪声平滑[3](@ref)
    smooth_noise = advanced_filter_pipeline(base_noise)

    # 生成幅度调制因子，使上下限曲线有自然波动[5](@ref)
    amplitude_factor = 0.5 * np.sin(np.linspace(0, 6 * np.pi, L))

    # 计算基础偏移量，考虑曲线本身的波动性[8](@ref)
    rolling_std = np.zeros(L)
    window_size = min(21, L // 5)  # 自适应窗口大小
    for i in range(L):
        start = max(0, i - window_size // 2)
        end = min(L, i + window_size // 2 + 1)
        rolling_std[i] = np.std(normalized[start:end])

    # 生成下限曲线：始终低于原始曲线[1](@ref)
    lower_base = 0.9 * rolling_std  # 基础偏移量随波动性变化
    N_lower = normalized - lower_base * amplitude_factor + smooth_noise

    # 生成上限曲线：始终高于原始曲线
    upper_base = 0.9 * rolling_std
    N_upper = normalized + upper_base * amplitude_factor + smooth_noise

    # 确保约束条件：N_lower < normalized < N_upper
    tolerance = 1e-1  # 最小间距
    for i in range(L):
        if N_lower[i] >= normalized[i] - tolerance:
            N_lower[i] = normalized[i] - tolerance - np.random.uniform(0.01, 0.03)
        if N_upper[i] <= normalized[i] + tolerance:
            N_upper[i] = normalized[i] + tolerance + np.random.uniform(0.01, 0.03)

    # 最终平滑处理[3](@ref)
    N_lower = advanced_filter_pipeline(N_lower)
    N_upper = advanced_filter_pipeline(N_upper)

    # 确保约束条件：N_lower < normalized < N_upper
    tolerance = 1e-1  # 最小间距
    for i in range(L):
        if N_lower[i] >= normalized[i] - tolerance:
            N_lower[i] = normalized[i] - tolerance - np.random.uniform(0.01, 0.03)
        if N_upper[i] <= normalized[i] + tolerance:
            N_upper[i] = normalized[i] + tolerance + np.random.uniform(0.01, 0.03)

    # 确保曲线在合理范围内[1](@ref)
    N_lower = np.clip(N_lower, 0.0, 1.0)
    N_upper = np.clip(N_upper, 0.0, 1.0)

    return N_lower, N_upper

def get_random_logging(resolution=0.1, depth_start=100, point_num=1000, plot=False, seed=None,
                   config_rxo_trend=[[3, 1], [4, 1], [30, 1], [20, 1], [5, 1], [7, 1], [7, 1], [6, 1], [20, 1]]):
    """
    生成具有地质特征的随机测井曲线

    参数:
    resolution: 曲线分辨率 (米/点)
    depth_start: 起始深度 (米)
    point_num: 数据点数量
    plot: 是否绘制曲线图
    seed: 随机种子 (确保可重复性)
    config_rxo_trend: 测井响应走势，N*2的list，每一行分别代表[测井响应，该响应地层长度占比]，即根据这一段进行捏造测井响应，要求是添加随机的过度合理的噪声

    返回:
    depth: 深度数组
    values: 原始曲线值
    normalized: 归一化曲线值 (如果normalize=True)
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. 根据配置计算各段点数
    total_weight = sum(weight for _, weight in config_rxo_trend)
    segment_points = []
    accumulated_points = 0

    for i, (value, weight) in enumerate(config_rxo_trend):
        # 计算当前段点数
        segment_size = int(point_num * weight / total_weight)

        # 调整最后一段点数，确保总点数正确
        if i == len(config_rxo_trend) - 1:
            segment_size = point_num - accumulated_points

        segment_points.append((value, segment_size))
        accumulated_points += segment_size

    # 2. 生成基础曲线
    values = np.zeros(point_num)
    current_index = 0
    # 生成各段基础值
    for value, size in segment_points:
        values[current_index:current_index + size] = value
        current_index += size

    # 3. 添加段内趋势变化
    for i, (value, size) in enumerate(segment_points):
        start_idx = current_index - size  # 当前段起始索引
        end_idx = current_index  # 当前段结束索引

        # 随机选择趋势类型：0=稳定, 1=上升, 2=下降, 3=波动
        trend_type = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])

        if trend_type == 1:  # 上升趋势
            increment = np.random.uniform(0.1, 0.3) * value
            values[start_idx:end_idx] += np.linspace(0, increment, size)
        elif trend_type == 2:  # 下降趋势
            decrement = np.random.uniform(0.1, 0.3) * value
            values[start_idx:end_idx] -= np.linspace(0, decrement, size)
        elif trend_type == 3:  # 波动趋势
            amplitude = np.random.uniform(0.05, 0.15) * value
            values[start_idx:end_idx] += amplitude * np.sin(np.linspace(0, 4 * np.pi, size))

    # 4. 添加随机噪声
    noise_level = np.random.uniform(0.2, 0.5) * values
    values += noise_level * np.random.randn(values.shape[0])

    # 5. 平滑处理
    values = advanced_filter_pipeline(values)

    # 6. 确保值在合理范围内
    values = np.clip(values, 0.01, 100)

    # 7. 归一化处理
    normalized = (values - np.min(values)) / (np.max(values) - np.min(values)) * 0.7 + 0.15
    # normalized = (np.log(values) - np.min(np.log(values))) / (np.max(np.log(values)) - np.min(np.log(values)))

    # 8. 创建深度数组
    depth = np.arange(depth_start, depth_start + (normalized.shape[0]) * resolution, resolution)

    # 9. 根据normalized生成两条上下限曲线，进行范围的随机框定
    N_lower, N_upper = get_randoms_curve_by_normalized(normalized)

    df_temp = pd.DataFrame({
        'depth': depth.ravel(),
        'middle_values': normalized.ravel()*255,
        'N_lower_values': N_lower.ravel()*255,
        'N_upper_values': N_upper.ravel()*255,
        'normalized': normalized.ravel(),
        'N_lower': N_lower.ravel(),
        'N_upper': N_upper.ravel(),
    })

    # 11. 可视化
    if plot:
        plt.figure(figsize=(12, 8))

        # 绘制原始值
        plt.subplot(211)
        plt.plot(values)
        plt.title('Original Logging Curve')
        plt.ylabel('Value')
        plt.grid(True)

        # 绘制归一化值
        plt.subplot(212)
        plt.plot(normalized)
        plt.title('Normalized Logging Curve')
        plt.xlabel('Depth (m)')
        plt.ylabel('Normalized Value')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return df_temp


class ImageDataset_FMI_SIMULATE_LAYER(Dataset):
    def __init__(self, path=r'F:\DeepLData\FMI_SIMULATION\simu_cracks_2\9_background_mask.png',
                 style_dyna_pic_path=r'F:\DeepLData\target_stage1_small_big_mix\FMI_IMAGE\ZG_FMI_SPLIT\guan17-11_10_3585.0025_3586.2525_dyna.png',
                 x_l=256, y_l=256, step=20, win_len=400):
        super().__init__()
        self.mask_dyna = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.mask_stat = cv2.imread(path.replace('dyna', 'stat'), cv2.IMREAD_GRAYSCALE)
        self.mask_style_dyna = cv2.imread(style_dyna_pic_path, cv2.IMREAD_GRAYSCALE)
        self.mask_style_stat = cv2.imread(style_dyna_pic_path.replace('dyna', 'stat'), cv2.IMREAD_GRAYSCALE)
        self.shape_base = self.mask_stat.shape

        # # ############# mask的开闭处理
        # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (np.random.randint(1, 4) * 2 + 1, np.random.randint(1, 2) * 2 + 1))
        # # # mask_s = cv2.morphologyEx(mask_s, cv2.MORPH_OPEN, kernel, iterations=1)
        # # self.mask_stat = cv2.erode(self.mask_stat, kernel, iterations=1)
        # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (np.random.randint(1, 4) * 2 + 1, np.random.randint(1, 2) * 2 + 1))
        # # self.mask_dyna = cv2.dilate(self.mask_dyna, kernel, iterations=1)
        # # self.mask_dyna, location_info = add_vugs_random(self.mask_dyna, vug_num_p=50, ratio_repetition=0.2)
        #
        # self.mask_dyna = pic_add_noise(self.mask_dyna)
        # self.mask_stat = pic_add_noise(self.mask_stat)
        # # print(self.mask_dyna.shape, self.mask_stat.shape, )

        self.length = (self.mask_dyna.shape[0]-win_len)//step + 1
        # print(self.mask_dyna.shape, self.length)

        self.x_l = x_l
        self.y_l = y_l
        self.step = step
        self.win_len = win_len

        self.random_curves = get_random_logging(
            resolution=0.0025,
            depth_start=1000,
            point_num=self.mask_dyna.shape[0],
            plot=False,          # 是否可视化随机生成的测井曲线信息
            seed=42,
            config_rxo_trend=[[10, 1], [20, 1], [30, 1], [30, 1], [20, 1], [10, 1],
                              [10, 1], [20, 1], [30, 1], [30, 1], [20, 1], [10, 1],
                              [10, 0.9755], [20, 0.9755], [30, 0.9735], [30, 0.9735], [20, 0.9735], [10, 0.9735],
                              [10, 0.939], [20, 0.939], [30, 0.958], [30, 0.958], [20, 0.958], [10, 0.958]]
        )


    def __getitem__(self, index):
        mask_d = self.mask_dyna[index*self.step:index*self.step+self.win_len, :]
        mask_s = self.mask_stat[index*self.step:index*self.step+self.win_len, :]
        # print(mask.shape)

        mask_d = cv2.resize(mask_d, (self.x_l, self.y_l))
        mask_s = cv2.resize(mask_s, (self.x_l, self.y_l))

        mask_d = mask_d.reshape((self.x_l, self.y_l))/256
        mask_s = mask_s.reshape((self.x_l, self.y_l))/256

        dyna_8 = cv2.resize(cv2.resize(mask_d, (8, 8)), (self.x_l, self.y_l))
        stat_8 = cv2.resize(cv2.resize(mask_s, (8, 8)), (self.x_l, self.y_l))
        dyna_4 = cv2.resize(cv2.resize(mask_d, (4, 4)), (self.x_l, self.y_l))
        stat_4 = cv2.resize(cv2.resize(mask_s, (4, 4)), (self.x_l, self.y_l))

        style_dyna, style_stat = self.get_style_random()

        pic_list = self.adjust_pic_by_rxo(pic_list=[stat_8, stat_4], index=index*self.step+self.win_len//2)
        stat_8, stat_4 = pic_list[0], pic_list[1]
        mask = np.array([mask_d, mask_s, dyna_8, stat_8, dyna_4, stat_4, style_dyna, style_stat])

        return mask
        # return pic_all_org, pic_all_mask


    def __len__(self):
        return self.length


    def get_style_random(self):
        pic_all_org = np.array([self.mask_style_dyna, self.mask_style_stat])
        pic_cropped = pic_tailor(pic_all_org, pic_shape_ex=[60, 60])

        style_dyna = pic_cropped[0]
        style_stat = pic_cropped[1]

        style_dyna = cv2.resize(style_dyna, (self.x_l, self.y_l))/256
        style_stat = cv2.resize(style_stat, (self.x_l, self.y_l))/256
        return style_dyna, style_stat

    def adjust_pic_by_rxo(self, pic_list=[], index=0):
        index_cols = self.random_curves.columns.get_loc('normalized')
        rxo_index = self.random_curves.iloc[index, index_cols]

        if (rxo_index <= 0):
            print(f'reset rxo {rxo_index} as 0.01')
            rxo_index = 0.01
        if (rxo_index > 1):
            print(f'reset rxo {rxo_index} as 1')
            rxo_index = 1

        for i in range(len(pic_list)):
            pic_list[i] = rxo_index * pic_list[i]
        return pic_list

    def get_base_shape(self):
        return self.shape_base


def adjust_pic_range(img, middle, lower, upper):
    """
    根据给定的中位数、下限和上限调整图像数值范围

    参数:
    img: 输入图像 (numpy数组)
    middle: 目标中位数
    lower: 目标下限 (数据高斯分布-σ)
    upper: 目标上限 (数据高斯分布+σ)

    返回:
    调整后的图像
    """
    # 1. 输入验证
    if not isinstance(img, np.ndarray):
        raise TypeError("输入必须是numpy数组")
    if img.size == 0:
        raise ValueError("输入图像不能为空")
    if lower >= upper:
        raise ValueError("下限必须小于上限")
    if not (lower <= middle <= upper):
        raise ValueError("中位数必须在上下限之间")

    # 2. 计算当前图像的统计特性
    current_min = np.min(img)
    current_max = np.max(img)
    current_median = np.median(img)

    # 3. 计算当前图像的分布范围
    # current_lower = np.percentile(img, 15.87)  # 约等于μ-σ (标准正态分布)
    # current_upper = np.percentile(img, 84.13)  # 约等于μ+σ (标准正态分布)
    current_lower = np.percentile(img, 9)  # 约等于μ-2σ (标准正态分布)
    current_upper = np.percentile(img, 91)  # 约等于μ+2σ (标准正态分布)

    # 4. 计算缩放因子和偏移量
    # 计算当前分布范围到目标分布范围的缩放因子
    scale_factor = (upper - lower) / (current_upper - current_lower)

    # 计算偏移量，使当前中位数映射到目标中位数
    offset = middle - (current_median * scale_factor)

    # 5. 应用线性变换
    adjusted_img = img.astype(np.float32) * scale_factor + offset

    # 6. 确保值在合理范围内
    adjusted_img = np.clip(adjusted_img, 0, 1)

    # # 7. 验证调整后的统计特性
    # adjusted_median = np.median(adjusted_img)
    # adjusted_lower = np.percentile(adjusted_img, 15.87)
    # adjusted_upper = np.percentile(adjusted_img, 84.13)
    # # 打印调试信息
    # print(f"原始图像统计: 最小值={current_min:.2f}, 最大值={current_max:.2f}, 中位数={current_median:.2f}")
    # print(f"原始分布范围: 下限≈{current_lower:.2f}, 上限≈{current_upper:.2f}")
    # print(f"调整后统计: 最小值={np.min(adjusted_img):.2f}, 最大值={np.max(adjusted_img):.2f}, 中位数={adjusted_median:.2f}")
    # print(f"调整后分布范围: 下限≈{adjusted_lower:.2f}, 目标={lower:.2f}, 上限≈{adjusted_upper:.2f}, 目标={upper:.2f}")

    return adjusted_img.astype(img.dtype)


def dynamic_enhancement_function(img):
    """
    电成像图像动态增强函数（简化版）

    该函数针对电成像数据的特点，专注于三个核心增强技术：
    1. 自适应直方图均衡化（CLAHE）增强局部对比度
    2. 多尺度锐化增强地质特征
    3. 动态范围调整优化整体对比度

    参数:
    img: 输入图像 (256×256 numpy数组), 值范围0-1或0-255

    返回:
    增强后的图像 (256×256 numpy数组), 值范围与输入相同
    """
    # ========================
    # 1. 输入验证和预处理
    # ========================
    # 验证输入是否为256×256的numpy数组
    if not isinstance(img, np.ndarray):
        raise ValueError("输入必须是 M×N 的numpy数组")

    # 保存原始数据类型和范围
    original_dtype = img.dtype
    is_normalized = np.max(img) <= 1.0  # 检查是否在0-1范围内

    # 转换为浮点类型并归一化到0-1范围
    if is_normalized:
        img_float = img.astype(np.float32)
    else:
        img_float = img.astype(np.float32) / 255.0

    # ===================================
    # 2. 自适应直方图均衡化（CLAHE）
    # ===================================
    # 增强局部对比度，特别适合电成像中的地层特征
    # 将图像转换为0-255范围用于CLAHE处理
    img_8bit = (img_float * 255).astype(np.uint8)

    # 创建CLAHE对象
    # clipLimit=2.0：限制对比度增强幅度，避免噪声放大
    # tileGridSize=(8,8)：将图像分为8x8的小块进行局部均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 应用CLAHE
    clahe_enhanced = clahe.apply(img_8bit)

    # 转换回浮点类型
    clahe_enhanced = clahe_enhanced.astype(np.float32) / 255.0

    # ===================================
    # 3. 多尺度锐化增强地质特征
    # ===================================
    # 使用不同尺度的锐化增强不同大小的地质特征
    # 小尺度锐化增强微裂缝
    # 使用高斯模糊和减法实现锐化
    blurred_small = cv2.GaussianBlur(clahe_enhanced, (0, 0), sigmaX=1.0)
    sharpened_small = cv2.addWeighted(clahe_enhanced, 1.5, blurred_small, -0.5, 0)

    # 中尺度锐化增强中等特征
    blurred_medium = cv2.GaussianBlur(clahe_enhanced, (0, 0), sigmaX=3.0)
    sharpened_medium = cv2.addWeighted(clahe_enhanced, 1.2, blurred_medium, -0.2, 0)

    # 大尺度锐化增强地层结构
    blurred_large = cv2.GaussianBlur(clahe_enhanced, (0, 0), sigmaX=7.0)
    sharpened_large = cv2.addWeighted(clahe_enhanced, 1.1, blurred_large, -0.1, 0)

    # 加权融合多尺度锐化结果
    # 权重分配：小尺度特征最重要，大尺度特征次要
    sharpened = 0.5 * sharpened_small + 0.3 * sharpened_medium + 0.2 * sharpened_large
    sharpened = np.clip(sharpened, 0, 1)  # 确保值在0-1范围内

    # ===================================
    # 4. 动态范围调整优化
    # ===================================
    # 自适应调整图像的动态范围
    # 使用2%和98%百分位值排除极端值
    v_min, v_max = np.percentile(sharpened, (2, 98))

    # 重新缩放强度范围
    adjusted = exposure.rescale_intensity(
        sharpened,
        in_range=(v_min, v_max),
        out_range=(0, 1)
    )

    # ===================================
    # 5. 裂缝特征增强
    # ===================================
    # 专门增强电成像中的裂缝特征
    # 计算梯度幅度（裂缝通常有高梯度）
    grad_x = cv2.Sobel(adjusted, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(adjusted, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_mag = grad_mag / np.max(grad_mag)  # 归一化到0-1

    # 创建裂缝增强掩模（高梯度区域）
    fracture_threshold = 0.7  # 裂缝增强阈值
    fracture_mask = np.where(grad_mag > fracture_threshold, 1.0, 0.0)

    # 增强裂缝区域（增加对比度）
    fracture_enhanced = adjusted.copy()
    fracture_enhanced = fracture_enhanced * (1 - fracture_mask) + (fracture_enhanced * 1.5) * fracture_mask
    fracture_enhanced = np.clip(fracture_enhanced, 0, 1)

    # ===================================
    # 6. 伽马校正优化整体视觉效果
    # ===================================
    # 调整图像的整体亮度分布
    gamma = 0.9  # 伽马值 < 1 增强暗部，> 1 增强亮部
    gamma_corrected = exposure.adjust_gamma(fracture_enhanced, gamma)

    # ========================
    # 后处理
    # ========================
    # 转换为原始数据类型和范围
    if is_normalized:
        result = gamma_corrected.astype(original_dtype)
    else:
        result = (gamma_corrected * 255).astype(original_dtype)

    return result


def stat_pic_dynamic_enhancement(stat_img, windows_length=20, step=1):
    """
    将静态的FMI图像，增强为动态FMI图像，逐窗口处理
    stat_img:静态电成像的FMI数据
    windows_length：遍历的窗长设定
    sttp:遍历的步长设定
    """
    windows_num = (stat_img.shape[0] - windows_length)//step + 1
    dyna_img = stat_img.copy()

    for i in range(windows_num):
        if i == windows_num - 1:
            window_stat = stat_img[-windows_length:, :]
        else:
            window_stat = stat_img[i*step:i*step+windows_length, :]

        windows_dyna = dynamic_enhancement_function(window_stat)

        if i == windows_num-1:
            dyna_img[-windows_length:, :] = windows_dyna
        else:
            dyna_img[i*step:i*step+windows_length, :] = windows_dyna

    return dyna_img


class ImageDataset_FMI_SPLIT_NO_REPEAT(Dataset):
    def __init__(self, path=r'F:\DeepLData\FMI_SIMULATION\simu_cracks_2\9_background_mask.png',
                 x_l=256, y_l=256, win_len=250):
        super().__init__()
        if os.path.exists(path):
            # 动态dyna_mask为对应的读取的原始mask数据
            self.mask_background = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.mask_dyna = self.mask_background.copy()

            # # self.mask_stat = self.mask_background.copy()
            # # <0.25：溶蚀（腐蚀） <0.50：膨胀 <0.75:开运算（先腐蚀后膨胀） <1.00:闭运算（先膨胀后腐蚀）
            self.mask_stat_p = random_morphology(self.mask_background.copy(), ratio=0.1, k_size=None)
            # 静态mask对应的噪声
            self.mask_stat_noise = np.random.random((self.mask_dyna.shape[0]//8, self.mask_dyna.shape[1]//8))*255
            self.mask_stat_noise = cv2.resize(self.mask_stat_noise, (self.mask_dyna.shape[1], self.mask_dyna.shape[0]))
            # _, self.mask_stat = cv2.threshold(self.mask_stat, 125, 255, cv2.THRESH_BINARY)    # 不能进行二值化，否则输出图像就显得很机械

            # 静态 stat_mask 为对应的读取的原始mask数据进行(溶蚀)处理后 加0.5的噪声 处理
            # self.mask_stat = self.mask_stat*0.65 + self.mask_stat_p*0.35
            self.mask_stat = self.mask_stat_noise*0.5 + self.mask_stat_p*0.5
        else:
            print(f'path {path} does not exist')
            exit(0)
        self.shape_full_fmi = self.mask_background.shape

        self.x_l = x_l              # 输出图像的x方向长度
        self.y_l = y_l              # 输出图像的y方向长度
        self.win_len = win_len      # 窗口截取图像的 窗口长度

        # 计算dataloader长度，看需要把FMI数据分成几张窗口数据
        self.length = math.ceil(self.mask_background.shape[0]/self.win_len)

        # 生成stat静态FMI图像对应的电阻率配置
        self.random_curves = get_random_logging(
            resolution=0.0025,
            depth_start=1000,
            point_num=self.mask_background.shape[0],
            plot=False,          # 是否可视化随机生成的测井曲线信息
            seed=42,
            config_rxo_trend=[[10, 1], [20, 1], [30, 1], [30, 1], [20, 1], [10, 1],
                              [10, 1], [20, 1], [30, 1], [30, 1], [20, 1], [10, 1],
                              [10, 0.9755], [20, 0.9755], [30, 0.9735], [30, 0.9735], [20, 0.9735], [10, 0.9735],
                              [10, 0.939], [20, 0.939], [30, 0.958], [30, 0.958], [20, 0.958], [10, 0.958]]
        )

        # self.dyna_mask_8，self.stat_mask_8 为动静态的mask8对应的图像矩阵分布，这里使用随机数，进行噪声随机分布添加
        random_matrix = np.random.randint(0, 256, size=(8 * self.length, 8), dtype=np.uint8)
        self.dyna_mask_8 = cv2.resize(random_matrix, (self.shape_full_fmi[1], self.shape_full_fmi[0]))
        random_matrix = np.random.randint(0, 256, size=(8 * self.length, 8), dtype=np.uint8)
        self.stat_mask_8 = cv2.resize(random_matrix, (self.shape_full_fmi[1], self.shape_full_fmi[0]))

    def __getitem__(self, index):
        if index >= self.length:
            print('index is out of range and reset it as mod(length):{}->{}'.format(index, index%self.length))
            index = index%self.length

        if index == self.length - 1:
            mask_dyna = self.mask_dyna[-self.win_len:, :]
            mask_stat = self.mask_stat[-self.win_len:, :]
            dyna_8 = self.dyna_mask_8[-self.win_len:, :]
            stat_8 = self.stat_mask_8[-self.win_len:, :]
        else:
            mask_dyna = self.mask_dyna[index*self.win_len:(index+1)*self.win_len, :]
            mask_stat = self.mask_stat[index*self.win_len:(index+1)*self.win_len, :]
            dyna_8 = self.dyna_mask_8[index*self.win_len:(index+1)*self.win_len, :]
            stat_8 = self.stat_mask_8[index*self.win_len:(index+1)*self.win_len, :]

        # 要缩放成模型可以认可的 指定的 长宽 的图像
        mask_dyna = cv2.resize(mask_dyna, (self.y_l, self.x_l))
        mask_stat = cv2.resize(mask_stat, (self.y_l, self.x_l))
        dyna_8 = cv2.resize(dyna_8, (self.y_l, self.x_l))
        stat_8 = cv2.resize(stat_8, (self.y_l, self.x_l))
        _, mask_dyna = cv2.threshold(mask_dyna, 5, 255, cv2.THRESH_BINARY)
        # _, mask_stat = cv2.threshold(mask_stat, 5, 255, cv2.THRESH_BINARY)            # mask_stat不能进行二值化，否则其对应的stat fmi数据会显得很人造

        mask_dyna = mask_dyna/256
        mask_stat = mask_stat/256
        dyna_8 = dyna_8/256
        stat_8 = stat_8/256

        pic_list = self.adjust_pic_by_r_curve(pic_list=[stat_8], index=index * self.win_len + self.win_len // 2)
        stat_8 = pic_list[0]
        mask = np.array([mask_dyna, mask_stat, dyna_8, stat_8])

        return mask

    def __len__(self):
        return self.length

    def adjust_pic_by_r_curve(self, pic_list=[], index=0):
        """
        通过dataloader生成的随机测井曲线，随机上下限，调整静态图像的图像范围
        """
        # 'depth': depth.ravel(),
        # 'middle_values': normalized.ravel()*255,
        # 'N_lower_values': N_lower.ravel()*255,
        # 'N_upper_values': N_upper.ravel()*255,
        # 'normalized': normalized.ravel(),
        # 'N_lower': N_lower.ravel(),
        # 'N_upper': N_upper.ravel(),
        index_cols = self.random_curves.columns.get_loc('normalized')
        r_index = self.random_curves.iloc[index, index_cols]
        index_cols = self.random_curves.columns.get_loc('N_lower')
        r_lower_index = self.random_curves.iloc[index, index_cols]
        index_cols = self.random_curves.columns.get_loc('N_upper')
        r_upper_index = self.random_curves.iloc[index, index_cols]

        for i in range(len(pic_list)):
            pic_list[i] = adjust_pic_range(pic_list[i], r_index, r_lower_index, r_upper_index)
        return pic_list

    def get_base_shape(self):
        return self.shape_full_fmi

    def combine_pic_list_to_full_fmi(self, img_gan):
        """
        # combine splits pic (from model) to a layer FMI image
        将模型输出的电成像数据进行合并，合并成完整的FMI数据
        """
        img_dyna_gan_full = np.zeros(self.get_base_shape())
        img_stat_gan_full = np.zeros(self.get_base_shape())
        # 根据模型生成的电成像图像矩阵[M*2*256*256]，逐窗口进行，电成像测井图像数据的合并
        for i in range(img_gan.shape[0]):
            dyna_t = img_gan[i, 0, :, :]
            stat_t = img_gan[i, 1, :, :]

            # 需要把模型输出数据进行shape的调整，调整为原始的模型输入数据
            dyna_t = cv2.resize(dyna_t, (self.shape_full_fmi[1], self.win_len))
            stat_t = cv2.resize(stat_t, (self.shape_full_fmi[1], self.win_len))

            # 将窗口数据，逐窗口进行合并
            if i != (img_gan.shape[0] - 1):
                img_dyna_gan_full[i * self.win_len:(i + 1) * self.win_len, :] = dyna_t
                img_stat_gan_full[i * self.win_len:(i + 1) * self.win_len, :] = stat_t
            else:
                img_dyna_gan_full[-self.win_len:, :] = dyna_t
                img_stat_gan_full[-self.win_len:, :] = stat_t

        # 图像的融合增强，逐窗口进行 stat_FMI 图像像素分布调整
        window_adjust_stat_image = 64
        img_stat_adjust = img_stat_gan_full.copy()
        for i in range(img_stat_gan_full.shape[0]-window_adjust_stat_image):
            stat_data_windows = img_stat_gan_full[i:i+window_adjust_stat_image, :]
            stat_data_windows = self.adjust_pic_by_r_curve(pic_list=[stat_data_windows], index=i+window_adjust_stat_image//2)[0]
            img_stat_adjust[i:i + window_adjust_stat_image, :] = stat_data_windows

        # 动态电成像生成，根据 静态电成像 计算 动态电成像数据
        img_dyna_fmi = stat_pic_dynamic_enhancement(img_stat_adjust, windows_length=20, step=1)

        img_dyna_fmi = img_dyna_fmi*255
        img_stat_adjust = img_stat_adjust*255
        # show_Pic([1-img_dyna_gan_full, 1-img_stat_gan_full, 1-img_stat_adjust, 1-img_dyna_fmi], pic_order='14', figure=(8, 10))
        return img_stat_adjust, img_dyna_fmi


if __name__ == '__main__':
    a = ImageDataset_FMI_SPLIT_NO_REPEAT()
    for i in range(10):
        b = a[i]
        show_Pic([b[0,:,:], b[1, :, :], b[2,:,:], b[3,:,:]], pic_order='22')

# # 使用示例
# if __name__ == "__main__":
#     # 生成随机曲线
#     df_temp = get_random_logging(
#         resolution=0.5,
#         depth_start=100000,
#         point_num=500,
#         plot=False,
#         seed=42,
#         config_rxo_trend=[[10, 1], [20, 1], [30, 1], [30, 1], [20, 1], [10, 1],
#                               [10, 1], [20, 1], [30, 1], [30, 1], [20, 1], [10, 1],
#                               [10, 0.9755], [20, 0.9755], [30, 0.9735], [30, 0.9735], [20, 0.9735], [10, 0.9735],
#                               [10, 0.939], [20, 0.939], [30, 0.958], [30, 0.958], [20, 0.958], [10, 0.958]]
#     )
#
#     visualize_well_logs(
#         data=df_temp,
#         depth_col='depth',
#         curve_cols=['N_lower_values', 'middle_values', 'N_upper_values', 'N_lower', 'normalized', 'N_upper'],
#         figsize=(8, 10),
#         range_limits=[[0, 255], [0, 255], [0, 255], [0, 1], [0, 1], [0, 1]]
#     )

