import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import trange
from fracture_mask_simulate.simulate_fractures_layer import add_depth_column
from use_cgan_model_create_fmi.dataloader_use_gan_model import stat_pic_dynamic_enhancement


def get_random_logging(resolution=0.1, depth_start=100, point_num=1000,
                              plot=False, seed=None,
                              config_rxo_trend=[[3, 1], [4, 1], [30, 1], [20, 1],
                                                [5, 1], [7, 1], [7, 1], [6, 1], [20, 1]]):
    """
    生成具有地质特征的随机测井曲线
    参数:
    resolution: 曲线分辨率 (米/点)
    depth_start: 起始深度 (米)
    point_num: 数据点数量
    plot: 是否绘制曲线图
    seed: 随机种子
    config_rxo_trend: 测井响应走势配置，格式为[[值1, 权重1], [值2, 权重2], ...]

    返回:
    df_temp: 包含深度、原始值、归一化值、上下限曲线的DataFrame
    """

    if seed is not None:
        np.random.seed(seed)

    # 1. 计算各段的点数
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

    for value, size in segment_points:
        values[current_index:current_index + size] = value
        current_index += size

    # 3. 添加随机噪声
    noise_level = np.random.uniform(0.2, 0.5) * values
    values += noise_level * np.random.randn(values.shape[0])

    # 4. 使用大窗口高斯滤波平滑衔接
    # 窗口大小占总点数的5%，最小为5
    window_size = max(5, int(point_num * 0.1))
    # 确保窗口为奇数
    if window_size % 2 == 0:
        window_size += 1

    # 高斯滤波
    values = gaussian_filter1d(values, sigma=window_size / 3)

    # 5. 确保值在合理范围内
    values = np.clip(values, 0, 100)

    # 6. 归一化处理
    normalized = (values - np.min(values)) / (np.max(values) - np.min(values)) * 0.6 + 0.2

    # 7. 创建深度数组
    depth = np.arange(depth_start, depth_start + point_num * resolution, resolution)

    # 8. 生成简单的上下限曲线，上下限在归一化曲线基础上加减一个随机偏移
    noise_scale = 0.3
    N_lower = normalized - noise_scale * np.abs(np.random.randn(len(normalized)))
    N_upper = normalized + noise_scale * np.abs(np.random.randn(len(normalized)))
    N_lower = gaussian_filter1d(N_lower, sigma=window_size / 7)
    N_upper = gaussian_filter1d(N_upper, sigma=window_size / 7)

    # 确保上下限在[0, 1]范围内
    N_lower = np.clip(N_lower, 0, 1)
    N_upper = np.clip(N_upper, 0, 1)

    # 9. 创建DataFrame
    df_temp = pd.DataFrame({
        'depth': depth[:len(normalized)],  # 确保长度一致
        'middle_values': normalized * 255,
        'N_lower_values': N_lower * 255,
        'N_upper_values': N_upper * 255,
        'normalized': normalized,
        'N_lower': N_lower,
        'N_upper': N_upper,
    })

    # 10. 可视化
    if plot:
        plt.figure(figsize=(8, 14))

        # 绘制归一化值和上下限曲线
        plt.plot(df_temp['normalized'], df_temp.index, 'b-', linewidth=1.5, label='Normalized')
        plt.plot(df_temp['N_lower'], df_temp.index, 'r--', linewidth=0.8, alpha=0.7, label='Lower Bound')
        plt.plot(df_temp['N_upper'], df_temp.index, 'g--', linewidth=0.8, alpha=0.7, label='Upper Bound')

        # 填充上下限之间的区域
        plt.fill_betweenx(df_temp.index,
                          df_temp['N_lower'],
                          df_temp['N_upper'],
                          color='gray', alpha=0.2, label='Uncertainty Range')

        # 设置图形属性
        plt.title('Normalized Logging Curve with Uncertainty Bounds')
        plt.xlabel('Normalized Value')
        plt.ylabel('Depth Index')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        # 反转Y轴，使深度从上到下增加（测井曲线常规显示方式）
        plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.show()

    return df_temp



def adjust_pic_range(img, middle, lower, upper, plot=False):
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
    current_median = np.median(img)
    # current_mean = np.mean(img)
    # current_min = np.min(img)
    # current_max = np.max(img)

    # 3. 计算当前图像的分布范围
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

    # # 7. 计算调整后的统计特性
    # adjusted_median = np.median(adjusted_img)
    # adjusted_mean = np.mean(adjusted_img)
    # adjusted_min = np.min(adjusted_img)
    # adjusted_max = np.max(adjusted_img)
    # adjusted_lower = np.percentile(adjusted_img, 9)
    # adjusted_upper = np.percentile(adjusted_img, 91)
    # # # 打印调试信息
    # # print(f"原始分布范围: 下限≈{current_lower:.2f}, 上限≈{current_upper:.2f}")
    # # print(f"调整后统计: 最小值={np.min(adjusted_img):.2f}, 最大值={np.max(adjusted_img):.2f}, 中位数={adjusted_median:.2f}")
    # # print(f"调整后分布范围: 下限≈{adjusted_lower:.2f}, 目标={lower:.2f}, 上限≈{adjusted_upper:.2f}, 目标={upper:.2f}")

    # 8. 绘制调整前后的像素分布频率图
    if plot:
        plt.figure(figsize=(11, 12))

        # 计算像素频率分布
        bins = 16
        hist_range = (0, 1)

        # 计算原始图像频率分布
        hist_original, bin_edges = np.histogram(img.flatten(), bins=bins, range=hist_range, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        # 计算调整后图像频率分布
        hist_adjusted, _ = np.histogram(adjusted_img.flatten(), bins=bins, range=hist_range, density=True)

        # 绘制柱状图
        plt.bar(bin_centers, hist_original/img.size, width=bin_width * 0.8, color='blue', alpha=0.5, edgecolor='darkblue', linewidth=0.5, label='原始图像')
        plt.bar(bin_centers, hist_adjusted/img.size, width=bin_width * 0.8, color='red', alpha=0.5, edgecolor='darkred', linewidth=0.5, label='调整后图像')

        # # 添加统计信息文本
        # stats_text_original = f'原始: 最小值={current_min:.3f}, 最大值={current_max:.3f}, 均值={current_mean:.3f}'
        # stats_text_adjusted = f'调整后: 最小值={adjusted_min:.3f}, 最大值={adjusted_max:.3f}, 均值={adjusted_mean:.3f}'
        # # target_text = f'目标: 中位数={middle:.3f}, 下限={lower:.3f}, 上限={upper:.3f}'

        # plt.text(0.02, 0.95, stats_text_original, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        # plt.text(0.02, 0.85, stats_text_adjusted, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        # # plt.text(0.02, 0.75, target_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        # # 添加变换信息
        # transform_text = f'变换参数: scale_factor={scale_factor:.4f}, offset={offset:.4f}'
        # plt.text(0.5, 0.95, transform_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # 设置图形属性
        plt.xlabel('像素值')
        plt.ylabel('频率密度')
        plt.title('像素分布频率对比 (原始 vs 调整后)')
        plt.xlim(hist_range)  # x轴范围为0-1
        # plt.ylim(0, 0.5)  # y轴范围为0-0.5
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=20)
        plt.tight_layout()
        plt.show()


    return adjusted_img.astype(img.dtype)


def adjust_image_by_resistivity(image, df_resisivity, window_pix=64):
    # 图像的融合增强，逐窗口进行 stat_FMI 图像像素分布调整
    index_norm = df_resisivity.columns.get_loc('normalized')
    index_lower = df_resisivity.columns.get_loc('N_lower')
    index_upper = df_resisivity.columns.get_loc('N_upper')

    image_target = image.copy()
    for i in trange(image.shape[0]-window_pix):
        index = i+window_pix//2
        r_mean = df_resisivity.iloc[index, index_norm]
        r_lower = df_resisivity.iloc[index, index_lower]
        r_upper = df_resisivity.iloc[index, index_upper]

        data_windows = image[i:i+window_pix, :]
        if i%1000 == 0:
            data_windows = adjust_pic_range(data_windows, r_mean, r_lower, r_upper, plot=True)
        else:
            data_windows = adjust_pic_range(data_windows, r_mean, r_lower, r_upper, plot=False)
        image_target[i:i + window_pix, :] = data_windows

    return image_target



if __name__ == '__main__':
    path_fmi_dyna = r'F:\logging_workspace\simu_beddings\different_r\fmi_dyna_bedding.png'
    path_fmi_stat = r'F:\logging_workspace\simu_beddings\different_r\fmi_dyna_bedding.png'

    fmi_stat = cv2.imread(path_fmi_stat, cv2.IMREAD_GRAYSCALE)
    print(fmi_stat.shape)
    fmi_stat = fmi_stat/256

    # # 简单配置：模拟砂泥岩剖面
    # simple_config = [
    #     [5, 3],  # 泥岩，厚度占比3
    #     [20, 2],  # 砂岩，厚度占比2
    #     [8, 1],  # 泥质砂岩，厚度占比1
    #     [15, 2],  # 砂岩，厚度占比2
    # ]
    #
    # # 生成测井曲线
    # df_result = get_random_logging(
    #     resolution=0.1,
    #     point_num=500,
    #     plot=True,
    #     seed=42,
    #     config_rxo_trend=simple_config
    # )

    # 生成stat静态FMI图像对应的电阻率配置
    random_curves = get_random_logging(
        resolution=0.0025,
        depth_start=0,
        point_num=fmi_stat.shape[0],
        plot=False,  # 是否可视化随机生成的测井曲线信息
        seed=None,
        config_rxo_trend=[
            [10, 29.9975],
            [20, 29.9975],
            [30, 29.9975],
            [40, 29.9975],
            [40, 29.9875],
            [30, 29.9875],
            [20, 29.9875],
            [10, 29.9875],
            [10, 29.94],
            [20, 29.94],
            [30, 29.94],
            [40, 29.94],
            [40, 29.9275],
            [30, 29.9275],
            [20, 29.9275],
            [10, 29.9275],
            [10, 29.8625],
            [20, 29.8625],
            [30, 29.8625],
            [40, 29.8625],
        ]
    )

    fmi_stat_adj = adjust_image_by_resistivity(fmi_stat, random_curves)
    print(fmi_stat_adj.shape)

    fmi_dyna_adj = stat_pic_dynamic_enhancement(fmi_stat_adj)

    fmi_dyna_adj = fmi_dyna_adj*256
    fmi_stat_adj = fmi_stat_adj*256

    path_dyna_save = r'F:\logging_workspace\simu_beddings\different_r\fmi_dyna_bedding_2.png'
    path_stat_save = r'F:\logging_workspace\simu_beddings\different_r\fmi_stat_bedding_2.png'
    cv2.imwrite(path_dyna_save, (fmi_dyna_adj).astype(np.uint8))
    cv2.imwrite(path_stat_save, (fmi_stat_adj).astype(np.uint8))

    path_dyna_save = r'F:\logging_workspace\simu_beddings\different_r\fmi_dyna_bedding_2.txt'
    path_stat_save = r'F:\logging_workspace\simu_beddings\different_r\fmi_stat_bedding_2.txt'
    img_dyna_gan_full = add_depth_column(fmi_dyna_adj, depth_start=0, depth_resolution=0.0025)
    np.savetxt(path_dyna_save, img_dyna_gan_full, comments='', delimiter='\t', fmt='%.6f', header='simu_dyna\n100\n104\nIMAGE.DYNA_SIMU\n\n\n\n')
    img_stat_gan_full = add_depth_column(fmi_stat_adj, depth_start=0, depth_resolution=0.0025)
    np.savetxt(path_stat_save, img_stat_gan_full, comments='', delimiter='\t', fmt='%.6f', header='simu_stat\n100\n104\nIMAGE.STAT_SIMU\n\n\n\n')
