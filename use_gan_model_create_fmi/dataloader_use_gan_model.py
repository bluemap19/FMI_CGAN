import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 在导入任何库之前设置
import random
import os
import cv2
import numpy as np
from scipy.signal import savgol_filter
from torch.utils.data import Dataset
from fracture_mask_simulate.fractures import add_vugs_random
from pic_preprocess import pic_binary_random, pic_tailor, pic_add_noise
from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, median_filter
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
    normalized = (values - np.min(values)) / (np.max(values) - np.min(values)) * 0.92 + 0.04
    # normalized = (np.log(values) - np.min(np.log(values))) / (np.max(np.log(values)) - np.min(np.log(values)))

    # 8. 创建深度数组
    depth = np.arange(depth_start, depth_start + (normalized.shape[0]-1) * resolution, resolution)

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

    return depth, values, normalized


class ImageDataset_FMI_SIMULATE_LAYER(Dataset):
    def __init__(self, path=r'D:\PycharmProject\FMI_GAN_Create\fracture_mask_simulate\p-t-2\dyna-mask_14_100_104.png',
                 style_dyna_pic_path=r'D:\DeepLData\target_stage1_small_big_mix\guan17-11_194_3677.0025_3678.2525_dyna.png',
                 x_l=256, y_l=256, step=20, win_len=400):
        super().__init__()
        self.mask_dyna = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.mask_stat = cv2.imread(path.replace('dyna', 'stat'), cv2.IMREAD_GRAYSCALE)
        self.mask_style_dyna = cv2.imread(style_dyna_pic_path, cv2.IMREAD_GRAYSCALE)
        self.mask_style_stat = cv2.imread(style_dyna_pic_path.replace('dyna', 'stat'), cv2.IMREAD_GRAYSCALE)
        self.shape_base = self.mask_stat.shape

        # mask的开闭处理
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (np.random.randint(1, 4) * 2 + 1, np.random.randint(1, 2) * 2 + 1))
        # # mask_s = cv2.morphologyEx(mask_s, cv2.MORPH_OPEN, kernel, iterations=1)
        # self.mask_stat = cv2.erode(self.mask_stat, kernel, iterations=1)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (np.random.randint(1, 4) * 2 + 1, np.random.randint(1, 2) * 2 + 1))
        # self.mask_dyna = cv2.dilate(self.mask_dyna, kernel, iterations=1)
        # self.mask_dyna, location_info = add_vugs_random(self.mask_dyna, vug_num_p=50, ratio_repetition=0.2)

        self.mask_dyna = pic_add_noise(self.mask_dyna)
        self.mask_stat = pic_add_noise(self.mask_stat)
        # print(self.mask_dyna.shape, self.mask_stat.shape, )

        self.length = (self.mask_dyna.shape[0]-win_len)//step + 1
        # print(self.mask_dyna.shape, self.length)

        self.x_l = x_l
        self.y_l = y_l
        self.step = step
        self.win_len = win_len

        depth, values, self.random_rxo = get_random_logging(
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
        rxo_index = self.random_rxo[index]

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



if __name__ == '__main__':
    a = ImageDataset_FMI_SIMULATE_LAYER()
    for i in range(10):
        b = a[i]
        # print(b.shape)
        show_Pic([b[0,:,:], b[1, :, :], b[2,:,:], b[3,:,:], b[4,:,:], b[5,:,:], b[6,:,:], b[7,:,:]], pic_order='24')

# # 使用示例
# if __name__ == "__main__":
#     # 生成随机曲线
#     depth, values, normalized = get_random_logging(
#         resolution=0.5,
#         depth_start=1500,
#         point_num=500,
#         plot=True,
#         seed=42,
#         config_rxo_trend=[[20, 1], [4, 1], [30, 1], [20, 1], [5, 1], [7, 1], [7, 1], [6, 1], [10, 1]]
#     )
