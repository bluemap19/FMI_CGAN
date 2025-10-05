import math
import numpy as np
import random
from scipy.signal import savgol_filter
from fracture_mask_simulate.fractures import add_vugs_random
from src_ele.pic_opeeration import show_Pic


def generate_multi_sine_bedding(width=256, height=256,
                                      bedding_x_shift=0.4,
                                      noise_level=0.02,
                                      bedding_width=30,
                                      bedding_amptitude=0.2,
                                      amptitude_random=[1.06, 1.1],
                                      # amptitude_random=[1.0, 1.5],
                                      ):
    """
    生成基于单个周期正弦曲线的裂缝图像
    参数:
    width: 图像宽度
    height: 图像高度
    bedding_x_shift: 裂缝X方向旋转像素个数，亦或者是百分比
    noise_level: 噪声强度 (0-1)
    bedding_width: 层厚，层理之间的宽度，决定了是 密集层理 还是 稀疏层理(像素)
    bedding_amptitude: 层理高度，控制波动幅度，层理与地层的角度，决定了是低角度层理 还是 高角度层理 (像素)
    return_grayscale: 是否返回灰度图
    返回:
    裂缝掩码图像 (二值或灰度)
    """
    # 1. 创建空白图像
    img = np.zeros((height, width), dtype=np.float32)

    # 2. 沿X方向进行曲线旋转偏移的设定
    if isinstance(bedding_x_shift, float):
        x_shift = bedding_x_shift * 2 * np.pi
    elif isinstance(bedding_x_shift, int):
        x_shift = bedding_x_shift/width * 2 * np.pi
    else:
        x_shift = 0

    if isinstance(bedding_amptitude, float):
        bedding_amptitude = int(bedding_amptitude * height)
    else:
        bedding_amptitude = bedding_amptitude

    # 3. 生成单个周期的正弦曲线数据，也是所有层理的基础
    # 生成一个完整周期的正弦数据点 (长度为width)
    x_range = np.linspace(0 + x_shift, 2 * np.pi + x_shift, width)  # 0到2π的一个完整周期
    y_signal = bedding_amptitude * np.sin(x_range)

    # 计算本次层理的正弦线个数
    signal_num = math.floor(0.98 * (height - 2*bedding_amptitude) / bedding_width)
    center_height_list = []
    signal_list = []

    for i in range(signal_num):
        # 4. 对正弦曲线添加噪声
        y_signal_temp = np.random.normal(0, noise_level, (y_signal.shape[0],)) * bedding_amptitude + y_signal

        # 5. 对噪声正弦曲线进行平滑降噪
        #**************************************************************
        # window_size = 7
        # window = np.ones(window_size) / window_size
        # y_signal = np.convolve(y_signal, window, mode='same')                     # 卷积平滑滤波
        #**************************************************************
        window_size = 20
        polyorder = 3  # polyorder为多项式拟合的阶数,它越小，则平滑效果越明显；越大，则更贴近原始曲线。
        y_signal_temp = savgol_filter(y_signal_temp, window_size, polyorder)        # SG滤波
        #**************************************************************

        signal_list.append(y_signal_temp)

        # 6.计算正弦线中心的 垂直方向index位置
        bedding_center = i*bedding_width + int(bedding_amptitude * (random.uniform(amptitude_random[0], amptitude_random[1])+(max(amptitude_random)-1)+0.1))
        center_height_list.append(bedding_center)

        if i%2 == 1:
            crack_down = signal_list[i-1] + center_height_list[i-1]
            crack_up = signal_list[i] + center_height_list[i]

            for j in range(width):
                # 计算裂缝的上下边界
                y_min = int(max(0, crack_down[j]))
                y_max = int(min(height, crack_up[j]))

                # 在裂缝区域填充
                if y_min < y_max:
                    img[y_min:y_max, j] = 1.0

    height_new = int((signal_num-1)*bedding_width + 2*bedding_amptitude + bedding_amptitude * (max(amptitude_random)-1))
    img = img[:height_new, :]
    print('generate bedding information as: width:{}; height:{}; layer num:{}; center list{}; bedding amptitude:{}'.format(width, height, signal_num, center_height_list, bedding_amptitude))

    # 7. 二值化处理
    img = np.clip((img * 255).astype(np.uint8), 0, 255)

    return img


def get_bedding_mask_simulated(width=300, height=400, add_random_vugs=20,
                               bedding_x_shift=0.6, noise_level=0.2, bedding_width=10,
                               bedding_amptitude=0.08, amptitude_random=[0.95, 1.05]):

    mask_dyna = generate_multi_sine_bedding(width=width, height=height,
                                bedding_x_shift=bedding_x_shift,
                                noise_level=noise_level,
                                bedding_width=bedding_width,
                                bedding_amptitude=bedding_amptitude,
                                amptitude_random=amptitude_random,
                                # amptitude_random=[1.0, 1.5],
                                )
    mask_stat = np.zeros_like(mask_dyna, dtype=np.uint8)

    if isinstance(add_random_vugs, bool):
        if add_random_vugs:
            mask_dyna, location_info = add_vugs_random(mask_dyna, 30, ratio_repetition=0.02)
            mask_stat, location_info = add_vugs_random(mask_stat, 30, ratio_repetition=0.02)
    elif isinstance(add_random_vugs, int):
        # 添加 add_random_vugs个数的孔洞信息
        mask_dyna, location_info = add_vugs_random(mask_dyna, add_random_vugs, ratio_repetition=0.02)
        mask_stat, location_info = add_vugs_random(mask_stat, add_random_vugs, ratio_repetition=0.02)

    # # print(mask_dyna.shape, mask_stat.shape)
    # show_Pic([mask_dyna, mask_stat], pic_order='12', figuresize=(8, 8))
    return mask_dyna, mask_stat

class shale_simulation(object):
    def __init__(self, ):



        pass

    def generate_random_laminate(self):

        pass


    def generate_random_layered(self):

        pass

    def generate_bedding_by_config(self, config_bedding={}):
        """
        生成基于单个周期正弦曲线的周期性层理、纹层图像
        参数:
        width: int，图像宽度
        height: int，图像高度
        bedding_x_shift: float 或 int 正弦线沿X方向旋转像素个数，亦或者是百分比
        noise_level: float，噪声强度 (0-1)
        bedding_width: int，层厚，层理之间的宽度，决定了是 密集层理 还是 稀疏层理(像素)
        bedding_amptitude: int 或 float，层理高度，控制波动幅度，层理与地层的角度，决定了是低角度层理 还是 高角度层理 (像素)
        返回:
        二值图像
        """
        # 默认配置初始化
        config_default = {'width':256, 'height':256, 'bedding_x_shift':0.4, 'noise_level':0.02,
                          'bedding_width':30, 'bedding_amptitude':0.2, 'amptitude_random':[1.06, 1.1]}

        # 配置合并
        config_bedding = {**config_default, **config_bedding}

        # 配置读取
        width = config_bedding['width']
        height = config_bedding['height']
        bedding_x_shift = config_bedding['bedding_x_shift']
        noise_level = config_bedding['noise_level']
        bedding_width = config_bedding['bedding_width']
        bedding_amptitude = config_bedding['bedding_amptitude']
        amptitude_random = config_bedding['amptitude_random']

        # 配置校正，确定他们的类型都是正确、合适的
        if not (isinstance(config_bedding['bedding_x_shift'], float) or isinstance(config_bedding['bedding_x_shift'], int)):
            print('config_bedding["bedding_x_shift"] must be float or int:{}'.format(config_bedding['bedding_x_shift']))
            exit(0)
        if not isinstance(config_bedding['noise_level'], float):
            print('config_bedding["noise_level"] must be float:{}'.format(config_bedding['noise_level']))
            exit(0)
        if not isinstance(config_bedding['bedding_width'], int):
            print('config_bedding["bedding_width"] must be int:{}'.format(config_bedding['bedding_width']))
            exit(0)
        if not (isinstance(config_bedding['bedding_amptitude'], int) or isinstance(config_bedding['bedding_amptitude'], float)):
            print('config_bedding["bedding_amptitude"] must be int:{}'.format(config_bedding['bedding_amptitude']))
            exit(0)
        if not isinstance(config_bedding['amptitude_random'], list):
            print('config_bedding["amptitude_random"] must be list, its a bedding_amptitude random range:{}'.format(config_bedding['amptitude_random']))
            exit(0)

        # 1. 创建空白图像
        img = np.zeros((height, width), dtype=np.float32)

        # 2. 沿X方向进行曲线旋转偏移的设定
        if isinstance(bedding_x_shift, float):
            x_shift = bedding_x_shift * 2 * np.pi
        elif isinstance(bedding_x_shift, int):
            x_shift = bedding_x_shift / width * 2 * np.pi
        else:
            x_shift = 0

        # 纹层或层理的幅度设定
        if isinstance(bedding_amptitude, float):
            bedding_amptitude = int(bedding_amptitude * height)
        else:
            bedding_amptitude = bedding_amptitude

        # 3. 生成单个周期的正弦曲线数据，也是所有层理的基础
        # 生成一个完整周期的正弦数据点 (长度为width)
        x_range = np.linspace(0 + x_shift, 2 * np.pi + x_shift, width)  # 0到2π的一个完整周期
        y_signal = bedding_amptitude * np.sin(x_range)

        # 计算本次层理的正弦线个数
        signal_num = math.floor(0.98 * (height - 2 * bedding_amptitude) / bedding_width)
        center_height_list = []
        signal_list = []

        for i in range(signal_num):
            # 4. 对正弦曲线添加噪声
            y_signal_temp = np.random.normal(0, noise_level, (y_signal.shape[0],)) * bedding_amptitude + y_signal

            # 5. 对噪声正弦曲线进行平滑降噪
            # **************************************************************
            window_size = int(width * 0.1)
            polyorder = 3                                                           # polyorder为多项式拟合的阶数,它越小，则平滑效果越明显；越大，则更贴近原始曲线。
            y_signal_temp = savgol_filter(y_signal_temp, window_size, polyorder)    # SG滤波
            # **************************************************************

            signal_list.append(y_signal_temp)

            # 6.计算正弦线中心的 垂直方向index位置
            bedding_center = i * bedding_width + int(bedding_amptitude * (
                        random.uniform(amptitude_random[0], amptitude_random[1]) + (max(amptitude_random) - 1) + 0.1))
            center_height_list.append(bedding_center)

            if i % 2 == 1:
                crack_down = signal_list[i - 1] + center_height_list[i - 1]
                crack_up = signal_list[i] + center_height_list[i]

                for j in range(width):
                    # 计算裂缝的上下边界
                    y_min = int(max(0, crack_down[j]))
                    y_max = int(min(height, crack_up[j]))

                    # 在裂缝区域填充
                    if y_min < y_max:
                        img[y_min:y_max, j] = 1.0

        height_new = int(
            (signal_num - 1) * bedding_width + 2 * bedding_amptitude + bedding_amptitude * (max(amptitude_random) - 1))
        img = img[:height_new, :]
        print(
            'generate bedding information as: width:{}; height:{}; layer num:{}; center list{}; bedding amptitude:{}'.format(
                width, height, signal_num, center_height_list, bedding_amptitude))

        # 7. 二值化处理
        img = np.clip((img * 255).astype(np.uint8), 0, 255)

        return img


if __name__ == '__main__':
    SS = shale_simulation()
    l1 = SS.generate_bedding_by_config(config_bedding={'width':256, 'height':1024, 'bedding_x_shift':0.3, 'noise_level':0.15,
                          'bedding_width':10, 'bedding_amptitude':0.01, 'amptitude_random':[1.06, 1.1]})
    l2 = SS.generate_bedding_by_config(config_bedding={'width':256, 'height':1024, 'bedding_x_shift':0.7, 'noise_level':0.07,
                          'bedding_width':35, 'bedding_amptitude':0.08, 'amptitude_random':[1.06, 1.1]})

    show_Pic([l1, l2], pic_order='12')