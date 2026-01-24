import cv2
import numpy as np
import pandas as pd

from fracture_mask_simulate.test_bedding_simulator import bedding_mask_simulator
from src_ele.pic_opeeration import show_Pic

def get_bedding_configs():
    # 1. 生成单个纹层图像
    config_bedding_base = {
        'width': 256,
        'height': 12000,
        'x_shift': 0.3,
        'noise_level': 0.1,
        'thickness': 6,
        'amplitude': 0.06,
        'thickness_variation': 0.05,
        'waveform': 'sine',
        'random_seed': 42,
        'break_config': {
            'break_num_range': [0, 0, 0, 0, 0, 0, 1, 2],  # 0-2 个断裂
            'break_length_range': (5, 20),  # 断裂长度8-20像素
            'min_separation': 15,  # 断裂最小间隔15像素
        }
    }

    # x_shift_list = [0, 0.25, 0.5, 0.75]
    x_shift_list = [0.25]
    # thickness_list = [5, 10, 15, 20, 25]
    thickness_list = [5, 10, 15, 20, 25]
    # amplitude_list = [5, 10, 15, 20, 25]
    amplitude_list = [10]

    config_bedding_list = []
    for x_shift in x_shift_list:
        for thickness in thickness_list:
            for amplitude in amplitude_list:
                config_temp = config_bedding_base.copy()
                config_temp['x_shift'] = x_shift
                config_temp['thickness'] = thickness
                config_temp['amplitude'] = amplitude
                config_bedding_list.append(config_temp)

    return config_bedding_list


class fmi_simulator():
    def __init__(self):
        # 创建优化后的模拟器
        self.simulator_bedding = bedding_mask_simulator()

        self.VALUE_BACKGROUND = 0
        self.VALUE_HIGH_CONDUCTIVITY = 1
        self.VALUE_HIGH_RESISTANCE = 2
        self.VALUE_INDUCED = 3
        self.VALUE_BEDDING = 4
        self.VALUE_BREAKOUT = 5


        # 定义映射字典
        self.mapping_dict = {
            self.VALUE_BACKGROUND: 0,
            self.VALUE_HIGH_CONDUCTIVITY: 255,
            self.VALUE_HIGH_RESISTANCE: 255,
            self.VALUE_INDUCED: 255,
            self.VALUE_BEDDING: 255
            # 注意：5没有在映射中，如果需要映射5，可以添加
            # 5: 你想要的值
        }

    def get_fmi_bedding_mask(self):
        config_bedding_list = get_bedding_configs()
        config_write_list = []
        image_save = np.array([])
        for config in config_bedding_list:
            mask_bedding, bedding_target = self.simulator_bedding.get_bedding_mask_by_config(config)
            config_write = {
                'width': config['width'],
                'height': mask_bedding.shape[0],
                'x_shift': config['x_shift'],
                'noise_level': config['noise_level'],
                'thickness': config['thickness'],
                'amplitude': config['amplitude'],
                'thickness_variation': config['thickness_variation'],
            }
            config_write_list.append(config_write)
            if image_save.size == 0:
                image_save = mask_bedding.copy()
            else:
                image_save = np.concatenate((image_save, mask_bedding.copy()), axis=0)
            # show_Pic([mask_bedding, bedding_target], figure=[16, 12])

        image_save = self.map_matrix_dict_method(image_save)

        df_write = pd.DataFrame(config_write_list)
        print(df_write.describe())
        cv2.imwrite('mask_bedding.png', image_save)
        df_write.to_csv('mask_bedding.csv', index=False)

    def map_matrix_dict_method(self, matrix):
        """
        使用字典映射方法
        """

        # 创建结果矩阵的副本
        result = matrix.copy().astype(np.uint8)  # 确保是8位无符号整数
        for value in self.mapping_dict.keys():
            # 应用映射
            result[matrix == value] = self.mapping_dict[value]


        return result


if __name__ == '__main__':
    fs = fmi_simulator()
    fs.get_fmi_bedding_mask()