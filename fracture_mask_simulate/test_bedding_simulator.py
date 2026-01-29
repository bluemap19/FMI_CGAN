import math
import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, List, Optional, Union, Tuple
import warnings

from src_ele.pic_opeeration import show_Pic


class bedding_mask_simulator:
    def __init__(self):
        # 常量定义
        self.VALUE_BACKGROUND = 0
        self.VALUE_HIGH_CONDUCTIVITY = 1
        self.VALUE_HIGH_RESISTANCE = 2
        self.VALUE_INDUCED = 3
        self.VALUE_BEDDING = 4
        self.VALUE_BREAKOUT = 5

        # 基础配置
        self.CONFIG_BEDDING_DEFAULT = {
            'width': 256,  # 图像宽度
            'height': 400,  # 图像高度
            'x_shift': 0.4,  # 纹层水平相位的偏移 [0,1]
            'noise_level': 0.1,  # 噪声水平 [0,1]
            'laminar_thickness': 10,  # 层厚（像素）
            'amplitude': 0.1,  # 波动幅度 [0,1] 或像素值
            'thickness_variation': 0.02,  # 厚度变化范围 [0,1]
            'waveform': 'sine',  # 波形类型: 'sine', 'cosine', 'sawtooth', 'triangle'
            'smooth_window': 0.05,  # 水平方向曲线平滑平滑窗口（占宽度的比例）
            'random_seed': None,  # 随机种子
        }
        self.CONFIG_BEDDING_DEFAULT['break_config']={
            'break_num_range': [0, 0, 0, 0, 1, 1, 2],  # 0-2 个断裂
            'break_length_range': (5, 20),  # 断裂长度8-20像素
            'break_height_range': (1, self.CONFIG_BEDDING_DEFAULT['laminar_thickness']),  # 断裂高度2-5像素
            'min_separation': 15,  # 断裂最小间隔15像素
        }

    def _validate_config(self, config: Dict) -> Dict:
        """验证和规范化配置参数"""
        default_config = self.CONFIG_BEDDING_DEFAULT.copy()
        default_config.update(config)

        # 参数范围验证
        if not (0 <= default_config['x_shift'] <= 1):
            warnings.warn(f"x_shift ({default_config['x_shift']}) should be in [0, 1], clamping...")
            default_config['x_shift'] = np.clip(default_config['x_shift'], 0, 1)

        if not (0 <= default_config['noise_level'] <= 1):
            warnings.warn(f"noise_level ({default_config['noise_level']}) should be in [0, 1], clamping...")
            default_config['noise_level'] = np.clip(default_config['noise_level'], 0, 1)

        if default_config['laminar_thickness'] <= 0:
            raise ValueError(f"thickness must be positive, got {default_config['laminar_thickness']}")

        if default_config['laminar_gap'] <= 0:
            raise ValueError(f"gap must be positive, got {default_config['laminar_thickness']}")

        if default_config['amplitude'] <= 0:
            raise ValueError(f"amplitude must be positive, got {default_config['amplitude']}")

        if not (0 <= default_config['thickness_variation'] <= 0.5):
            warnings.warn(
                f"thickness_variation ({default_config['thickness_variation']}) should be in [0, 0.5], clamping...")
            default_config['thickness_variation'] = np.clip(default_config['thickness_variation'], 0, 0.5)

        if default_config['waveform'] not in ['sine', 'cosine', 'sawtooth', 'triangle']:
            raise ValueError(
                f"waveform must be one of ['sine', 'cosine', 'sawtooth', 'triangle'], got {default_config['waveform']}")

        return default_config

    def _generate_base_waveform(self, width: int, x_shift: float, amplitude: float, waveform_type: str = 'sine') -> np.ndarray:
        """生成基础波形（向量化实现）"""
        x_range = np.linspace(0, 2 * np.pi, width, endpoint=False) + x_shift * 2 * np.pi

        if waveform_type == 'sine':
            base_curve = amplitude * np.sin(x_range)
        elif waveform_type == 'cosine':
            base_curve = amplitude * np.cos(x_range)
        elif waveform_type == 'sawtooth':
            # 锯齿波
            base_curve = amplitude * (2 * (x_range / (2 * np.pi) - np.floor(0.5 + x_range / (2 * np.pi))))
        elif waveform_type == 'triangle':
            # 三角波
            base_curve = 2 * amplitude * np.abs(2 * (x_range / (2 * np.pi) - np.floor(x_range / (2 * np.pi) + 0.5))) - 1
        else:
            base_curve = amplitude * np.sin(x_range)

        return base_curve

    def _add_noise_to_curve(self, base_curve: np.ndarray, noise_level: float, amplitude: float, random_state: np.random.RandomState) -> np.ndarray:
        """为曲线添加噪声"""
        if noise_level <= 0:
            return base_curve.copy()

        noise = random_state.normal(0, noise_level, base_curve.shape) * amplitude
        return base_curve + noise

    def _smooth_curve(self, curve: np.ndarray, window_ratio: float, polyorder: int = 3) -> np.ndarray:
        """平滑曲线"""
        if len(curve) < 5:  # 曲线太短无法平滑
            return curve

        window_length = max(5, int(len(curve) * window_ratio))
        if window_length % 2 == 0:  # Savitzky-Golay需要奇数窗口
            window_length += 1

        try:
            smoothed = savgol_filter(curve, window_length, polyorder, mode='mirror')
            return smoothed
        except:
            # 如果平滑失败，返回原始曲线
            return curve

    def _calculate_layer_centers(self, num_layers: int, base_center: float, thickness: float,
                                 thickness_variation: float, gap:int, random_state: np.random.RandomState) -> np.ndarray:
        """计算纹层中心位置（向量化实现）"""
        if num_layers <= 0:
            return np.array([])

        # 生成随机的厚度变化
        variations = 1 + random_state.uniform(-thickness_variation, thickness_variation, num_layers)

        # # 累积计算中心位置
        centers = []
        for layer in range(num_layers):
            if layer == 0:
                centers.append(base_center)     ####### 第一个中心位置
            else:
                centers.append(centers[-1]+gap)
            centers.append(centers[-1]+thickness*variations[layer])

        return np.array(centers).astype(int)

    def _create_layer_mask_vectorized(
            self,
            height: int,
            width: int,
            upper_curve: np.ndarray,
            lower_curve: np.ndarray,
            layer_value: int,
            break_config: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        创建带水平断裂的纹层掩模（完全向量化版本）

        优化点：
        - 完全向量化操作，无Python循环
        - 使用NumPy布尔索引高效处理
        - 内存使用优化

        工作流程：
        1. 创建基础层理掩模
        2. 生成断裂位置掩模
        3. 从基础掩模中移除断裂区域
        4. 返回最终掩模
        """
        # 1. 创建行索引矩阵
        row_indices = np.arange(height)[:, np.newaxis]

        # 2. 计算上下边界（确保为整数且在有效范围内）
        lower_bound = np.maximum(0, np.floor(lower_curve)).astype(int)
        upper_bound = np.minimum(height, np.ceil(upper_curve)).astype(int)

        # 3. 创建基础掩模
        mask = (row_indices >= lower_bound+1) & (row_indices < upper_bound)

        # 4. 如果没有断裂配置，直接返回
        if break_config is None or not break_config:
            layer_mask = np.zeros((height, width), dtype=np.uint8)
            layer_mask[mask] = layer_value
            return layer_mask

        # 5. 解析配置
        break_num_range = break_config.get('break_num_range', [0, 0, 0, 0, 0, 1,])
        break_length_range = break_config.get('break_length_range', (5, 20))

        # 7. 随机确定断裂数量
        break_num = np.random.choice(break_num_range)

        # 8. 向量化生成断裂列掩模
        # 创建全为True的列掩模（True表示该列有层理）
        column_mask = np.any(mask, axis=0)  # 形状: (width,)

        if break_num > 0 and np.any(column_mask):
            # 找到有层理的列索引
            valid_columns = np.where(column_mask)[0]

            if len(valid_columns) > 0:
                # 生成断裂起始列（从有效列中随机选择）
                if len(valid_columns) >= break_num:
                    # 从有效列中随机选择断裂起始列
                    start_cols = np.random.choice(
                        valid_columns,
                        size=min(break_num, len(valid_columns)),
                        replace=False,
                    )
                else:
                    # 如果有效列不足，则使用所有有效列
                    start_cols = valid_columns.copy()

                # 创建断裂区域标记数组
                break_region = np.zeros(width, dtype=bool)

                for start_col in start_cols:
                    # 随机确定断裂长度
                    max_length = min(break_length_range[1], width - start_col)
                    if max_length < break_length_range[0]:
                        continue

                    break_length = np.random.randint(
                        break_length_range[0],
                        max_length + 1
                    )

                    # 计算断裂影响的列范围
                    end_col = min(start_col + break_length, width)
                    break_region[start_col+1:end_col] = True

                # 9. 从基础掩模中移除断裂区域
                # 创建断裂列掩模（True表示该列没有断裂）
                no_break_columns = ~break_region

                # 将断裂列的掩模设置为False
                # 通过布尔索引高效移除断裂列
                for col in range(width):
                    if not no_break_columns[col]:
                        # 完全清除该列的层理
                        mask[:, col] = False

        # 10. 创建最终掩模图像
        layer_mask = np.zeros((height, width), dtype=np.uint8)
        layer_mask[mask] = layer_value

        return layer_mask


    def _crop_image_to_content(self, image: np.ndarray, min_margin: int = 5) -> np.ndarray:
        """裁剪图像到有效内容区域"""
        if image.size == 0:
            return image

        # 找到有内容的行
        non_zero_rows = np.where(np.any(image != self.VALUE_BACKGROUND, axis=1))[0]

        if len(non_zero_rows) == 0:
            return image[:min_margin, :]  # 返回最小高度的空白图像

        start_row = max(0, non_zero_rows[0] - min_margin)
        end_row = min(image.shape[0], non_zero_rows[-1] + 1 + min_margin)

        return image[start_row:end_row, :]

    def generate_bedding_by_config(self, config: Optional[Dict] = None, random_seed: Optional[int] = None) -> np.ndarray:
        """
        生成基于波形曲线的周期性层理、纹层图像（优化版本）
        参数:
            config: 配置字典，包含以下参数:
                width: 图像宽度（默认256）
                height: 图像高度（默认400）
                x_shift: 相位偏移 [0,1]（默认0.4）
                noise_level: 噪声水平 [0,1]（默认0.1）
                thickness: 层厚（像素）（默认10）
                amplitude: 波动幅度，可为[0,1]或像素值（默认0.1）
                thickness_variation: 厚度变化范围 [0,0.5]（默认0.02）
                waveform: 波形类型 ['sine','cosine','sawtooth','triangle']（默认'sine'）
                smooth_window: 平滑窗口比例 [0,1]（默认 0.1）
            random_seed: 随机种子（默认None）
        返回:
            np.ndarray: 纹层掩模图像
        """
        # 1. 设置随机种子、配置读取
        random_state = np.random.RandomState(random_seed)

        # 提取参数
        width = config['width']
        height = config['height']
        x_shift = config['x_shift']
        noise_level = config['noise_level']
        thickness = config['laminar_thickness']
        gap = config['laminar_gap']
        amplitude = config['amplitude']
        thickness_variation = config['thickness_variation']
        waveform = config['waveform']
        smooth_window = config['smooth_window']
        break_config = config['break_config']

        # 2. 创建空白图像
        img = np.full((height, width), self.VALUE_BACKGROUND, dtype=np.uint8)

        # 3. 标准化 层厚 振幅 以及 纹层之间的间隔
        if amplitude <= 1:  # 如果小于等于1，认为是比例
            amplitude_pixels = int(amplitude * height)
        else:  # 否则认为是像素值
            amplitude_pixels = int(amplitude)
        if gap <= 1:  # 如果小于等于1，认为是比例
            gap_pixels = int(gap * height)
        else:  # 否则认为是像素值
            gap_pixels = int(gap)
        if thickness <= 1:
            thickness_pixels = int(thickness * height)
        else:
            thickness_pixels = int(thickness)

        # 4. 生成基础波形
        base_wave = self._generate_base_waveform(width, x_shift, amplitude_pixels, waveform)

        # 5. 计算纹层数量
        # usable_height = height - 2 * amplitude_pixels
        # signal_num = max(1, math.floor(usable_height / thickness_pixels))
        usable_height = height - 2 * amplitude_pixels
        signal_num = max(1, math.floor((usable_height+gap_pixels) / (thickness_pixels+gap_pixels)))

        # 6. 预分配曲线数组
        curves = []
        centers = self._calculate_layer_centers(signal_num, amplitude_pixels, thickness_pixels, thickness_variation, gap_pixels, random_state)

        # 7. 生成所有纹层曲线
        for i in range(2*signal_num):
            # 添加噪声
            noisy_curve = self._add_noise_to_curve(base_wave, noise_level, amplitude_pixels, random_state)

            # 平滑处理
            smoothed_curve = self._smooth_curve(noisy_curve, smooth_window)

            # 存储曲线
            curves.append(smoothed_curve)

            # 8. 创建纹层掩模（每两条曲线形成一个纹层）
            if i >= 1 and i % 2 == 1:  # 从第二条曲线开始，每对曲线形成一个纹层
                lower_curve = curves[i - 1] + centers[i - 1]
                upper_curve = curves[i] + centers[i]

                # 使用向量化方法创建纹层
                layer_mask = self._create_layer_mask_vectorized(height, width, upper_curve, lower_curve, layer_value=self.VALUE_BEDDING, break_config=break_config)

                # 合并到主图像
                img = np.where(layer_mask != self.VALUE_BACKGROUND, layer_mask, img)

        # 9. 裁剪图像到有效内容
        img = self._crop_image_to_content(img)

        # 10. 记录生成信息
        self._log_generation_info(width, height, signal_num, centers, amplitude_pixels, thickness_pixels, gap_pixels, waveform)

        return img

    def get_bedding_mask_by_config(self, config: Optional[Dict] = None, random_seed: Optional[int] = None) -> np.ndarray:
        # 1. 验证和合并配置
        config = self._validate_config(config or {})
        config_target = config.copy()
        config_target['noise_level'] = 0
        config_target['break_config'] = {}

        if random_seed is None:
            random_seed = config_target['random_seed']

        bedding_mask = self.generate_bedding_by_config(config, random_seed)
        bedding_target = self.generate_bedding_by_config(config_target, random_seed)

        return bedding_mask, bedding_target

    def _log_generation_info(self, width: int, height: int, signal_num: int,
                             centers: List[int], amplitude: int, thickness_pixels:int, gap_pixels:int, waveform: str):
        """记录生成信息"""
        print(f"Generated bedding image: {width}x{height}, "
              f"layers: {signal_num // 2}, centers: {centers[:10]}..., "
              f"amplitude: {amplitude}, thickness: {thickness_pixels}, gap: {gap_pixels}, waveform: {waveform}")




# 使用示例
if __name__ == "__main__":
    # 创建优化后的模拟器
    simulator = bedding_mask_simulator()

    # 1. 生成单个纹层图像
    config = {
        'width': 256,
        'height': 400,
        'x_shift': 0.3,
        'noise_level': 0.01,
        'laminar_thickness': 2,
        'laminar_gap': 2,
        'amplitude': 0.06,
        'thickness_variation': 0.05,
        'waveform': 'sine',
        'random_seed': 42,
    }

    masks_bedding_list = []
    for i in range(4):
        mask_bedding, bedding_target = simulator.get_bedding_mask_by_config(config)
        masks_bedding_list.append(mask_bedding)
        masks_bedding_list.append(bedding_target)

    show_Pic(masks_bedding_list, figure=[16, 12])
