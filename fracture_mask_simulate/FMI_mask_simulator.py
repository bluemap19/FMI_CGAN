import math
import numpy as np
from pygments.formatters import img
from scipy.signal import savgol_filter

from src_ele.pic_opeeration import show_Pic


class FMI_mask_simulator:
    def __init__(self):
        self.VALUE_BACKGROUND = 0
        self.VALUE_HIGH_CONDUCTIVITY = 1
        self.VALUE_HIGH_RESISTANCE = 2
        self.VALUE_INDUCED = 3
        self.VALUE_BEDDING = 4
        self.VALUE_BREAKOUT = 5

        # 基础的纹层参数配置
        self.CONFIG_BEDDING_DEFAULT = {
            'width_background':256,                    # 图像的宽度
            'height_background':400,                   # 图像的高度
            'bedding_x_shift':0.4,          # 纹层在X方向的角度，也即是纹层与正北方的夹角
            'noise_level':0.1,             # 噪声水平
            'bedding_thickness':10,         # 纹层层厚
            'bedding_amplitude':0.1,        # 纹层的波动幅度，也即是纹层与水平界面的角度
            'thickness_random':[-0.01, 0.01]  # 纹层发育密度的随机值，会随机的将thickness进行放大或者是缩小
        }

        # 基础的裂缝参数配置
        self.CONFIG_CRACK_DEFAULT = {
            'width_background': 256,        # 背景图像的宽度
            'height_background': 400,       # 背景图像的高度
            'crack_x_shift': np.random.random(),           # 正弦裂缝沿着x方向的偏移角度，0-1，百分比配置，这个主要是
            'noise_level': 0.15+0.15*np.random.random(),            # 裂缝之间的噪声水平
            'crack_thickness': 25,              # 裂缝的宽度
            'crack_value': self.VALUE_HIGH_CONDUCTIVITY,        # 裂缝填充信息，是高导缝、高阻缝、层理缝或者是诱导缝、断层
        }


    def generate_sin_crack_by_config(self, config_crack={}):
        """
        generate a new crack by config
        根据配置文件生成相对应的单条正弦裂缝图像
        config_crack['width_background']        ：背景的宽度
        config_crack['height_background']       ：背景的长度
        config_crack['crack_x_shift']           ：裂缝在X方向上的偏移百分比
        config_crack['noise_level']             ：噪声水平
        config_crack['crack_thickness']             ：裂缝的宽度
        config_crack['crack_value']             ：裂缝是什么类型的裂缝，分别包括高导、高阻、诱导
        """
        width, height, crack_x_shift, noise_level, crack_thickness, crack_value = config_crack['width_background'], config_crack['height_background'], config_crack['crack_x_shift'], config_crack['noise_level'], config_crack['crack_thickness'], config_crack['crack_value']

        # 1. 创建空白背景图像
        img = np.zeros((height, width), dtype=np.uint8)
        img.fill(self.VALUE_BACKGROUND)
        # crack_height: 裂缝中心线高度
        crack_height = (height) // 2
        # crack_amplitude: 裂缝振幅 (控制波动幅度)
        crack_amplitude = int((height - crack_thickness) // 2 * 0.9)

        # 2. 沿X方向进行曲线旋转偏移的设定
        if isinstance(crack_x_shift, float):
            if crack_x_shift<1:
                x_shift = crack_x_shift * 2 * np.pi
            else:
                x_shift = crack_x_shift
        elif isinstance(crack_x_shift, int):
            x_shift = crack_x_shift / width * 2 * np.pi
        else:
            x_shift = 0

        print(f'generated crack x_shifted is :{crack_x_shift:.4f}, re calculted it as {x_shift:.4f} π')

        # 3. 生成单个周期的正弦曲线数据
        # 生成一个完整周期的正弦数据点 (长度为width)
        x_range = np.linspace(0 + x_shift, 2 * np.pi + x_shift, width)  # 0到2π的一个完整周期
        y_signal = crack_amplitude * np.sin(x_range)

        # 4. 对正弦曲线添加噪声，并生成上下两边的正弦线
        if noise_level > 0:
            y_signal1 = np.random.normal(0, noise_level, (y_signal.shape[0],)) * crack_amplitude + y_signal
            y_signal2 = np.random.normal(0, noise_level, (y_signal.shape[0],)) * crack_amplitude + y_signal
        else:
            y_signal1 = y_signal
            y_signal2 = y_signal

        # 5. 对噪声正弦曲线进行平滑降噪
        # # 卷积平滑滤波
        # window_size = 7
        # window = np.ones(window_size) / window_size
        # y_signal = np.convolve(y_signal, window, mode='same')
        # SG滤波
        window_size = int(0.1 * width)
        polyorder = 3  # polyorder为多项式拟合的阶数,它越小，则平滑效果越明显；越大，则更贴近原始曲线。
        y_signal1 = savgol_filter(y_signal1, window_size, polyorder)
        y_signal2 = savgol_filter(y_signal2, window_size, polyorder)

        # 3. 确保所有点都在图像范围内
        crack_up = y_signal1 + crack_height + crack_thickness // 2
        crack_down = y_signal2 + crack_height - crack_thickness // 2
        crack_up = np.clip(crack_up, 0, height-1)
        crack_down = np.clip(crack_down, 0, height-1)

        # 4. 创建裂缝区域
        # 创建行索引矩阵
        rows = np.arange(height)[:, np.newaxis]
        # 创建布尔掩码
        mask = (rows >= crack_down) & (rows < crack_up)
        # 一次性填充所有像素
        img[mask] = crack_value

        if 'crack_cut_off' in config_crack.keys() and 'cut_off_width' in config_crack.keys() and 'noise_cut_off' in config_crack.keys():
            # # 裂缝的随机断裂，产生裂缝在水平方向的随机断裂
            img = self.add_random_crack_cut_off(img.copy(), x_shift, config_crack['crack_cut_off'], config_crack['cut_off_width'], config_crack['noise_cut_off'])

        return img

    def crack_cut_off_mask_fun1(self, mask, cut_off_bottom_center, cut_off_top_center, cut_off_width, noise_level):
        """
            添加倾斜裂缝截断（断层）到掩模中

            参数:
                mask: 原始掩模图像 [height, width]
                cut_off_bottom_center: 底部截断中心点x坐标
                cut_off_top_center: 顶部截断中心点x坐标
                cut_off_width: 截断宽度（像素）
                noise_level: 截断边缘噪声水平（像素）

            返回:
                添加截断后的掩模图像
        """
        height, width = mask.shape
        x_dict = [cut_off_top_center, cut_off_bottom_center]
        y_dict = [0, height]

        # 1. 计算截断线的线性方程 y = ax + b
        # 通过两个点（顶部中心、底部中心）确定一条直线
        a, b = np.polyfit(x_dict, y_dict, 1)

        # 2. 计算截断线的水平范围
        x_range = abs(cut_off_bottom_center - cut_off_top_center)

        # 3. 沿x方向逐点处理截断
        for i in range(x_range):
            # 确定x方向的处理顺序
            if cut_off_top_center < cut_off_bottom_center:
                x = cut_off_bottom_center - i  # 从左到右
            elif cut_off_top_center > cut_off_bottom_center:
                x = cut_off_bottom_center + i  # 从右到左
            else:
                raise ValueError('cut_off_top_center 必须不等于 cut_off_bottom_center')

            # 4. 计算截断线在当前x处的y坐标
            y = a * x + b

            # 5. 计算截断区域的上下边界，添加噪声
            # 安全处理 noise_level 为 0 的情况
            if noise_level > 0:
                # 添加随机噪声到截断边界
                noise_up = np.random.randint(-noise_level, noise_level)
                noise_down = np.random.randint(-noise_level, noise_level)
            else:
                # noise_level 为 0 时，不使用噪声
                noise_up = 0
                noise_down = 0

            # 计算截断上边界
            y_up = y + cut_off_width // 2 + noise_up
            # 确保 y_up 在 [0, height) 范围内
            y_up = int((y_up + height) % height)  # 取模确保不越界
            y_up = min(y_up, height)  # 二次保护，确保不超过图像高度

            # 计算截断下边界
            y_down = y - cut_off_width // 2 + noise_down
            y_down = int(max(y_down, 0))  # 确保不小于0

            # 6. 处理x坐标的边界情况
            if x >= width:
                x = x % width  # 循环映射到图像宽度内
            elif x < 0:
                x = width + x  # 处理负坐标
            # 注意：这里不需要 else，因为 x 已经在 [0, width) 内

            # 7. 应用截断到掩模
            if y_down < y_up:
                # 将截断区域设置为背景值（擦除裂缝）
                mask[y_down:y_up, x] = 1

        return mask

    def crack_cut_off_mask_fun2(self, mask, cut_off_width, noise_level):
        """
            添加垂直裂缝截断（垂直断裂带）到掩模中

            参数:
                mask: 原始掩模图像 [height, width]
                cut_off_width: 截断宽度（像素）
                noise_level: 截断边缘噪声水平（像素）

            返回:
                添加垂直截断后的掩模图像
        """
        height, width = mask.shape

        # 1. 选择裂缝最薄弱的列（裂缝像素最少的前30%）
        count = int(0.25 * width)  # 选择前30%的列
        mask_x_statics = np.sum(mask, axis=0)  # 每列裂缝像素数量

        # 获取裂缝最薄弱的列索引
        sorted_indices = np.argsort(mask_x_statics)[:count]

        # 2. 从薄弱列中随机选择一列进行垂直截断
        x_random = np.random.choice(sorted_indices)

        # 3. 沿y方向添加垂直截断带
        for i in range(height):
            # 安全处理 noise_level 为 0 的情况
            half_noise = noise_level // 2
            if half_noise > 0:
                # 添加随机噪声到垂直截断的起始和结束位置
                noise_shift_start = np.random.randint(-half_noise, half_noise)
                noise_shift_end = np.random.randint(-half_noise, half_noise)
            else:
                # half_noise 为 0 时，不使用噪声
                noise_shift_start = 0
                noise_shift_end = 0

            # 计算垂直截断的x范围
            x_start = x_random - cut_off_width // 12 + noise_shift_start
            x_end = x_random + cut_off_width // 12 + noise_shift_end

            # 确保x坐标在图像范围内
            x_start = max(0, min(x_start, width - 1))
            x_end = max(0, min(x_end, width - 1))

            # 应用垂直截断
            mask[i, x_start:x_end] = 1

        return mask

    def add_random_crack_cut_off(self, img=np.array([]), x_shift=0.0, cut_off_num=1, cut_off_width=0.05, noise_level=0.2):
        """
        为裂缝图像添加随机截断（模拟断层和断裂带）
        参数:
            img: 输入裂缝图像 [height, width]
            crack_x_shift: 裂缝在x方向的相位偏移
            cut_off_num: 截断数量
            cut_off_width: 截断宽度（归一化或像素值）
            noise_level: 截断噪声水平（归一化或像素值）
        返回:
            添加截断后的裂缝图像
        """
        # 1. 读取图像基本的形状参数
        height, width = img.shape

        # 2. 沿X方向进行曲线旋转偏移的设定
        x_shift = int(0.75*width - x_shift / (2 * np.pi) * width)

        # 3. 初始化裂缝断裂的宽度
        if isinstance(cut_off_width, float):
            cut_off_width = int(cut_off_width * height)
        else:
            cut_off_width = int(cut_off_width)

        # 4. 初始化裂缝断裂面 的 噪声水平
        if isinstance(noise_level, float):
            noise_level = int(noise_level * width)
        else:
            noise_level = int(noise_level)

        # 5. 安全处理：确保 noise_level 至少为 1
        # 这是解决报错的关键：防止 noise_level 为 0
        noise_level = max(1, noise_level)  # 至少为 1 像素

        # 6. 裂缝底部的中心
        cut_off_bottom_center = (x_shift + width) % width

        # 7. 裂缝底部中心对应的顶部两个可断裂中心
        cut_off_top_center1 = cut_off_bottom_center - int(0.5 * width)
        cut_off_top_center2 = cut_off_bottom_center + int(0.5 * width)

        # 8. 创建截断掩模
        mask = np.zeros_like(img)

        # 9. 根据截断数量添加不同类型的截断
        if cut_off_num == 1:
            # 随机选择一个顶部中心点添加倾斜截断
            cut_off_top_center = np.random.choice([cut_off_top_center1, cut_off_top_center2])
            mask = self.crack_cut_off_mask_fun1(mask, cut_off_bottom_center, cut_off_top_center, cut_off_width, noise_level)
        elif cut_off_num >= 2:
            # 添加两个倾斜截断
            list_top_center = [cut_off_top_center1, cut_off_top_center2]
            for cut_off_top_center in list_top_center:
                mask = self.crack_cut_off_mask_fun1(mask, cut_off_bottom_center, cut_off_top_center, cut_off_width, noise_level)
        elif cut_off_num == 0:
            pass  # 不添加截断
        else:
            raise ValueError('cut_off_num 必须是大于等于0的整数')

        # 10. 如果需要，添加垂直截断
        if cut_off_num > 2:
            mask = self.crack_cut_off_mask_fun2(mask, cut_off_width, noise_level)

        # 11. 将截断区域恢复为背景值
        img[mask == 1] = self.VALUE_BACKGROUND

        return img

    def generate_crack_by_config(self, crack_config={}, crack_ratio_setting=0.04, trying_num=20):
        """
        # 根据裂缝配置尝试生成新的裂缝
        主要功能是根据裂缝配置参数、裂缝最小面积占比、最大尝试次数进行尝试的裂缝mask构造
        crack_config：           裂缝生成相对应的配置
        crack_ratio_setting：    裂缝面积占整体面积最小百分比，低于这个值，裂缝将会重新尝试重建
        trying_num：             最大尝试次数，最多尝试多少次进行裂缝的重建
        返回的结果是裂缝mask、分割目标对应的mask
        """
        ################# ['width_background'], ['height_background'], ['crack_x_shift'], ['noise_level'], ['crack_thickness'], ['crack_break_ratio'], ['crack_value']
        crack_config_input = self.get_config_norm_crack()

        if crack_config:
            crack_config_input.update(crack_config)

        ########## 这个config，是用来做出来给深度学习模型的，是模型的训练数据
        crack_config_target = {
            'width_background': crack_config_input['width_background'],  # 背景图像的宽度
            'height_background': crack_config_input['height_background'],  # 背景图像的高度
            'crack_x_shift': crack_config_input['crack_x_shift'],  # 正弦裂缝沿着x方向的偏移角度，0-1，百分比配置，这个主要是
            'noise_level': 0,  # 裂缝之间的噪声水平
            'crack_thickness': 10,  # 裂缝的宽度
            'crack_value': crack_config_input['crack_value'],  # 裂缝填充信息，是高导缝、高阻缝、层理缝或者是诱导缝、断层
        }

        crack_ratio = 0
        crack_random = np.array([])
        while(crack_ratio<crack_ratio_setting and trying_num > 0):
            crack_random = self.generate_sin_crack_by_config(crack_config_input)
            crack_ratio = np.sum(crack_random)/(crack_random.size*self.VALUE_HIGH_CONDUCTIVITY)
            trying_num -= 1

        crack_target = self.generate_sin_crack_by_config(crack_config_target)

        return crack_random, crack_target

    def generate_bedding_by_config(self, config_bedding={}, random_seed=42):
        """
        生成基于单个周期正弦曲线的周期性层理、纹层图像
        参数:
        width_background: int，图像宽度
        height_background: int，图像高度
        bedding_x_shift: float 或 int 正弦线沿X方向旋转像素个数，亦或者是百分比
        noise_level: float，噪声强度 (0-1)
        bedding_thickness: int，层厚，层理之间的宽度，决定了是 密集层理 还是 稀疏层理(像素)
        bedding_amplitude: int 或 float，层理高度，控制波动幅度，层理与地层的角度，决定了是低角度层理 还是 高角度层理 (像素)
        返回:
        二值图像
        """
        np.random.seed(random_seed)

        # 默认配置初始化
        config_default = self.CONFIG_BEDDING_DEFAULT

        # 配置合并
        config_bedding = {**config_default, **config_bedding}

        # 配置读取
        width_background = config_bedding['width_background']
        height_background = config_bedding['height_background']
        bedding_x_shift = config_bedding['bedding_x_shift']
        noise_level = config_bedding['noise_level']
        bedding_thickness = config_bedding['bedding_thickness']
        bedding_amplitude = config_bedding['bedding_amplitude']
        thickness_random = config_bedding['thickness_random']

        # 配置校正，确定他们的类型都是正确、合适的
        if not (isinstance(config_bedding['bedding_x_shift'], float) or isinstance(config_bedding['bedding_x_shift'], int)):
            print('config_bedding["bedding_x_shift"] must be float or int:{}'.format(config_bedding['bedding_x_shift']))
            exit(0)
        if not isinstance(config_bedding['noise_level'], float):
            print('config_bedding["noise_level"] must be float:{}'.format(config_bedding['noise_level']))
            exit(0)
        if not isinstance(config_bedding['bedding_thickness'], int):
            print('config_bedding["bedding_thickness"] must be int:{}'.format(config_bedding['bedding_thickness']))
            exit(0)
        if not (isinstance(config_bedding['bedding_amplitude'], int) or isinstance(config_bedding['bedding_amplitude'], float)):
            print('config_bedding["bedding_amplitude"] must be int:{}'.format(config_bedding['bedding_amplitude']))
            exit(0)
        if not isinstance(config_bedding['thickness_random'], list):
            print('config_bedding["thickness_random"] must be list, its a bedding_amplitude random range:{}'.format(config_bedding['thickness_random']))
            exit(0)

        # 1. 创建空白图像
        img = np.zeros((height_background, width_background), dtype=np.uint8)
        img.fill(self.VALUE_BACKGROUND)

        # 2. 沿X方向进行曲线旋转偏移的设定
        if isinstance(bedding_x_shift, float):
            x_shift = bedding_x_shift * 2 * np.pi
        elif isinstance(bedding_x_shift, int):
            x_shift = bedding_x_shift / width_background * 2 * np.pi
        else:
            x_shift = 0

        # 纹层或层理的幅度设定
        if isinstance(bedding_amplitude, float):
            bedding_amplitude = int(bedding_amplitude * height_background)
        else:
            bedding_amplitude = bedding_amplitude

        # 3. 生成单个周期的正弦曲线数据，也是所有层理的基础
        # 生成一个完整周期的正弦数据点 (长度为width_background)
        x_range = np.linspace(0 + x_shift, 2 * np.pi + x_shift, width_background)  # 0到2π的一个完整周期
        y_signal = bedding_amplitude * np.sin(x_range)

        # 计算本次层理的正弦线个数
        signal_num = math.floor(0.99 * (height_background - 3 * bedding_amplitude) / bedding_thickness)
        center_height_list = []         # 纹层垂直方向的正中心
        signal_list = []                # 纹层对应的信号

        for i in range(signal_num):
            # 4. 对正弦曲线添加噪声
            y_signal_temp = np.random.normal(0, noise_level, (y_signal.shape[0],)) * bedding_amplitude + y_signal

            # 5. 对噪声正弦曲线进行平滑降噪
            # **************************************************************
            window_size = int(width_background * 0.1)
            polyorder = 3                                                           # polyorder为多项式拟合的阶数,它越小，则平滑效果越明显；越大，则更贴近原始曲线。
            y_signal_temp = savgol_filter(y_signal_temp, window_size, polyorder)    # SG滤波
            # **************************************************************

            signal_list.append(y_signal_temp)

            # 6.累计计算正弦线中心的 垂直方向index位置
            if len(center_height_list) == 0:
                center_height_list.append(bedding_amplitude)
            else:
                bedding_center = center_height_list[-1] + bedding_thickness*(np.random.uniform(thickness_random[0], thickness_random[1])+1)
                center_height_list.append(int(bedding_center))

            if i % 2 == 1:
                crack_down = signal_list[i - 1] + center_height_list[i - 1]
                crack_up = signal_list[i] + center_height_list[i]

                for j in range(width_background):
                    # 计算裂缝的上下边界
                    y_min = int(max(0, crack_down[j]))
                    y_max = int(min(height_background, crack_up[j]))

                    # 在裂缝区域填充
                    if y_min < y_max:
                        img[y_min:y_max, j] = self.VALUE_BEDDING
        if len(center_height_list)%2 == 0:
            height_new = center_height_list[-1]+bedding_amplitude
        else:
            height_new = center_height_list[-2]+bedding_amplitude
        # int((signal_num - 1) * bedding_thickness + 2 * bedding_amplitude + bedding_amplitude * (max(thickness_random) - 1))
        img = img[:height_new, :]
        print('generate bedding information as: width_background:{}; height:{}; layer num:{}; center list{}; bedding amplitude:{}'.format(
                width_background, height_background, signal_num, center_height_list, bedding_amplitude))

        return img

    def get_config_bedding_random(self):
        """随机纹层配置信息"""
        # 基础的纹层参数配置
        config_bedding = {
            'width_background': 256,  # 图像的宽度
            'height_background': 400,  # 图像的高度
            'bedding_x_shift': np.random.random(),  # 纹层在X方向的角度，也即是纹层与正北方的夹角
            'noise_level': np.random.random()*0.4,  # 噪声水平
            'bedding_thickness': np.random.randint(5, 20),  # 纹层层厚
            'bedding_amplitude': (np.random.random()+0.5)*0.05,  # 纹层的波动幅度，也即是纹层与水平界面的角度
            'thickness_random': [-0.05, 0.05]  # 纹层发育密度的随机值，会随机的将thickness进行放大或者是缩小
        }
        config_bedding['noise_level'] = config_bedding['bedding_amplitude']*1.5
        return config_bedding

    def generate_bedding_random(self):
        """
        generate a random bedding fmi mask
        """
        config_bedding_random = self.get_config_bedding_random()
        config_bedding_target = config_bedding_random.copy()
        config_bedding_target['noise_level'] = 0.0

        bedding_mask = self.generate_bedding_by_config(config_bedding_random)
        bedding_target = self.generate_bedding_by_config(config_bedding_target)

        return bedding_mask, bedding_target

    def get_config_norm_crack(self):
        """好用常见的裂缝配置，这个是细长裂缝，搭配上随机自然的裂缝断裂，主要问题是不是宽裂缝"""
        return self.CONFIG_CRACK_DEFAULT
    def get_config_crack_random(self):
        """随机裂缝配置信息"""
        crack_norm = {
            'width_background': 256,        # 背景图像的宽度
            'height_background': np.random.randint(300, 500),       # 背景图像的高度
            'crack_x_shift': np.random.random(),           # 正弦裂缝沿着x方向的偏移角度，0-1，百分比配置，这个主要是
            'noise_level': 0.15+0.15*np.random.random(),            # 裂缝之间的噪声水平
            'crack_thickness': np.random.randint(22, 34),              # 裂缝的宽度
            'crack_value': self.VALUE_HIGH_CONDUCTIVITY,        # 裂缝填充信息，是高导缝、高阻缝、层理缝或者是诱导缝、断层
            'crack_cut_off': np.random.choice([0, 1, 1, 1, 2, 2, 3, 3, 3, 4]),             # 裂缝可能不是连续的，裂缝水平方向进行截断的个数，一般是1-2次，
            'cut_off_width': 0.2+0.4*np.random.random(),           # 水平方向截断的裂缝断裂宽度，0-1，这个
            'noise_cut_off': 0.03,          # 水平方向截断的裂缝截断面的噪声水平
        }
        return crack_norm

    def get_config_crack_wide(self):
        """宽裂缝配置信息"""
        crack_config_wide = {
            'width_background': 256,        # 背景图像的宽度
            'height_background': np.random.randint(300, 500),       # 背景图像的高度
            'crack_x_shift': np.random.random(),           # 正弦裂缝沿着x方向的偏移角度，0-1，百分比配置，这个主要是
            'noise_level': 0.15+np.random.random()*0.3,             # 描述裂缝正弦线的噪声水平
            'crack_thickness': np.random.randint(50, 100),              # 裂缝的宽度
            'crack_value': self.VALUE_HIGH_CONDUCTIVITY,        # 裂缝填充信息，是高导缝、高阻缝、层理缝或者是诱导缝、断层
            'crack_cut_off': np.random.choice([0, 1, 1, 1, 2, 2, 3, 3, 3, 4]),             # 裂缝可能不是连续的，裂缝水平方向进行截断的个数，一般是1-2次，
            'cut_off_width': 0.25+np.random.random()*0.2,           # 水平方向截断的裂缝断裂宽度，0-1，这个
            'noise_cut_off': 0.02+np.random.random()*0.02,          # 水平方向截断的裂缝截断面的噪声水平
        }
        return crack_config_wide

    def generate_crack_norm(self, crack_config={}):
        """生产一个细长条的，电成像裂缝图像"""
        crack_default = {
            'crack_cut_off': np.random.choice([0, 1, 1, 1, 2, 2, 3, 3, 3, 4]),             # 裂缝可能不是连续的，裂缝水平方向进行截断的个数，一般是1-2次，
            'cut_off_width': 0.2+0.4*np.random.random(),           # 水平方向截断的裂缝断裂宽度，0-1，这个
            'noise_cut_off': 0.03,          # 水平方向截断的裂缝截断面的噪声水平
        }
        config_crack = self.get_config_norm_crack()
        config_crack.update(crack_default)
        if crack_config:
            config_crack.update(crack_config)
        crack_mask, crack_target = self.generate_crack_by_config(config_crack, trying_num=20, crack_ratio_setting=0.035)

        return crack_mask, crack_target
    def generate_crack_wide(self, crack_config={}):
        """
        生产一个宽缝图像
        """
        config_crack = self.get_config_crack_wide()
        if crack_config:
            config_crack.update(crack_config)
        crack_mask, crack_target = self.generate_crack_by_config(config_crack, trying_num=20, crack_ratio_setting=0.035)

        return crack_mask, crack_target
    def generate_crack_random(self, crack_config={}):
        """
        生产一个宽缝图像
        """
        config_crack = self.get_config_crack_random()
        if crack_config:
            config_crack.update(crack_config)
        crack_mask, crack_target = self.generate_crack_by_config(config_crack, trying_num=20, crack_ratio_setting=0.035)

        return crack_mask, crack_target

    def add_crack_to_img(self, config_crack={}, img_background=np.array([])):
        """
        add a new crack on input background image by config
        """

    def add_bedding_to_img(self, bedding_config={}, img_background=np.array([])):
        """
        add a new bedding on input background image by config
        """
        pass

if __name__ == '__main__':
    fms = FMI_mask_simulator()

    # masks_crack_list = []
    # for i in range(8):
    #     # crack_random = fms.generate_crack_random()
    #     # crack_random, crack_target = fms.generate_crack_norm(crack_config={'crack_x_shift': i * 0.1})
    #     crack_random, crack_target = fms.generate_crack_random(crack_config={'crack_x_shift': i * 0.1})
    #     # crack_random, crack_target = fms.generate_crack_wide(crack_config={'crack_x_shift':i*0.1})
    #     masks_crack_list.append(crack_random*64)
    #     masks_crack_list.append(crack_target*64)
    #
    # show_Pic(masks_crack_list, pic_order='44', figure=[10, 12])

    masks_bedding_list = []
    for i in range(8):
        mask_bedding, mask_target = fms.generate_bedding_random()
        masks_bedding_list.append(mask_bedding)
        masks_bedding_list.append(mask_target)

    show_Pic(masks_bedding_list, pic_order='44', figure=[12, 12])
