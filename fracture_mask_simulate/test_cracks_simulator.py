import cv2
import numpy as np
from scipy.signal import savgol_filter
from src_ele.pic_opeeration import show_Pic

def get_pic_top_and_bottle_index(IMG):
    # 多缝的截图程序，把上下部分，没什么用的黑色背景板截图丢掉
    SUM_front = []
    SUN_behind = []
    for i in range(IMG.shape[0]):
        SUM_front.append(np.sum(IMG[:i + 1, :]))
        SUN_behind.append(np.sum(IMG[i:, :]))
    # 计算截图使用的最大最小值，即截图使用的图像上下限index
    index_start = np.max(np.where(np.array(SUM_front) <= 1))
    index_end = np.min(np.where(np.array(SUN_behind) <= 1))

    return index_start, index_end


class cracks_mask_simulator:
    def __init__(self):
        self.VALUE_BACKGROUND = 0
        self.VALUE_HIGH_CONDUCTIVITY = 1
        self.VALUE_HIGH_RESISTANCE = 2
        self.VALUE_INDUCED = 3
        self.VALUE_BEDDING = 4
        self.VALUE_BREAKOUT = 5

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
        config_crack['crack_thickness']         ：裂缝的宽度
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

        # print(f'generated crack x_shifted is :{crack_x_shift:.4f}, re calculted it as {x_shift:.4f} π')

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
            'crack_thickness': max(crack_config_input['crack_thickness']//3, 3),  # 裂缝的宽度
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

    def get_config_norm_crack(self):
        """好用常见的裂缝配置，这个是细长裂缝，搭配上随机自然的裂缝断裂，主要问题是不是宽裂缝"""
        config_crack_temp = {
            'crack_cut_off': np.random.choice([0, 0, 1, 1, 1, 2, 2, 3, 4, 5]),  # 裂缝可能不是连续的，裂缝水平方向进行截断的个数，一般是1-2次，
            'cut_off_width': 0.2 + 0.8 * np.random.random(),  # 水平方向截断的裂缝断裂宽度，0-1，这个
            'noise_cut_off': 0.03,  # 水平方向截断的裂缝截断面的噪声水平
        }
        config_norm = self.CONFIG_CRACK_DEFAULT
        config_norm.update(config_crack_temp)
        return config_norm

    def get_config_crack_random(self):
        """随机裂缝配置信息"""
        crack_norm = {
            'width_background': 256,        # 背景图像的宽度
            'height_background': np.random.randint(300, 500),       # 背景图像的高度
            'crack_x_shift': np.random.random(),           # 正弦裂缝沿着x方向的偏移角度，0-1，百分比配置，这个主要是
            'noise_level': 0.15+0.15*np.random.random(),            # 裂缝之间的噪声水平
            'crack_thickness': np.random.randint(22, 34),              # 裂缝的宽度
            'crack_value': self.VALUE_HIGH_CONDUCTIVITY,        # 裂缝填充信息，是高导缝、高阻缝、层理缝或者是诱导缝、断层
            'crack_cut_off': np.random.choice([0, 0, 1, 1, 1, 2, 2, 3, 4, 5]),             # 裂缝可能不是连续的，裂缝水平方向进行截断的个数，一般是1-2次，
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

    def generate_random_multi_cracks(self, config_multi_fractures={}):
        """
        根据配置信息config_multi_fractures，产生多缝地层，这个裂缝地层可能是平行缝，也可能是交叉缝
        这个的主要作用是进行配置，即给一些残缺的配置config，他将汇出完整且合适的配置config，并进行mask的模拟生成
        然后返回的是裂缝 地层图像IMG, 裂缝默认模拟配置config_multi_fractures, 裂缝参数曲线df_cracks_para
        """
        config_multi_fractures_default = {
            'height_background': 512,
            'width_background': 256,
            'crack_x_shift': 0.6,
            'crack_start_height': 50,
            'cracks_num': np.random.randint(2, 3),
            'parallel_cracks': np.random.choice([True, False, False]),
            'crack_width': np.random.randint(40, 90),                  # 只是为了配置参数的统一，但是没什么实际用处，起不到什么作用
            'crack_value': self.VALUE_HIGH_RESISTANCE,
        }

        # 多缝模拟的参数曲线初始化
        config_multi_fractures = {**config_multi_fractures_default, **config_multi_fractures}
        height_background = config_multi_fractures['height_background']             # 背景高度
        width_background = config_multi_fractures['width_background']               # 背景宽度
        crack_x_shift = config_multi_fractures['crack_x_shift']                     # x方向偏移
        crack_start_height = config_multi_fractures['crack_start_height']           # y方向，从哪个像素点开始绘制裂缝图像（一般都是从上到下）
        NUM_Fractures_per_img = config_multi_fractures['cracks_num']                # 多缝中的，裂缝条数
        parallel_cracks = config_multi_fractures['parallel_cracks']                 # 是否生成平行的多缝
        crack_width = config_multi_fractures['crack_width']                         # 裂缝宽度配置参数
        crack_type = config_multi_fractures['crack_value']                          # 裂缝填充类型：高导、高阻
        IMG_background = np.zeros((height_background, width_background), dtype=np.uint8)
        IMG_background.fill(self.VALUE_BACKGROUND)
        IMG_background_target = IMG_background.copy()
        list_configs = []

        # 裂缝配置信息的随机生成
        for i in range(NUM_Fractures_per_img):
            if parallel_cracks:
                # 平行多缝的随机参数设置
                height_background_t = 300
                height_draw_t = np.random.randint(70, 90)
                crack_thickness = int(height_background_t * (np.random.random()+0.5)*0.2)
                noise_level_t = (crack_thickness / 300 + 32 / height_draw_t) * 0.5
                crack_x_shift_t = crack_x_shift * (0.9 + 0.2 * np.random.random())
                crack_start_height_t = crack_start_height + np.random.randint(crack_width, crack_width*2) * i
                config_t = {
                    'width_background':width_background,
                    'height_background':height_background_t,
                    'crack_x_shift': crack_x_shift_t,
                    'noise_level': noise_level_t,
                    'crack_thickness': crack_thickness,
                    'crack_type': crack_type,
                    'height_draw': height_draw_t,                   # 将这个图像缩放到多高来进行粘贴
                    'crack_start_height': crack_start_height_t,     # 将这个图像从哪个高度开始来进行粘贴
                }
                list_configs.append(config_t)
            else:
                # 交叉多缝的随机参数配置 {'crack_x_shift': 0.2, 'noise_level': 0.1, 'crack_width': 30, 'height_draw': 256, 'crack_start_height': 200}
                # 分别控制crack_x_shift：X方向偏移百分比， noise_level裂缝上下限噪声水平， crack_width裂缝高亮色宽度， height_draw裂缝图像生成后压缩的图像高度， crack_start_height裂缝从Y方向哪个像素位置开始的
                # config_crack['width_background']        ：背景的宽度
                # config_crack['height_background']       ：背景的长度
                # config_crack['crack_x_shift']           ：裂缝在X方向上的偏移百分比
                # config_crack['noise_level']             ：噪声水平
                # config_crack['crack_thickness']         ：裂缝的宽度
                # config_crack['crack_value']             ：裂缝是什么类型的裂缝，分别包括高导、高阻、诱导
                if i != NUM_Fractures_per_img - 1:
                    height_background_t = 300
                    height_draw_t = int(256//(2**i) * (0.8 + np.random.random()*0.4))
                    crack_thickness = np.random.randint(30+i*np.random.randint(13, 19), 40+i*np.random.randint(13, 19))
                    noise_level_t = (crack_thickness/300 + 24/height_draw_t)*0.5
                    crack_x_shift_t = crack_x_shift * (0.9+0.2*np.random.random())
                    crack_start_height_t = crack_start_height + np.random.randint(10, 20) - 15
                    config_t = {
                        'width_background':width_background,
                        'height_background':height_background_t,
                        'crack_x_shift': crack_x_shift_t,
                        'noise_level': noise_level_t,
                        'crack_thickness': crack_thickness,
                        'crack_type': crack_type,
                        'height_draw': height_draw_t,
                        'crack_start_height': crack_start_height_t,
                    }
                    list_configs.append(config_t)
                else:
                    # (3缝的最后一个裂缝，可以添加成为水平缝)
                    parallel_fracture_flag = np.random.choice([True, False])
                    if parallel_fracture_flag:
                        # print('生成随机水平第三缝！！！！！！！！')
                        height_background_t = 100
                        height_draw_t = int(np.random.randint(26, 46) * (0.8 + np.random.random() * 0.4))
                        crack_thickness = np.random.randint(int(0.4*height_background_t), int(0.65*height_background_t))
                        noise_level_t = (crack_thickness/300 + 96/height_draw_t)*0.5
                        crack_x_shift_t = crack_x_shift * (0.9+0.2*np.random.random())
                        crack_start_height_t = crack_start_height + np.random.randint(10, 20) - 8
                        config_t = {
                            'width_background':width_background,
                            'height_background':height_background_t,
                            'crack_x_shift': crack_x_shift_t,
                            'noise_level': noise_level_t,
                            'crack_thickness': crack_thickness,
                            'crack_type': crack_type,
                            'height_draw': height_draw_t,
                            'crack_start_height': crack_start_height_t,
                        }
                        config_crack_temp = {
                            'crack_cut_off': np.random.choice([0, 0, 0, 0, 1, 1, 1, 2, 3, 4, 5]),
                            # 裂缝可能不是连续的，裂缝水平方向进行截断的个数，一般是1-2次，
                            'cut_off_width': 0.4 + 0.4 * np.random.random(),  # 水平方向截断的裂缝断裂宽度，0-1，这个
                            'noise_cut_off': 0.03,  # 水平方向截断的裂缝截断面的噪声水平
                        }
                        config_t.update(config_crack_temp)
                        list_configs.append(config_t)
                    else:
                        height_background_t = 300
                        height_draw_t = int(256//(2**i) * (0.8 + np.random.random() * 0.4))
                        crack_thickness = np.random.randint(30+i*np.random.randint(18, 25), 50+i*np.random.randint(18, 25))
                        noise_level_t = (crack_thickness/300 + 32/height_draw_t)*0.5
                        crack_x_shift_t = crack_x_shift * (0.9+0.2*np.random.random())
                        crack_start_height_t = crack_start_height + np.random.randint(10, 20) - 15
                        config_t = {
                            'width_background':width_background,
                            'height_background':height_background_t,
                            'crack_x_shift': crack_x_shift_t,
                            'noise_level': noise_level_t,
                            'crack_thickness': crack_thickness,
                            'crack_type': crack_type,
                            'height_draw': height_draw_t,
                            'crack_start_height': crack_start_height_t,
                        }
                        list_configs.append(config_t)

        for config_temp in list_configs:
            print(config_temp)
            crack_random, crack_target = fms.generate_crack_by_config(crack_config=config_temp)
            crack_start_height = config_temp['crack_start_height']
            height_draw = config_temp['height_draw']
            crack_random = cv2.resize(crack_random, (width_background, height_draw))
            crack_target = cv2.resize(crack_target, (width_background, height_draw))
            IMG_background[crack_start_height:crack_start_height+height_draw] += crack_random
            IMG_background_target[crack_start_height:crack_start_height+height_draw] += crack_target

        index_start, index_end = get_pic_top_and_bottle_index(IMG_background)
        return IMG_background[index_start:index_end, :], IMG_background_target[index_start:index_end, :]


if __name__ == '__main__':
    fms = cracks_mask_simulator()

    masks_crack_list = []
    for i in range(8):
        crack_random, crack_target = fms.generate_random_multi_cracks()
        masks_crack_list.append(crack_random)
        masks_crack_list.append(crack_target)

    show_Pic(masks_crack_list, pic_order='44', figure=[10, 12])

