import random
from distutils.command.config import config
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from src_ele.pic_opeeration import show_Pic
from src_plot.plot_logging import visualize_well_logs


# 裂缝的随机断裂
def add_random_crack_bresk(img, break_position=None):
    # 处理断裂位置参数
    if break_position is None:
        # 完全随机位置：在裂缝存在的区域内随机选择
        crack_pixels = np.where(img > 0)
        if len(crack_pixels[1]) > 0:
            # 从裂缝存在的X位置中随机选择
            break_position = random.choice(crack_pixels[1])
        else:
            # 如果没有裂缝，使用图像中心
            break_position = img.shape[1] // 2
    # 如果提供的是比例值
    elif isinstance(break_position, float):
        if not (0.0 <= break_position <= 1.0):
            break_position = random.random()  # 无效时使用随机值
        break_position = int(break_position * img.shape[1])
    # 如果提供的是像素位置
    elif isinstance(break_position, int):
        if not (0 <= break_position < img.shape[1]):
            # 无效时在有效范围内随机选择
            break_position = random.randint(0, img.shape[1] - 1)
    # 其他类型默认为随机
    else:
        break_position = random.randint(0, img.shape[1] - 1)

    # 在这里写一个裂缝随机的断裂函数，break_position为X方向裂缝断裂位置，现在你需要再生成一个裂缝break_position的对应的斜率，斜率应该是在[1,+∞]及[-∞，-1]之间，
    # 随机设置裂缝断裂宽度在（30，50）像素之间，要求是裂缝断裂部分，需要重新填充为0，表明裂缝在这个部分产生了断裂
    # 然后返回的结果是，裂缝断裂后的图像以及裂缝断裂区域与裂缝重合区域面积大小占裂缝面积的比例

    # 计算原始裂缝面积
    total_crack_area = np.sum(img > 0)
    # 如果图像中没有裂缝，直接返回
    if total_crack_area == 0:
        return img, 0.0

    # 随机生成断裂参数
    slope_sign = random.choice([-1, 1])  # 斜率方向
    slope = slope_sign * random.uniform(1.0, 25.0)  # 斜率绝对值在[1,25]之间
    break_width = random.randint(60, 100)  # 断裂宽度

    # 创建断裂区域的掩码
    break_mask = np.zeros_like(img, dtype=bool)
    # 计算断裂区域，X方向中心位置
    start_x = max(0, break_position - break_width // 2)
    end_x = min(img.shape[1], break_position + break_width // 2)

    # 为每个x位置计算断裂带的高度范围
    for x in range(start_x, end_x):
        # 计算断裂带中心线在y方向的位置
        center_y = img.shape[0] // 2 + slope * (x - break_position)

        # 计算断裂带的高度范围
        y_min = max(0, int((center_y - break_width // 2) * (0.9+0.2*random.random())))
        y_max = min(img.shape[0], int((center_y + break_width // 2) * (0.9+0.2*random.random())))

        # 标记断裂区域
        if y_min < y_max:
            break_mask[y_min:y_max, x] = True

    # 计算断裂区域与裂缝重合的面积
    break_overlap_area = np.sum((img > 0) & break_mask)

    # 创建断裂后的图像
    img_broken = img.copy()
    img_broken[break_mask] = 0  # 将断裂区域设为背景色

    # 计算断裂比例
    break_ratio = break_overlap_area / total_crack_area if total_crack_area > 0 else 0.0

    return img_broken, break_ratio

def generate_single_period_sine_crack(config):
    """
    生成基于单个周期正弦曲线的裂缝图像
    参数:
    width: 图像宽度
    height: 图像高度
    crack_x_shift: 裂缝X方向旋转像素个数，亦或者是百分比
    noise_level: 噪声强度 (0-1)
    smooth_sigma: 平滑系数
    crack_width: 裂缝宽度 (像素)
    crack_break_ratio: 裂缝产生随机break（裂缝中间断裂）的概率

    返回:
    裂缝掩码图像 (二值或灰度)
    """
    width, height, crack_x_shift, noise_level, crack_width, crack_break_ratio = config['width_background'], config['height_background'], config['crack_x_shift'], config['noise_level'], config['crack_width'], config['crack_break_ratio']
    # 1. 创建空白图像
    img = np.zeros((height, width), dtype=np.uint8)
    # crack_height: 裂缝中心线高度
    crack_height = (height)//2
    # crack_amplitude: 裂缝振幅 (控制波动幅度)
    crack_amplitude = int((height - crack_width)//2 * 0.9)

    # 2. 沿X方向进行曲线旋转偏移的设定
    if isinstance(crack_x_shift, float):
        x_shift = crack_x_shift * 2 * np.pi
    elif isinstance(crack_x_shift, int):
        x_shift = crack_x_shift/width * 2 * np.pi
    else:
        x_shift = 0

    # 3. 生成单个周期的正弦曲线数据
    # 生成一个完整周期的正弦数据点 (长度为width)
    x_range = np.linspace(0 + x_shift, 2 * np.pi + x_shift, width)  # 0到2π的一个完整周期
    y_signal = crack_amplitude * np.sin(x_range)

    # 4. 对正弦曲线添加噪声
    y_signal1 = np.random.normal(0, noise_level, (y_signal.shape[0],)) * crack_amplitude + y_signal
    y_signal2 = np.random.normal(0, noise_level, (y_signal.shape[0],)) * crack_amplitude + y_signal
    # y_signal = np.random.normal(1-noise_level, 1+noise_level, (y_signal.shape[0])) * y_signal

    # 5. 对噪声正弦曲线进行平滑降噪
    # # 卷积平滑滤波
    # window_size = 7
    # window = np.ones(window_size) / window_size
    # y_signal = np.convolve(y_signal, window, mode='same')
    # SG滤波
    window_size = int(0.1*width)
    polyorder = 3           # polyorder为多项式拟合的阶数,它越小，则平滑效果越明显；越大，则更贴近原始曲线。
    y_signal1 = savgol_filter(y_signal1, window_size, polyorder)
    y_signal2 = savgol_filter(y_signal2, window_size, polyorder)

    # 3. 确保所有点都在图像范围内
    y_signal1 = np.clip(y_signal1, -crack_height-crack_width, crack_height+crack_width)
    y_signal2 = np.clip(y_signal2, -crack_height-crack_width, crack_height+crack_width)

    crack_up = y_signal1 + crack_height + crack_width//2
    crack_down = y_signal2 + crack_height - crack_width//2


    # 4. 创建裂缝区域
    # 创建行索引矩阵
    rows = np.arange(height)[:, np.newaxis]
    # 创建布尔掩码
    mask = (rows >= crack_down) & (rows < crack_up)
    # 一次性填充所有像素
    img[mask] = 255
    # # 4. 旧的创建裂缝区域
    # for i in range(width):
    #     # 计算裂缝的上下边界
    #     y_min = int(max(0, crack_down[i]))
    #     y_max = int(min(height, crack_up[i]))
    #     # print(y_min,y_max)
    #
    #     # 在裂缝区域填充
    #     if y_min < y_max:
    #         img[y_min:y_max, i] = 255

    # 裂缝产生随机的断裂
    if random.random() < crack_break_ratio:
        # # 裂缝的随机断裂，产生裂缝在水平方向的随机断裂
        img_final, break_ratio = add_random_crack_bresk(img.copy())
        if random.random() < crack_break_ratio:
            img_final, break_ratio = add_random_crack_bresk(img.copy())
    else:
        img_final = img

    return img_final


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


class cracks_simulation(object):
    # 初始化类
    def __init__(self, ):

        pass

    # 新生成一个裂缝图像，并返回
    def genrate_new_crack(self, config_crack={}):
        """
        生成新的裂缝图像，函数测试用的，后面基本没啥用
        参数:
        config_crack: 配置字典，包含以下可选参数:
            - width: 图像宽度
            - height: 图像高度
            - crack_x_shift: 裂缝X方向旋转像素个数，亦或者是百分比
            - noise_level: 噪声强度 (0-1)
            - smooth_sigma: 平滑系数
            - crack_width: 裂缝宽度 (像素)
        返回:
        生成的裂缝图像
        """
        # 1. 设置默认参数
        default_config = {
            'width_background': 256,
            'height_background': 256,
            'crack_x_shift': 0.0,
            'noise_level': 0.05,
            'crack_width': 20,
            'crack_break_ratio': 0.5,
        }
        # 2. 合并配置参数
        config_crack = {**default_config, **config_crack}

        # 3. 验证参数
        if not isinstance(config_crack['width_background'], int) or config_crack['width_background'] <= 0:
            raise ValueError("width必须是正整数")
        if not isinstance(config_crack['height_background'], int) or config_crack['height_background'] <= 0:
            raise ValueError("height必须是正整数")
        if not 0.0 <= config_crack['noise_level'] <= 1.0:
            raise ValueError("noise_level必须在[0.0, 1.0]范围内")
        if not isinstance(config_crack['crack_width'], int) or config_crack['crack_width'] <= 0:
            raise ValueError("crack_width必须为整数，代表了裂缝宽度的像素个数")

        # 4. 根据参数生成新的裂缝
        fracture_t = generate_single_period_sine_crack(config_crack)

        return fracture_t

    # 产生随机的单条裂缝图像，当然也可以通过配置参数控制器中某些变量，并返回裂缝模拟配置参数以及裂缝的参数曲线
    def genrate_random_single_crack(self, config_crack={}):
        # 1. 设置随机参数
        default_config = {
            'width_background': 256,
            'height_background': 256,
            'crack_x_shift': random.random(),
            'crack_width': random.randint(20, 50),
            'crack_break_ratio':0.6,
        }
        default_config['noise_level'] = default_config['crack_width']/200
        # 2. 合并配置参数
        config_crack = {**default_config, **config_crack}

        # 3. 根据参数，生成新的裂缝
        fracture_t = generate_single_period_sine_crack(config_crack)

        # 4. 根据生成的裂缝，生产一打响应的裂缝参数曲线，其中的深度列换位像素index信息
        depth = np.arange(0, fracture_t.shape[0])
        # 创建测井曲线数据，并进行合适的初始化
        crack_length = np.zeros(len(depth), dtype=np.float32)
        crack_width = np.zeros(len(depth), dtype=np.float32)
        crack_width += config_crack['crack_width']
        crack_area = np.zeros(len(depth), dtype=np.float32)
        crack_angle = np.zeros(len(depth), dtype=np.float32)
        crack_angle[len(depth)//2] = config_crack['crack_break_ratio']
        crack_inclination = np.zeros(len(depth), dtype=np.float32)
        crack_inclination[len(depth)//2] = config_crack['height_background']
        crack_density = np.ones(len(depth), dtype=np.float32)
        for i in range(0, len(depth)):
            crack_area[i] = np.sum(fracture_t[i, :])/255
            crack_length[i] = crack_area[i]/crack_width[i]

        dp_crack_para = pd.DataFrame({
            'depth': depth,
            'crack_length': crack_length,
            'crack_width': crack_width,
            'crack_area': crack_area,
            'crack_angle': crack_angle,
            'crack_inclination': crack_inclination,
            'crack_density': crack_density
        })
        return fracture_t, config_crack, dp_crack_para

    # 在给定的图像上，给定指定的裂缝参数，新增多个裂缝
    def add_cracks_on_pic(self, IMG=np.zeros((4, 4), dtype=np.uint8), list_configs=[], show_process=False):
        """
        给定一张地层图片IMG，以及默认配置列表list_configs，进行裂缝mask模拟，
        返回的是模拟后的裂缝地层图像mask以及裂缝对应的地层参数曲线
        """

        # 图像必须是array-2D格式的
        if not isinstance(IMG, np.ndarray):
            print('IMG is not np.ndarray')
            exit(0)
        # 画布不能太小
        if IMG.size <= 160:
            print(f'IMG is too small:{IMG.shape}')
            exit(0)
        # 裂缝的 list_configs 必须是 list 格式的，里面还不能为空，必须放的有dict格式的裂缝参数配置
        if not isinstance(list_configs, list):
            print('list_configs is not list')
            exit(0)
        if len(list_configs) == 0:
            print('list_configs is empty, now add fracture configs as default')
            list_configs = [
                {'crack_x_shift': 0.20, 'noise_level': 0.10, 'crack_width': 30, 'height_draw': 256, 'crack_start_height': 200},
                {'crack_x_shift': 0.18, 'noise_level': 0.15, 'crack_width': 40, 'height_draw': 128, 'crack_start_height': 200},
                {'crack_x_shift': 0.21, 'noise_level': 0.20, 'crack_width': 60, 'height_draw': 64 , 'crack_start_height': 200}]
            # height_draw 参数用来进行多缝模拟时， 控制每单缝Y方向的像素个数
            # crack_start_height 参数用来进行多缝模拟时， 控制单缝在Y方向 开始像素起始位置 （从上往下数）

        # 4. 根据生成的裂缝，生产一打相应的裂缝参数曲线
        depth = np.arange(0, IMG.shape[0])
        # 创建测井曲线数据，并进行合适的初始化
        crack_length = np.zeros(len(depth), dtype=np.float64)
        crack_width = np.zeros(len(depth), dtype=np.float64)
        crack_area = np.zeros(len(depth), dtype=np.float64)
        crack_angle = np.zeros(len(depth), dtype=np.float64)
        crack_inclination = np.zeros(len(depth), dtype=np.float64)
        crack_density = np.zeros(len(depth), dtype=np.float64)
        df_cracks_para = pd.DataFrame({
            'depth': depth,
            'crack_length': crack_length,
            'crack_width': crack_width,
            'crack_area': crack_area,
            'crack_angle': crack_angle,
            'crack_inclination': crack_inclination,
            'crack_density': crack_density
        }).astype(np.float64)
        list_paras = ['crack_length', 'crack_width', 'crack_area', 'crack_angle', 'crack_inclination', 'crack_density']

        for config in list_configs:
            if show_process:
                IMG_DRAW_RAW = IMG.copy()
            height_draw = config['height_draw']
            crack_start_height = config['crack_start_height']
            if height_draw > IMG.shape[0]:
                print(f'fracture height ({height_draw}) is longer than background image shape({IMG.shape})')
                exit(0)
            if height_draw + crack_start_height > IMG.shape[0]:
                print(f'error as : height_draw ({height_draw}) + crack_start_height ({crack_start_height}) > IMG.shape[0] ({IMG.shape[0]})')
                exit(0)
            fracture_t, fracture_config, fracture_parameter = self.genrate_random_single_crack(config)
            fracture_t_reshape = cv2.resize(fracture_t, [IMG.shape[1], height_draw], interpolation=cv2.INTER_CUBIC)
            # visualize_well_logs(
            #     data=fracture_parameter,
            #     depth_col='depth',
            #     curve_cols=list_paras,
            # )
            fracture_config, fracture_parameter = self.adjust_crack_para_by_target_window_height(fracture_config, fracture_parameter, fracture_t_reshape.shape)

            # 裂缝图像信息，合并到背景图像上
            IMG[crack_start_height:height_draw+crack_start_height, :] += fracture_t_reshape
            # 裂缝参数dataframe的合并，将裂缝的参数进行累加合并到地层总的裂缝参数曲线上
            df_cracks_para.loc[crack_start_height:height_draw+crack_start_height-1, list_paras] += fracture_parameter[list_paras].values
            # visualize_well_logs(
            #     data=df_cracks_para,
            #     depth_col='depth',
            #     curve_cols=list_paras,
            # )

            if show_process:
                show_Pic([IMG_DRAW_RAW, IMG], pic_order='12', figure=(8, 4))
        return IMG, df_cracks_para

    # 可视化不同参数的单缝效果，测试用的，后续基本没啥用
    def visulize_random_fracture(self):
        f1 = self.genrate_new_crack()
        f2 = self.genrate_new_crack(config_crack={'crack_x_shift': 0.3})
        f3 = self.genrate_new_crack(config_crack={'noise_level': 0.1})
        f4 = self.genrate_new_crack(config_crack={'crack_width': 40, 'noise_level':0.15})

        f5, cf5, cp5 = self.genrate_random_single_crack(config_crack={'crack_x_shift':0.5})
        f6, cf6, cp6 = self.genrate_random_single_crack(config_crack={'crack_x_shift':0.5})

        show_Pic([f1, f2, f3, f4, f5, f6], pic_order='23', pic_str=[], path_save='', title='', figure=(12, 9), show=True)

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
            'crack_start_height': 100,
            'cracks_num': random.randint(2, 3),
            'parallel_cracks': random.choice([True, False, False]),
            'crack_width': random.randint(60, 90),                  # 只是为了配置参数的统一，但是没什么实际用处，起不到什么作用
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
        IMG_background = np.zeros((height_background, width_background), dtype=np.uint8)
        list_configs = []

        if parallel_cracks:
            print('生成随机平行多缝地层')
        else:
            print('生成随机交叉多缝地层')

        # 裂缝配置信息的随机生成
        for i in range(NUM_Fractures_per_img):
            if parallel_cracks:
                # 平行多缝的随机参数设置
                height_draw_t = random.randint(70, 90)
                crack_width_t = random.randint(60, height_draw_t-5)
                noise_level_t = (crack_width_t / 300 + 32 / height_draw_t) * 0.5
                crack_x_shift_t = crack_x_shift * (0.9 + 0.2 * random.random())
                crack_start_height_t = crack_start_height + random.randint(70, 80) * i
                config_t = {'crack_x_shift': crack_x_shift_t, 'noise_level': noise_level_t,
                            'crack_width': crack_width_t, 'height_draw': height_draw_t,
                            'crack_start_height': crack_start_height_t}
                list_configs.append(config_t)
            else:
                # 交叉多缝的随机参数配置 {'crack_x_shift': 0.2, 'noise_level': 0.1, 'crack_width': 30, 'height_draw': 256, 'crack_start_height': 200}
                # 分别控制crack_x_shift：X方向偏移百分比， noise_level裂缝上下限噪声水平， crack_width裂缝高亮色宽度， height_draw裂缝图像生成后压缩的图像高度， crack_start_height裂缝从Y方向哪个像素位置开始的
                if i != NUM_Fractures_per_img - 1:
                    height_draw_t = int(256//(2**i) * (0.8 + random.random() * 0.4))
                    crack_width_t = random.randint(30+i*random.randint(13, 19), 40+i*random.randint(13, 19))
                    # noise_level_t = 0.1 + 0.03 * random.randint(1, 2) * (i + 1)
                    noise_level_t = (crack_width_t/300 + 24/height_draw_t)*0.5
                    crack_x_shift_t = crack_x_shift * (0.9+0.2*random.random())
                    crack_start_height_t = crack_start_height + random.randint(10, 20) - 15
                    config_t = {'crack_x_shift': crack_x_shift_t, 'noise_level': noise_level_t,
                                'crack_width': crack_width_t, 'height_draw': height_draw_t,
                                'crack_start_height': crack_start_height_t}
                    list_configs.append(config_t)
                else:
                    # (3缝的最后一个裂缝，可以添加成为水平缝)
                    parallel_fracture_flag = random.choice([True, False])
                    if parallel_fracture_flag:
                        print('生成随机水平第三缝！！！！！！！！')
                        height_draw_t = int(random.randint(26, 46) * (0.8 + random.random() * 0.4))
                        crack_width_t = random.randint(160, 190)
                        noise_level_t = (crack_width_t/300 + 96/height_draw_t)*0.5
                        crack_x_shift_t = crack_x_shift * (0.9+0.2*random.random())
                        crack_start_height_t = crack_start_height + random.randint(10, 20) - 8
                        config_t = {'crack_x_shift': crack_x_shift_t, 'noise_level': noise_level_t,
                                    'crack_width': crack_width_t, 'height_draw': height_draw_t,
                                    'crack_start_height': crack_start_height_t}
                        list_configs.append(config_t)
                    else:
                        height_draw_t = int(256//(2**i) * (0.8 + random.random() * 0.4))
                        crack_width_t = random.randint(30+i*random.randint(18, 25), 50+i*random.randint(18, 25))
                        noise_level_t = (crack_width_t/300 + 32/height_draw_t)*0.5
                        crack_x_shift_t = crack_x_shift * (0.9+0.2*random.random())
                        crack_start_height_t = crack_start_height + random.randint(10, 20) - 15
                        config_t = {'crack_x_shift': crack_x_shift_t, 'noise_level': noise_level_t,
                                    'crack_width': crack_width_t, 'height_draw': height_draw_t,
                                    'crack_start_height': crack_start_height_t}
                        list_configs.append(config_t)

        IMG, df_cracks_para = self.add_cracks_on_pic(IMG=IMG_background, list_configs=list_configs, show_process=False)
        # 二值化，将所有裂缝区域进行 最大值化
        _, IMG = cv2.threshold(IMG, 5, 255, cv2.THRESH_BINARY)     # cv2.THRESH_BINARY：当像素值大于阈值时，赋予最大值；否则为0。

        index_start, index_end = get_pic_top_and_bottle_index(IMG)
        return IMG[index_start:index_end, :], config_multi_fractures, df_cracks_para.iloc[index_start:index_end, :]

    def adjust_crack_para_by_target_window_height(self, fracture_config, fracture_parameter, shape_target):
        """
        根据新的窗长，调整裂缝配置参数fracture_config以及裂缝相应参数fracture_parameter
        调整裂缝参数响应，用来进行单挑裂缝参数的合并，组成多缝裂缝参数
        """
        # 根据shape_target重新调整fracture_config={'width_background': 256, 'height_background': 512, 'crack_x_shift': 0.70, 'crack_width': 27, 'crack_break_ratio': 0.6, 'noise_level': 0.135}
        # 计算裂缝深度（裂缝跨度或裂缝的高）缩放因子
        scale_ratio_config = shape_target[0]/fracture_config['height_background']
        fracture_config['height_background'] = shape_target[0]                                          # 调整裂缝高度，裂缝在图板上的展开高度
        fracture_config['crack_width'] = fracture_config['crack_width'] * scale_ratio_config       # 调整裂缝张开度，也就是裂缝的宽
        # print(f'{fracture_config}', scale_ratio)

        scale_ratio_para = shape_target[0]/fracture_parameter.shape[0]
        # depth_temp = np.linspace(start=0, stop=shape_target[0]-1, num=shape_target[0])
        # data_temp = np.zeros((shape_target[0], fracture_parameter.shape[1]))
        # fracture_parameter_n = pd.DataFrame(data_temp, columns=fracture_parameter.columns)
        # fracture_parameter_n['depth'] = depth_temp

        # 根据shape_target重新调整 fracture_parameter=pd.Dataframe(Index(['depth', 'crack_length', 'crack_width', 'crack_area', 'crack_angle', 'crack_inclination', 'crack_density'])
        fracture_parameter_n = self.resample_logging_data(fracture_parameter, new_size=shape_target[0])
        fracture_parameter_n['crack_area'] = fracture_parameter_n['crack_area']*scale_ratio_para         # 调整裂缝面积曲线
        fracture_parameter_n['crack_width'] = fracture_parameter_n['crack_width']*scale_ratio_para       # 调整裂缝宽度曲线
        fracture_parameter_n['crack_angle'] = 0.0
        fracture_parameter_n['crack_inclination'] = 0.0
        col_idx_angle_n = fracture_parameter_n.columns.get_loc('crack_angle')
        col_idx_inclination_n = fracture_parameter_n.columns.get_loc('crack_inclination')
        col_idx_angle = fracture_parameter.columns.get_loc('crack_angle')
        col_idx_inclination = fracture_parameter.columns.get_loc('crack_inclination')
        cols_cracks = ['crack_length', 'crack_width', 'crack_area', 'crack_angle', 'crack_inclination', 'crack_density']
        for i in range(fracture_parameter.shape[0]):
            if fracture_parameter.iloc[i, col_idx_angle] > 0:
                fracture_parameter_n.iloc[int(i*scale_ratio_para), col_idx_angle_n] = fracture_parameter.iloc[i, col_idx_angle]               # 调整裂缝角度曲线
            if fracture_parameter.iloc[i, col_idx_inclination] > 0:
                fracture_parameter_n.iloc[int(i*scale_ratio_para), col_idx_inclination_n] = fracture_parameter.iloc[i, col_idx_inclination]   # 调整裂缝角度曲线
            # fracture_parameter_n.loc[int(i*scale_ratio_para), cols_cracks] = fracture_parameter.loc[i, cols_cracks]
        fracture_parameter_n = fracture_parameter_n.astype(np.float64)                                          # 调整曲线的数值类型
        return fracture_config, fracture_parameter_n

    def resample_logging_data(self, df, new_size=512):
        """
        dataframe数据的重采样
        数据新的长度为new_size
        默认深度列为depth，且默认深度从0开始，且深度的分辨率为1
        """
        # 1. 获取所有非depth列
        non_depth_cols = [col for col in df.columns if col != 'depth']
        # 2. 创建新的索引序列（0到new_size-1）
        new_index = np.linspace(0, len(df) - 1, new_size)

        # 3. 对每条非depth曲线进行重采样
        resampled_data = {}
        for col in non_depth_cols:
            # 获取原始值
            original_values = df[col].values

            # 使用线性插值
            resampled_values = np.interp(new_index, np.arange(len(original_values)), original_values)
            resampled_data[col] = resampled_values

        # 4. 创建新的DataFrame
        resampled_df = pd.DataFrame(resampled_data)

        # 5. 添加depth列并初始化为0
        resampled_df['depth'] = np.linspace(0, new_size-1, new_size)

        # 6. 确保depth列是第一列
        columns = ['depth'] + non_depth_cols
        resampled_df = resampled_df[columns]

        return resampled_df



if __name__ == '__main__':
    SFS = cracks_simulation()
    # SFS.visulize_random_fracture()

    BASE_BACKGROUND = np.zeros((512, 256), dtype=np.uint8)
    # SFS.add_fractures_on_pic(BASE_BACKGROUND, list_configs=[], show_process=True)
    # IMG = SFS.generate_random_multi_fractures(config_multi_fractures={'crack_x_shift': 0.3})
    # show_Pic([IMG], pic_order='11')

    LIST_IMG_MULTI = []
    for i in range(4):
        IMG_T, config_cracks, df_cracks_para = SFS.generate_random_multi_cracks(config_multi_fractures={'crack_x_shift': 0.1 * i})
        LIST_IMG_MULTI.append(IMG_T)
        print(IMG_T.shape, df_cracks_para.shape)
        visualize_well_logs(
            data=df_cracks_para,
            depth_col='depth',
            curve_cols=['crack_length', 'crack_width', 'crack_area', 'crack_angle', 'crack_inclination', 'crack_density'],
        )
    show_Pic(LIST_IMG_MULTI, figure=(9, 16))

    # LIST_IMG_SINGLE = []
    # for i in range(9):
    #     IMG_T, CF_T, CP_T = SFS.genrate_random_single_crack(config_crack={'crack_x_shift':0.1*i})
    #     LIST_IMG_SINGLE.append(IMG_T)
    #     print(CF_T, IMG_T.shape, CP_T.columns)
    #
    #     visualize_well_logs(
    #         data=CP_T,
    #         depth_col='depth',
    #         curve_cols=['crack_length', 'crack_width', 'crack_area', 'crack_angle', 'crack_inclination'],
    #         type_cols=[],
    #         legend_dict={0: 'Type0', 1: 'Type1', 2: 'Type2', 3: 'Type3'}
    #     )
    # show_Pic(LIST_IMG_SINGLE, pic_order='33', figure=(9, 9))