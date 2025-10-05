import random
from distutils.command.config import config
import cv2
import numpy as np
from scipy.signal import savgol_filter
from src_ele.pic_opeeration import show_Pic


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

    返回:
    裂缝掩码图像 (二值或灰度)
    """
    width, height, crack_x_shift, noise_level, crack_width, crack_break_ratio = config['width'], config['height'], config['crack_x_shift'], config['noise_level'], config['crack_width'], config['crack_break_ratio']
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




class cracks_simulation(object):
    # 初始化类
    def __init__(self, ):

        pass

    # 新生成一个裂缝图像，并返回
    def genrate_new_crack(self, config_crack={}):
        """
        生成新的裂缝图像
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
            'width': 256,
            'height': 256,
            'crack_x_shift': 0.0,
            'noise_level': 0.05,
            'crack_width': 20,
            'crack_break_ratio': 0.5,
        }
        # 2. 合并配置参数
        config_crack = {**default_config, **config_crack}

        # 3. 验证参数
        if not isinstance(config_crack['width'], int) or config_crack['width'] <= 0:
            raise ValueError("width必须是正整数")
        if not isinstance(config_crack['height'], int) or config_crack['height'] <= 0:
            raise ValueError("height必须是正整数")
        if not 0.0 <= config_crack['noise_level'] <= 1.0:
            raise ValueError("noise_level必须在[0.0, 1.0]范围内")
        if not isinstance(config_crack['crack_width'], int) or config_crack['crack_width'] <= 0:
            raise ValueError("crack_width必须为整数，代表了裂缝宽度的像素个数")

        # 4. 根据参数生成新的裂缝
        fracture_t = generate_single_period_sine_crack(config_crack)

        return fracture_t

    # 产生随机的单条裂缝图像，当然也可以通过配置参数控制器中某些变量
    def genrate_random_single_crack(self, config_crack={}):
        # 1. 设置随机参数
        default_config = {
            'width': 256,
            'height': 256,
            'crack_x_shift': random.random(),
            'crack_width': random.randint(20, 50),
            'crack_break_ratio':0.6,
        }
        default_config['noise_level'] = default_config['crack_width']/200
        # 2. 合并配置参数
        config_crack = {**default_config, **config_crack}

        # 3. 根据参数，生成新的裂缝
        fracture_t = generate_single_period_sine_crack(config_crack)

        return fracture_t

    # 在给定的图像上，给定指定的裂缝参数，新增一个裂缝
    def add_cracks_on_pic(self, IMG=np.zeros((4, 4), dtype=np.uint8), list_configs=[], show_process=False):
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
            fracture_t = self.genrate_random_single_crack(config)
            fracture_t_reshape = cv2.resize(fracture_t, [IMG.shape[1], height_draw], interpolation=cv2.INTER_CUBIC)

            IMG[crack_start_height:height_draw + crack_start_height, :] += fracture_t_reshape
            if show_process:
                show_Pic([IMG_DRAW_RAW, IMG], pic_order='12', figure=(8, 4))

        return IMG

    # 可视化不同参数的单缝效果
    def visulize_random_fracture(self):
        f1 = self.genrate_new_crack()
        f2 = self.genrate_new_crack(config_crack={'crack_x_shift': 0.3})
        f3 = self.genrate_new_crack(config_crack={'noise_level': 0.1})
        f4 = self.genrate_new_crack(config_crack={'crack_width': 40, 'noise_level':0.15})

        f5 = self.genrate_random_single_crack(config_crack={'crack_x_shift':0.5})
        f6 = self.genrate_random_single_crack(config_crack={'crack_x_shift':0.5})

        show_Pic([f1, f2, f3, f4, f5, f6], pic_order='23', pic_str=[], path_save='', title='', figure=(12, 9), show=True)

    def generate_random_multi_cracks(self, config_multi_fractures={}):
        config_multi_fractures_default = {
            'crack_x_shift': 0.6,
            'crack_start_height': 100,
            'height_background':512,
            'width_background':256,
            'cracks_num':random.randint(2,3),
            'rotate_y':random.choice([True, False]),
            'parallel_fractures':random.choice([True, False])
        }

        config_multi_fractures = {**config_multi_fractures_default, **config_multi_fractures}

        height_background = config_multi_fractures['height_background']             # 背景高度
        width_background = config_multi_fractures['width_background']               # 背景宽度
        crack_x_shift = config_multi_fractures['crack_x_shift']                     # x方向偏移
        crack_start_height = config_multi_fractures['crack_start_height']           # y方向，从哪个像素点开始绘制裂缝图像（一般都是从上到下）
        NUM_Fractures_per_img = config_multi_fractures['cracks_num']                # 多缝中的，裂缝条数
        rotate_y = config_multi_fractures['rotate_y']                               # 是否在结束生成后翻转Y轴？
        parallel_fractures = config_multi_fractures['parallel_fractures']           # 是否生成平行的多缝

        IMG_background = np.zeros((height_background, width_background), dtype=np.uint8)
        list_configs = []

        if parallel_fractures:
            print('生成随机平行多缝地层')
        else:
            print('生成随机交叉多缝地层')

        # 裂缝配置信息的随机生成
        for i in range(NUM_Fractures_per_img):
            if parallel_fractures:
                # 平行多缝的随机参数设置
                height_draw_t = random.randint(70, 90)
                crack_width_t = random.randint(60, height_draw_t-5)
                # noise_level_t = 0.1 + 0.03 * random.randint(1, 2) * (i + 1)
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
                        crack_width_t = random.randint(180, 210)
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

        IMG = self.add_cracks_on_pic(IMG=IMG_background, list_configs=list_configs, show_process=False)
        # 二值化，将所有裂缝区域进行 最大值化
        _, IMG = cv2.threshold(IMG, 5, 255, cv2.THRESH_BINARY)     # cv2.THRESH_BINARY：当像素值大于阈值时，赋予最大值；否则为0。

        # 多缝的截图程序，把上下部分，没什么用的黑色背景板截图丢掉
        SUM_front = []
        SUN_behind = []
        for i in range(IMG.shape[0]):
            SUM_front.append(np.sum(IMG[:i+1, :]))
            SUN_behind.append(np.sum(IMG[i:, :]))
        index_start = np.max(np.where(np.array(SUM_front) <= 1))
        index_end = np.min(np.where(np.array(SUN_behind) <= 1))
        # print(index_start, index_end)

        return IMG[index_start:index_end, :]

if __name__ == '__main__':
    SFS = cracks_simulation()
    # SFS.visulize_random_fracture()

    BASE_BACKGROUND = np.zeros((512, 256), dtype=np.uint8)
    # SFS.add_fractures_on_pic(BASE_BACKGROUND, list_configs=[], show_process=True)
    # IMG = SFS.generate_random_multi_fractures(config_multi_fractures={'crack_x_shift': 0.3})
    # show_Pic([IMG], pic_order='11')

    LIST_IMG_MULTI = []
    for i in range(9):
        IMG_T = SFS.generate_random_multi_cracks(config_multi_fractures={'crack_x_shift': 0.1 * i})
        LIST_IMG_MULTI.append(IMG_T)
    show_Pic(LIST_IMG_MULTI, pic_order='33', figure=(9, 16))

    LIST_IMG_SINGLE = []
    for i in range(9):
        IMG_T = SFS.genrate_random_single_crack(config_crack={'crack_x_shift':0.1*i})
        LIST_IMG_SINGLE.append(IMG_T)
    show_Pic(LIST_IMG_SINGLE, pic_order='33', figure=(9, 9))