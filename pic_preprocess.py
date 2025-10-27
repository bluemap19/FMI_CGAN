import copy
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

from src_ele.file_operation import get_test_ele_data
from src_ele.pic_opeeration import show_Pic, pic_enhence_random, pic_scale_normal
from src_ele.pic_spilt_ostu import OtsuFastMultithreshold
from skimage.util import random_noise

def show_curve(curve_list, curve_order='12', pic_str=[], save_pic=False, path_save=''):
    if len(curve_order) != 2:
        print('pic order error:{}'.format(curve_order))

    num = int(curve_order[0]) * int(curve_order[1])

    if num != len(curve_list):
        print('pic order num is not equal to pic_list num:{},{}'.format(len(pic_list), curve_order))

    # print(num)
    while( len(pic_str) < len(curve_list)):
        pic_str.append('pic_str'+str(len(curve_list)-len(pic_str)))
    # print(pic_str)

    for i in range(len(curve_list)):
        for j in range(curve_list[i].shape[0]):
            for k in range(curve_list[i].shape[1]):
                if curve_list[i][j][k] < 0:
                    curve_list[i][j][k] = 0

    plt.close('all')
    fig = plt.figure(figsize=(16, 9))
    for i in range(len(curve_list)):
        order_str = int(curve_order+str(i+1))
        ax = fig.add_subplot(order_str)
        ax.set_title(pic_str[i])
        plt.axis('off')
        ax.imshow(curve_list[i], cmap='hot')
        # ax.imshow(curve_list[i], cmap='afmhot')
        # ax.imshow(curve_list[i], cmap='gist_heat')
    plt.show()

    if save_pic:
        if path_save == '':
            plt.savefig('temp.png')
        else:
            plt.savefig(path_save)
        plt.close()

# 获取随机的方位角曲线，为了接下来的进行图像绕井壁旋转
# 获得一根随机的RB曲线（绕井壁旋转用的）
def get_random_RB_curve_1(depth, start_angle=np.random.randint(-60, 60)):
    max_rotate_angle = 1
    Rb_random = np.zeros(depth.shape)

    for i in range(depth.shape[0]):
        np.random.seed()
        rotate_angle = (np.random.random()-0.5)*max_rotate_angle
        if i == 0:
            Rb_random[i][0] = start_angle
        else:
            Rb_random[i][0] = max(min(Rb_random[i-1][0] + rotate_angle, 180), -180)

    return Rb_random


def get_random_RB_curve_2(depth, dep_inf=[-1, -1]):

    if dep_inf[0] < 0:
        # 计算 图像能够旋转的 最大值
        ratate_angle = min(int(depth.shape[0] * 0.075) + 2, 80)
        start_angle = np.random.randint(-(180-ratate_angle-30), 180-ratate_angle-30)
        end_angle = start_angle + np.random.randint(-ratate_angle, ratate_angle)
    else:
        start_angle, end_angle = dep_inf
    # np.random.seed()
    # print('ratate_angle:{}, start_angle:{}, end_angle:{}'.format(ratate_angle, start_angle, end_angle))

    Rb_random = np.zeros_like(depth)

    rotate_angle = (end_angle-start_angle)/depth.shape[0]
    # print('rotate_angle + {}'.format(rotate_angle))
    for i in range(depth.shape[0]):
        if i == 0:
            Rb_random[i][0] = start_angle
        else:
            Rb_random[i][0] = max(min(Rb_random[i-1][0] + rotate_angle + np.random.randint(-60, 60)/depth.shape[0], 180), -180)

    return Rb_random

# 根据RB曲线进行图像旋转
def pic_rotate_by_Rb(pic=np.zeros((10,10)), Rb=np.zeros((10, 1))):
    if pic.shape[0] != Rb.shape[0]:
        print('pic length is not equal to depth length:{}, {}'.format(pic.shape, Rb.shape))
        exit(0)

    pic_new = np.zeros(pic.shape)
    # print(pic_new.shape)
    temp = 360/pic.shape[1]
    for i in range(pic.shape[0]):
        pixel_rotate = int(Rb[i][0] / temp)
        # print(pixel_rotate)
        if pixel_rotate != 0:
            pic_new[i, pixel_rotate:] = pic[i, :-pixel_rotate]
            pic_new[i, :pixel_rotate] = pic[i, -pixel_rotate:]
        else:
            pic_new[i, :] = pic[i, :]

    return pic_new


def get_random_RB_all(depth=np.zeros((5, 1)), RB_index=-1, ratio=np.random.random()):
    rb_random = np.array([])

    # RB_index_new = 0
    if RB_index == -1:
        if ratio < 0.3:
            # print('way 1, R:{}'.format(ratio))
            rb_random = get_random_RB_curve_1(depth)
            RB_index = 0
        elif ratio < 0.6:
            # print('way 2, R:{}'.format(ratio))
            rb_random = get_random_RB_curve_2(depth)
            RB_index = 1
        else:
            # print('way 3, R:{}'.format(ratio))
            rb_random = np.zeros((depth.shape[0], 1))
            rb_random.fill(np.random.randint(-120, 120))
            RB_index = 2
        return rb_random, RB_index

    else:
        if RB_index == 0:
            if ratio < 0.5:
                rb_random = get_random_RB_curve_2(depth)
                RB_index = 1
            else:
                rb_random = np.zeros((depth.shape[0], 1))
                rb_random.fill(np.random.randint(-120, 120))
                RB_index = 2

        elif RB_index == 1:
            if ratio < 0.5:
                rb_random = get_random_RB_curve_1(depth)
                RB_index = 0
            else:
                rb_random = np.zeros((depth.shape[0], 1))
                rb_random.fill(np.random.randint(-120, 120))
                RB_index = 2

        elif RB_index == 2:
            if ratio < 0.5:
                rb_random = get_random_RB_curve_1(depth)
                RB_index = 0
            else:
                rb_random = get_random_RB_curve_2(depth)
                RB_index = 1
        else:
            print('RB_index error:{}'.format(RB_index))
            exit(0)

        return rb_random, RB_index


# 图像 生成随机RB曲线 并旋转
def pic_rotate_random(pic=np.zeros((5, 5)), depth=np.zeros((5, 1)), ratio=np.random.random()):
    # print('depth shape is :{}'.format(depth.shape))
    if ratio < 0.4:
        # print('way 1, R:{}'.format(ratio))
        rb_random = get_random_RB_curve_1(depth)
    elif ratio < 0.95:
        # print('way 2, R:{}'.format(ratio))
        rb_random = get_random_RB_curve_2(depth)
    else:
        # print('way 3, R:{}'.format(ratio))
        rb_random = np.zeros((pic.shape[0], 1))
        return pic, rb_random
    pic_new = pic_rotate_by_Rb(pic, rb_random)
    return pic_new, rb_random


# 获取图像的 像素分布
def image_hist(img, static_length=64):
    # 处理彩色图像：转换为灰度
    if len(img.shape) == 3:
        # 使用加权平均法转换灰度：0.299*R + 0.587*G + 0.114*B
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    # 获取像素值范围
    min_val = np.min(img)
    max_val = np.max(img)

    # 处理特殊情况：所有像素值相同
    if min_val == max_val:
        x_range = np.array([min_val])
        y_range = np.array([1.0])
        return x_range, y_range

    # 计算分组边界（避免除零错误）
    bin_edges = np.linspace(min_val, max_val, static_length + 1)

    # 计算分组中值点（作为x_range）
    x_range = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 使用numpy的histogram函数高效计算直方图
    hist, _ = np.histogram(img, bins=bin_edges)

    # 计算百分比
    total_pixels = img.size
    y_range = hist / total_pixels

    return x_range, y_range

def pic_dim_shift(pic=np.random.random((5, 5))*255):
    """
    图像色域偏移
    :param pic:
    :return:
    """

    ran_int = np.random.randint(-30, 30)

    if pic.shape.__len__() >= 3:
        print('pic_dim_shift fun dim error:{}'.format(pic.shape))
        exit(0)

    pic = pic + ran_int

    pic = np.clip(pic, 0, 255)
    return pic


# 图像随机去噪处理
def pic_denoise(pic=np.zeros((5, 5)), ratio=np.random.random(), k_size=np.random.randint(1, 2)*2+1):
    """
    图像降噪处理
    :param pic:原始图像
    :param ratio: 降噪方法随即银子
    :param k_size: 核大小，随机生成
    :return: pic_new 降噪后图像
    """
    if ratio < 0.15:
        # 高斯滤波
        pic_new = cv2.GaussianBlur(np.uint8(pic), (k_size, k_size), 0)
    elif ratio < 0.3:
        # 中值滤波
        pic_new = cv2.medianBlur(np.uint8(pic), k_size)
    elif ratio < 0.6:
        # 双边滤波
        pic_new = cv2.bilateralFilter(np.uint8(pic), k_size+2, 75, 75)
    elif ratio < 0.8:
        # 直方图均衡化
        pic_new = cv2.equalizeHist(np.uint8(pic))
    elif ratio < 0.85:
        # 随机偏移图像增强
        pic_new = pic_enhence_random(pic, windows_shape=k_size)
    else:
        pic_new = pic
    return pic_new




def pic_add_noise(pic, ratio=None, noise_type=None, intensity=0.1, seed=None):
    """
    为图像添加多种类型的随机噪声

    参数:
    pic: 原始图像 (numpy数组, 可以是灰度或彩色图像)
    ratio: 随机噪声类型选择因子 (0-1之间), 如果为None则随机生成
    noise_type: 指定噪声类型 (可选: 'gaussian', 'poisson', 'salt_pepper', 'speckle', 'uniform', 'periodic')
    intensity: 噪声强度 (0-1之间), 默认0.1
    seed: 随机种子 (确保可重复性)

    返回:
    pic_new: 添加噪声后的图像

    支持的噪声类型:
    1. 高斯噪声 (Gaussian)
    2. 泊松噪声 (Poisson)
    3. 椒盐噪声 (Salt & Pepper)
    4. 乘性噪声 (Speckle)
    5. 均匀噪声 (Uniform)
    6. 周期性噪声 (Periodic)
    """
    # 验证输入图像
    if not isinstance(pic, np.ndarray):
        raise ValueError("输入必须是numpy数组")

    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # 确定噪声类型
    if noise_type is None:
        if ratio is None:
            ratio = np.random.random()

        if ratio < 0.2:
            noise_type = 'poisson'
        elif ratio < 0.4:
            noise_type = 'salt_pepper'
        elif ratio < 0.6:
            noise_type = 'speckle'
        elif ratio < 0.8:
            noise_type = 'uniform'
        # elif ratio<0.9:
        #     # 周期性噪声
        #     noise_type = 'periodic'
        else:
            noise_type = 'gaussian'

    # 确保强度在合理范围
    intensity = np.clip(intensity, 0.01, 0.99)

    # 保存原始图像范围
    orig_min = np.min(pic)
    orig_max = np.max(pic)
    orig_dtype = pic.dtype

    # 归一化图像到[0,1]范围
    if orig_dtype != np.float32 and orig_dtype != np.float64:
        pic_normalized = pic.astype(np.float32) / 255.0
    else:
        pic_normalized = pic.copy()

    # 应用不同类型的噪声
    if noise_type == 'gaussian':
        # 高斯噪声 - 适合模拟传感器噪声
        mean = 0
        var = intensity * 0.1  # 方差控制噪声强度
        sigma = var ** 0.5
        noise = np.random.normal(mean, sigma, pic_normalized.shape)
        noisy = pic_normalized + noise

    elif noise_type == 'poisson':
        # 泊松噪声 - 适合模拟光子计数噪声
        # 使用skimage库更准确的泊松噪声实现
        noisy = random_noise(pic_normalized, mode='poisson')

    elif noise_type == 'salt_pepper':
        # 椒盐噪声 - 适合模拟数据传输错误
        # 更高效的实现方式
        noisy = pic_normalized.copy()
        s_vs_p = 0.5  # 盐和胡椒的比例

        # 生成随机掩码
        random_mask = np.random.random(pic_normalized.shape)

        # 设置盐噪声 (最大值)
        noisy[random_mask < intensity / 2] = 1.0

        # 设置胡椒噪声 (最小值)
        noisy[random_mask > 1 - intensity / 2] = 0.0

    elif noise_type == 'speckle':
        # 乘性噪声 - 适合模拟超声波图像噪声
        noise = np.random.randn(*pic_normalized.shape) * intensity
        noisy = pic_normalized + pic_normalized * noise

    elif noise_type == 'uniform':
        # 均匀噪声 - 适合模拟量化噪声
        noise = np.random.uniform(-intensity, intensity, pic_normalized.shape)
        noisy = pic_normalized + noise

    elif noise_type == 'periodic':
        # 周期性噪声 - 适合模拟电磁干扰，
        noisy = pic_normalized.copy()
        rows, cols = pic_normalized.shape[:2]

        # 创建周期性噪声模式
        for i in range(rows):
            # 正弦波模式
            noise_val = intensity * np.sin(2 * np.pi * i / 20)
            if len(pic_normalized.shape) == 2:  # 灰度图像
                noisy[i, :] += noise_val
            else:  # 彩色图像
                noisy[i, :, :] += noise_val

    else:
        raise ValueError(f"未知噪声类型: {noise_type}")

    # 裁剪到[0,1]范围
    noisy = np.clip(noisy, 0.0, 1.0)

    # 恢复原始数据类型和范围
    if orig_dtype != np.float32 and orig_dtype != np.float64:
        noisy = (noisy * 255).astype(orig_dtype)
        ret, noisy = cv2.threshold(noisy, 127, 255, cv2.THRESH_BINARY)
    else:
        # 恢复原始范围
        noisy = orig_min + noisy * (orig_max - orig_min)

    return noisy



def motion_blur(image, degree=8, angle=20):
    """
        对输入图像应用运动模糊效果

        参数:
        image: 输入图像（PIL图像或numpy数组）
        degree: 模糊程度（核大小），值越大模糊效果越强（默认8）
        angle: 模糊角度（度），0为水平模糊，90为垂直模糊（默认20）

        返回:
        blurred: 应用运动模糊后的图像（numpy数组，uint8类型）
        """
    image = np.array(image)

    # 生成运动模糊核
    # 1. 创建旋转矩阵：以模糊核中心为旋转中心，指定角度和缩放比例1（不缩放）
    #    (degree/2, degree/2) - 旋转中心点（核的中心）
    #    angle - 旋转角度（度）
    #    1 - 缩放因子（保持原始大小）
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)

    # 2. 创建初始模糊核：对角线为1的degree×degree矩阵
    #    这个矩阵表示沿着对角线的运动轨迹
    motion_blur_kernel = np.diag(np.ones(degree))

    # 3. 应用旋转矩阵到模糊核：将初始对角线核旋转到指定角度
    #    使用双线性插值（cv2.warpAffine默认）保持核的平滑性
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    # 4. 归一化模糊核：使所有元素之和为1，避免改变图像整体亮度
    motion_blur_kernel = motion_blur_kernel / np.sum(motion_blur_kernel)

    # 应用模糊核到图像
    # 使用cv2.filter2D进行二维卷积操作
    # -1参数表示输出图像深度与输入相同
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # 归一化处理：将像素值缩放到0-255范围
    # 使用MINMAX归一化方法：将最小值和最大值分别映射到0和255
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)

    # 将浮点型数组转换为uint8类型（图像标准格式）
    blurred = np.array(blurred, dtype=np.uint8)

    return blurred


# 图像添加噪声
def pic_add_blur(pic=np.zeros((5, 5)), ratio=np.random.random()):
    """
    随即添加模糊
    :param pic:原始图像数据
    :param ratio: 随机因子，添加模糊的类型
    :return: 处理后的图像数据
    """
    # print(ratio)
    pic_new = np.zeros((10, 10))
    if ratio < 0.2:
        # 添加高斯模糊
        # print('添加高斯模糊')
        pic_new = cv2.GaussianBlur(pic, (9, 9), 0)  # 高斯模糊
    elif ratio < 0.4:
        pic_new = motion_blur(pic)  # 运动模糊
    elif ratio < 0.6:
        # 此为均值模糊
        # （30,1）为一维卷积核，指在 x，y 方向偏移多少位
        pic_new = cv2.blur(pic, (8, 1))
    elif ratio < 0.8:
        # 此为中值模糊，常用于去除椒盐噪声
        pic_new = cv2.medianBlur(pic, 5)
    else:
        # 还有概率不添加运动模糊
        pic_new = pic

    return pic_new



# 图像镜像
def pic_mirror(pic=np.zeros((5, 5)), ratio=np.random.random()):
    if len(pic.shape)==2:
        pic_new = np.ones_like(pic)

        if ratio < 0.25:
            # print('上下镜像')
            for i in range(pic.shape[0]):
                pic_new[i] = pic[-i]
        elif ratio < 0.5:
            # print('左右镜像')
            for j in range(pic.shape[1]):
                pic_new[:, j] = pic[:, -j]
        elif ratio < 0.75:
            # print('上下左右镜像')
            for i in range(pic.shape[0]):
                for j in range(pic.shape[1]):
                    pic_new[i, j] = pic[-i, -j]
        else:
            # print('原图像')
            pic_new = pic
        return pic_new
    elif len(pic.shape)==3:
        pic_new = np.zeros_like(pic)

        if ratio < 0.1:
            # pic_new = pic
            # # print('上下镜像')
            for i in range(pic.shape[0]):
                for j in range(pic.shape[1]):
                    pic_new[i, j, :] = pic[i, -j, :]
        elif ratio < 0.4:
            # print('左右镜像')
            for i in range(pic.shape[0]):
                for k in range(pic.shape[2]):
                    pic_new[i, :, k] = pic[i, :, -k]
        elif ratio < 0.75:
            pic_new = pic
            # # print('上下左右镜像')
            for i in range(pic.shape[0]):
                for j in range(pic.shape[1]):
                    for k in range(pic.shape[2]):
                        pic_new[i, j, k] = pic[i, -j, -k]
        else:
            # print('原图像')
            pic_new = pic
        return pic_new
    else:
        print('error tailor shape:{}'.format(pic.shape))
        exit(0)


# def pic_tailor_3dim(pic=np.zeros((3, 50, 50)), pic_shape_ex=[224, 224]):
#     if len(pic.shape) != 3:
#         print('error pic shape:{}'.format(pic.shape))
#         exit(0)
#     elif (pic_shape_ex[0] > pic.shape[-2]) or  (pic_shape_ex[1] > pic.shape[-1]):
#         print('too large tailor shape:{}, pic shape is:{}'.format(pic_shape_ex, pic.shape))
#         exit(0)
#
#     if isinstance(pic_shape_ex[0], int):
#         x_windows_pixel = np.random.randint(pic_shape_ex[-2] - 5, pic_shape_ex[-2] + 5)
#         y_windows_pixel = np.random.randint(pic_shape_ex[-1] - 5, pic_shape_ex[-1] + 5)
#     elif isinstance(pic_shape_ex[1], float):
#         x_windows_pixel = int(pic_shape_ex[-2] * pic.shape[-2])
#         y_windows_pixel = int(pic_shape_ex[-1] * pic.shape[-1])
#
#     x_index_start = np.random.randint(0, pic.shape[-2] - x_windows_pixel - 1)
#     y_index_start = np.random.randint(0, pic.shape[-1] - y_windows_pixel - 1)
#
#     pic_new = pic[:, x_index_start:x_index_start + x_windows_pixel, y_index_start:y_index_start + y_windows_pixel]
#     return pic_new


def pic_tailor_3dim(pic=np.zeros((3, 50, 50)), pic_shape_ex=[224, 224], min_size=16):
    """
    图像随机裁剪函数（修复版）

    参数:
    pic: 输入图像 (C, H, W)
    pic_shape_ex: 目标裁剪尺寸 [height, width] 或比例 [height_ratio, width_ratio]
    min_size: 最小裁剪尺寸（防止裁剪过小）

    返回:
    pic_new: 裁剪后的图像
    """
    # 1. 验证输入形状
    if len(pic.shape) != 3:
        raise ValueError(f"输入图像必须是3维 (C,H,W)，当前形状: {pic.shape}")

    # 2. 计算实际裁剪尺寸
    if isinstance(pic_shape_ex[0], int):
        # 整数模式：在目标尺寸±5像素范围内随机
        x_windows_pixel = np.random.randint(
            max(min_size, pic_shape_ex[0] - 5),  # 下限
            min(pic.shape[1], pic_shape_ex[0] + 6)  # 上限
        )
        y_windows_pixel = np.random.randint(
            max(min_size, pic_shape_ex[1] - 5),
            min(pic.shape[2], pic_shape_ex[1] + 6)
        )
    elif isinstance(pic_shape_ex[0], float):
        # 浮点数模式：按比例裁剪
        x_windows_pixel = int(pic_shape_ex[0] * pic.shape[1])
        y_windows_pixel = int(pic_shape_ex[1] * pic.shape[2])
    else:
        raise TypeError("pic_shape_ex 元素必须是 int 或 float")

    # 3. 确保裁剪尺寸有效
    x_windows_pixel = max(min_size, min(x_windows_pixel, pic.shape[1]))
    y_windows_pixel = max(min_size, min(y_windows_pixel, pic.shape[2]))

    # 4. 计算安全起始位置
    max_x_start = max(0, pic.shape[1] - x_windows_pixel)
    max_y_start = max(0, pic.shape[2] - y_windows_pixel)

    # 5. 随机选择起始点（确保有空间裁剪）
    if max_x_start > 0:
        x_index_start = np.random.randint(0, max_x_start)
    else:
        x_index_start = 0  # 如果无法移动，从0开始

    if max_y_start > 0:
        y_index_start = np.random.randint(0, max_y_start)
    else:
        y_index_start = 0

    # 6. 执行裁剪
    pic_new = pic[
              :,
              x_index_start:x_index_start + x_windows_pixel,
              y_index_start:y_index_start + y_windows_pixel
              ]

    return pic_new


# 图像随机裁剪
def pic_tailor(pic=np.zeros((5, 5)), pic_shape_ex=[224, 224]):
    if len(pic.shape) == 2:
        if (pic.shape[0] < pic_shape_ex[0]) | (pic.shape[1] < pic_shape_ex[1]):
            print('shape error ,pic shape is smaller than target shape:{},{}'.format(pic.shape, pic_shape_ex))
            exit(0)
        pic = pic.reshape((1, pic.shape[0], pic.shape[1]))
        pic_new = pic_tailor_3dim(pic=pic, pic_shape_ex=pic_shape_ex)
        pic_new = pic_new[0, :, :]
        return pic_new

    elif len(pic.shape) == 3:
        pic_new = pic_tailor_3dim(pic=pic, pic_shape_ex=pic_shape_ex)
        return pic_new


def random_morphology(image, ratio=random.random(), k_size=None):
    """
    对输入图像进行随机的形态学处理（溶蚀、膨胀、开运算、闭运算）

    参数:
    image: 输入图像（灰度或二值图像）
    ratio: 随机选择形态学操作的参数（0-1之间），如果为None则随机生成
    <0.25：溶蚀（腐蚀）
    <0.50：膨胀
    <0.75:开运算（先腐蚀后膨胀）
    <1.00:闭运算（先膨胀后腐蚀）
    k_size: 核大小（奇数），如果为None则随机生成

    返回:
    处理后的图像
    """
    # 确保输入是numpy数组
    if not isinstance(image, np.ndarray):
        raise ValueError("输入必须是numpy数组")

    if k_size is None:
        k_size = np.random.choice([3, 5, 7])

    # 创建形态学核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    # 根据随机比例选择形态学操作
    if ratio < 0.25:
        # 溶蚀（腐蚀）
        result = cv2.erode(image, kernel, iterations=1)
        # print(f"应用溶蚀操作 (k_size={k_size})")
    elif ratio < 0.5:
        # 膨胀
        result = cv2.dilate(image, kernel, iterations=1)
        # print(f"应用膨胀操作 (k_size={k_size})")
    elif ratio < 0.75:
        # 开运算（先腐蚀后膨胀）
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        # print(f"应用开运算 (k_size={k_size})")
    else:
        # 闭运算（先膨胀后腐蚀）
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        # print(f"应用闭运算 (k_size={k_size})")

    return result



def get_pic_random(data_img_o, data_depth, RB_index=-1, pic_shape=(224, 224)):
    data_img = copy.deepcopy(data_img_o)

    if len(data_img.shape) == 2:
        # data_img, data_depth = get_test_ele_data()
        # data_img = data_img[1200:1600, :]
        # data_depth = data_depth[1200:1600, :]
        # data_depth = np.zeros((data_img.shape[0], 1))

        # 图片绕井壁 按RB曲线 旋转
        pic_new, rb_random = pic_rotate_random(data_img, depth=data_depth)
        # print('111{}'.format(pic_new.shape))
        # plt.figure(figsize=(10, 7))
        # plt.plot(data_depth.ravel(), rb_random.ravel())
        # plt.show()

        # 图片像素偏移
        pic_new = pic_dim_shift(pic_new)
        # print('222{}'.format(pic_new.shape))

        # 图像降噪 以及 图像去噪
        ratio = np.random.random()
        if ratio < 0.7:
            pic_new = pic_denoise(pic_new)
        elif ratio < 0.8:
            pic_new = pic_add_noise(pic_new)
        # print('333{}'.format(pic_new.shape))

        # 图像镜像
        pic_new = pic_mirror(pic_new)
        # print('444{}'.format(pic_new.shape))
        pic_shape = (224, 224)
        # 图像裁剪
        pic_new = np.clip(pic_tailor(pic_new, pic_shape), 0, 255)
        # print('555{}'.format(pic_new.shape))

        # 图像缩放以及 归一化
        pic_new = cv2.resize(pic_new, pic_shape)/256
        # pic_new = pic_scale_normal(pic_new, pic_shape)/256
        return pic_new
    elif len(data_img.shape) == 3:
        pic_new = np.zeros_like(data_img)

        # 获取随机的RB曲线 并对成像数据进行 绕井壁旋转
        np.random.seed()
        rb_random, RB_index = get_random_RB_all(depth=data_depth, RB_index=RB_index)
        # print(rb_random.shape)
        for i in range(data_img.shape[0]):
            pic_new[i, :, :] = pic_rotate_by_Rb(pic=data_img[i, :, :], Rb=rb_random)
        # show_Pic([pic_new[0], pic_new[1]], pic_order='12')

        np.random.seed()
        # 图像降噪 以及 图像去噪
        # 只对 第一维 数据进行 图像的 降噪、模糊等 处理
        ratio = np.random.random()
        if ratio < 0.6:
            pic_new[0, :, :] = pic_denoise(pic_new[0, :, :])
        elif ratio < 0.7:
            pic_new[0, :, :] = pic_add_noise(pic_new[0, :, :])
        elif ratio < 0.8:
            pic_new[0, :, :] = pic_add_blur(pic_new[0, :, :])
        # print('333{}'.format(pic_new.shape))

        # np.random.seed()
        # # 图像镜像
        # pic_new = pic_mirror(pic_new)
        # print('444{}'.format(pic_new.shape))

        # # 图像裁剪
        pic_new = pic_tailor(pic_new, pic_shape)

        # 图像缩放以及 归一化
        answer = []
        for i in range(pic_new.shape[0]):
            answer.append(cv2.resize(pic_new[i, :, :], pic_shape))
        # pic_new = pic_scale_normal(pic_new, pic_shape)/256
        return np.array(answer)/256, RB_index
        # rb_random = get_random_RB_all(depth=data_depth)
        #
        # for i in range(data_img.shape[0]):
        #     data_img[i, :, :] = pic_rotate_by_Rb(pic=data_img[i, :, :], Rb=rb_random)
        #
        # return data_img
    else:
        print('wrong img shape:{}'.format(data_img.shape))
        exit(0)


def get_pic_random_VIT_teacher(data_img_o, data_depth, RB_index=-1):
    data_img = copy.deepcopy(data_img_o)

    if len(data_img.shape) == 2:

        # 图片绕井壁 按RB曲线 旋转
        pic_new, rb_random = pic_rotate_random(data_img, depth=data_depth)

        # 图片像素偏移
        pic_new = pic_dim_shift(pic_new)

        # 图像降噪 以及 图像去噪
        ratio = np.random.random()
        if ratio < 0.7:
            pic_new = pic_denoise(pic_new)
        elif ratio < 0.8:
            pic_new = pic_add_noise(pic_new)

        # 图像镜像
        pic_new = pic_mirror(pic_new)
        pic_shape = (196, 196)
        # 图像裁剪
        pic_new = np.clip(pic_tailor(pic_new, pic_shape), 0, 255)

        # 图像缩放以及 归一化
        pic_new = cv2.resize(pic_new, pic_shape)
        return pic_new
    elif len(data_img.shape) == 3:
        pic_new = np.zeros_like(data_img)

        np.random.seed()
        rb_random, RB_index = get_random_RB_all(depth=data_depth, RB_index=RB_index)

        # from matplotlib import pyplot as plt
        # plt.plot(data_depth, rb_random)
        # plt.show()

        for i in range(data_img.shape[0]):
            pic_new[i, :, :] = pic_rotate_by_Rb(pic=data_img[i, :, :], Rb=rb_random)

        np.random.seed()
        # 图像降噪 以及 图像去噪
        ratio = np.random.random()
        if ratio < 0.6:
            pic_new[0, :, :] = pic_denoise(pic_new[0, :, :])
        elif ratio < 0.7:
            pic_new[0, :, :] = pic_add_noise(pic_new[0, :, :])
        elif ratio < 0.8:
            pic_new[0, :, :] = pic_add_blur(pic_new[0, :, :])
        # print('333{}'.format(pic_new.shape))

        np.random.seed()
        # 图像镜像
        pic_new = pic_mirror(pic_new)
        # print('444{}'.format(pic_new.shape))

        # pic_shape = (224, 224)

        # # # # 图像裁剪
        # # pic_new = pic_tailor(pic_new, pic_shape)
        #
        # # 图像缩放以及 归一化
        # answer = []
        # for i in range(pic_new.shape[0]):
        #     answer.append(cv2.resize(pic_new[i, :, :], pic_shape))
        #     # answer.append(pic_new[i, :, :])
        # # pic_new = pic_scale_normal(pic_new, pic_shape)/256
        return pic_new.astype(np.float32), RB_index

    else:
        print('wrong img shape:{}'.format(data_img.shape))
        exit(0)


def get_pic_random_VIT_student(data_img_o, data_depth, RB_index=-1):
    data_img = copy.deepcopy(data_img_o)

    if len(data_img.shape) == 2:

        # 图片绕井壁 按RB曲线 旋转
        pic_new, rb_random = pic_rotate_random(data_img, depth=data_depth)

        # 图片像素偏移
        pic_new = pic_dim_shift(pic_new)

        # 图像降噪 以及 图像去噪
        ratio = np.random.random()
        if ratio < 0.7:
            pic_new = pic_denoise(pic_new)
        elif ratio < 0.8:
            pic_new = pic_add_noise(pic_new)

        # 图像镜像
        pic_new = pic_mirror(pic_new)
        pic_shape = (196, 196)
        # 图像裁剪
        pic_new = np.clip(pic_tailor(pic_new, pic_shape), 0, 255)

        # 图像缩放以及 归一化
        pic_new = cv2.resize(pic_new, pic_shape)
        return pic_new
    elif len(data_img.shape) == 3:
        pic_new = np.zeros_like(data_img)

        np.random.seed()
        rb_random, RB_index = get_random_RB_all(depth=data_depth, RB_index=RB_index)

        for i in range(data_img.shape[0]):
            pic_new[i, :, :] = pic_rotate_by_Rb(pic=data_img[i, :, :], Rb=rb_random)

        np.random.seed()
        # 图像降噪 以及 图像去噪
        ratio = np.random.random()
        if ratio < 0.3:
            pic_new[0, :, :] = pic_denoise(pic_new[0, :, :])
        elif ratio < 0.35:
            pic_new[0, :, :] = pic_add_noise(pic_new[0, :, :])
        elif ratio < 0.4:
            pic_new[0, :, :] = pic_add_blur(pic_new[0, :, :])
        # print('333{}'.format(pic_new.shape))

        # np.random.seed()
        # # 图像镜像
        # pic_new = pic_mirror(pic_new)
        # # print('444{}'.format(pic_new.shape))

        pic_shape = (96, 96)

        # # 图像裁剪
        pic_new = pic_tailor(pic_new, pic_shape)

        # 图像缩放以及 归一化
        answer = []
        for i in range(pic_new.shape[0]):
            answer.append(cv2.resize(pic_new[i, :, :], pic_shape))
            # answer.append(pic_new[i, :, :])
        # pic_new = pic_scale_normal(pic_new, pic_shape)/256
        return np.array(answer).astype(np.float32), RB_index

    else:
        print('wrong img shape:{}'.format(data_img.shape))
        exit(0)


def pic_rorate_random(pic, Rotate_Angle=np.random.randint(10, 350)):
    """
    pic rorate around well wall，直接旋转，不是使用RB曲线进行旋转
    :param pic: pic to process
    :param Rotate_Angle:  angle the pic to process
    :return: the result rotated pic
    """

    # 裂缝绕井壁旋转操作
    pic_NEW = copy.deepcopy(pic)
    pic_NEW[:, 0:Rotate_Angle] = pic[:, -Rotate_Angle:]
    pic_NEW[:, Rotate_Angle:] = pic[:, 0:-Rotate_Angle]

    return pic_NEW


def pic_binary_random(img, kThreshold_shift=1.2, erode=True):
    """
    根据多阈值OSTU方法二值化图像
    :param img:图像的原始数据
    :param kThreshold_shift:图像动态截止值的偏移指数，相乘计算
    :return:
    """

    otsu = OtsuFastMultithreshold()
    otsu.load_image(img)
    kThresholds = otsu.calculate_k_thresholds(1)
    kThresholds[0] = int(kThresholds[0] * kThreshold_shift)
    # print('otsu thresholds is :{}'.format(kThresholds[0]))

    binary = otsu.apply_thresholds_to_image(kThresholds)

    if erode:
        binary = random_morphology(binary, k_size=3)


    return binary, img



if __name__ == '__main__':

    data_img_dyna, data_img_stat, data_depth = get_test_ele_data()
    print(data_img_dyna.shape, data_img_stat.shape, data_depth.shape)

    data_img_dyna_new = pic_tailor(data_img_dyna, pic_shape_ex=[0.5, 0.5])

    print(data_img_dyna.shape, data_img_stat.shape, data_img_dyna_new.shape, data_depth.shape)
    show_Pic([data_img_dyna, data_img_dyna_new], pic_order='12')

    pic_all = np.array([data_img_dyna, data_img_stat])
    print(pic_all.shape)
    pic_all_new = pic_tailor(pic_all, pic_shape_ex=[0.5, 0.5])

    show_Pic([data_img_dyna, data_img_stat, pic_all_new[0, :, :], pic_all_new[1, :, :]], pic_order='22')
    # pic_binary_random(data_img_dyna, 25)
    # pic_binary_random(data_img_stat, 30)
    # pic_new = pic_denoise(data_img_dyna)

    # Rb_random = get_random_RB_curve_2(depth=data_depth)
    # # plt.figure(figsize=(10, 7))
    # # plt.plot(data_depth.ravel(), Rb_random.ravel())
    # # plt.ylim([-180, 180])
    # # plt.show()
    # pic_new = pic_rotate_by_Rb(data_img, Rb_random)
    # pic_new = pic_dim_shift(pic_new)
    # pic_new = pic_denoise(pic_new)

    # pic_new = pic_mirror(data_img)

    # data_img_dyna, data_img_stat, data_depth = get_test_ele_data()
    # pic_dyna = data_img_dyna.reshape((1, data_img_dyna.shape[0], data_img_dyna.shape[1]))
    # pic_stat = data_img_stat.reshape((1, data_img_stat.shape[0], data_img_stat.shape[1]))
    # # print(pic_dyna.shape)
    # pic_all = np.append(pic_dyna, pic_stat, axis=0)
    # pic_all = np.array(pic_all)
    #
    # p_n_1, RB_index = get_pic_random_vit(pic_all, data_depth)
    # p_n_2, RB_index = get_pic_random_vit(pic_all, data_depth, RB_index)
    #
    # show_Pic([pic_all[0]*256, pic_all[1]*256, p_n_1[0]*256, p_n_2[0]*256], '22')

    # data_img = np.array(pic_all)
    # pic_show = [data_img_dyna]
    # for i in range(5):
    #     ratio = 0.1 + 0.2*i
    #     pic_t = pic_add_blur(data_img_dyna, ratio=ratio)
    #     print(pic_t.shape)
    #     pic_show.append(pic_t)
    #
    # show_Pic(pic_show, '23')
    # print(len(pic_show))





    # data_img = data_img[1200:1600, :]
    # data_depth = data_depth[1200:1600, :]
    # show_Pic([data_img, get_pic_random(), get_pic_random()], pic_str=['img_org', 'img_random1', 'img_random2'], pic_order='13')
    # RB_info = np.hstack((data_depth, Rb_random+180))
    # np.savetxt('pic_rotate_by_rb_test/Rb_info_{}.txt'.format('test_rb'), RB_info, fmt='%.4f', delimiter='\t', comments='',
    #            header='WELLNAME={}\nSTDEP\t= {}\nENDEP\t= {}\nLEV\t= {:.4f}\nUNIT\t= meter\nCURNAMES= {}\n\n#DEPTH\t{}'.format(
    #                'test_1', data_depth[0, 0], data_depth[-1, 0], data_depth[1, 0] - data_depth[0, 0], 'Rb_random', 'Rb_random'))
    # np.savetxt('pic_rotate_by_rb_test/pic_org_info_{}.txt'.format('test_rb'), np.hstack((data_depth, data_img)), fmt='%.4f', delimiter='\t', comments='',
    #            header='WELLNAME={}\nSTDEP\t= {}\nENDEP\t= {}\nLEV\t= {:.4f}\nUNIT\t= meter\nCURNAMES= {}\n\n#DEPTH\t{}'.format(
    #                'test_1', data_depth[0, 0], data_depth[-1, 0], data_depth[1, 0] - data_depth[0, 0], 'img_org', 'img_org'))
    # np.savetxt('pic_rotate_by_rb_test/pic_rotate_info_{}.txt'.format('test_rb'), np.hstack((data_depth, pic_new)), fmt='%.4f', delimiter='\t', comments='',
    #            header='WELLNAME={}\nSTDEP\t= {}\nENDEP\t= {}\nLEV\t= {:.4f}\nUNIT\t= meter\nCURNAMES= {}\n\n#DEPTH\t{}'.format(
    #                'test_1', data_depth[0, 0], data_depth[-1, 0], data_depth[1, 0] - data_depth[0, 0], 'img_rotate', 'img_rotate'))


    # data_img_shifted = pic_dim_shift(data_img)
    # show_Pic([data_img, data_img_shifted], pic_order='12', pic_str=['原始成像', '色域偏移成像'], save_pic=False, path_save='')
    # x_range, y_range = image_hist(data_img)
    # x_range_shifted, y_range_shifted = image_hist(data_img_shifted)
    # print(x_range, y_range)
    # # 全部的线段风格
    # styles = ['c:s', 'y:8', 'r:^', 'r:v', 'g:D', 'm:X', 'b:p', ':>'] # 其他可用风格 ':<',':H','k:o','k:*','k:*','k:*'
    # # # 获取全部的图例
    # # columns = [i[:-2] for i in data.columns]
    # # n,m = data.shape
    # plt.figure(figsize=(10, 7))
    #
    # # 设置字体
    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams.update({'font.size': 22})
    # plt.rc('legend', fontsize=15)
    #
    # plt.plot(x_range, y_range, styles[2], markersize=4, label='img_org')
    #
    # # 设置图片的x,y轴的限制，和对应的标签
    # plt.xlim([0, 256])
    # plt.ylim([0, 0.04])
    # plt.xlabel("pixel value distribution")
    # plt.ylabel("percentage")
    #
    # # 设置图片的方格线和图例
    # plt.grid()
    # plt.legend(loc='lower right', framealpha=0.7)
    # plt.tight_layout()
    # # plt.show()
    #
    # # 如果想保存图片，请把plt.show注释掉，然后把下面这行代码打开注释
    # plt.savefig("img.png", dpi=800)
    # plt.plot(x_range_shifted, y_range_shifted, styles[0], markersize=4, label='img_shifted')
    #
    # # 设置图片的x,y轴的限制，和对应的标签
    # plt.xlim([0, 256])
    # plt.ylim([0, 0.04])
    # plt.xlabel("pixel value distribution")
    # plt.ylabel("percentage")
    #
    # # 设置图片的方格线和图例
    # plt.grid()
    # plt.legend(loc='lower right', framealpha=0.7)
    # plt.tight_layout()
    # # plt.show()
    #
    # # 如果想保存图片，请把plt.show注释掉，然后把下面这行代码打开注释
    # plt.savefig("img2.png", dpi=800)epth[1, 0] - data_depth[0, 0], 'img_org', 'img_org'))
    # np.savetxt('pic_rotate_by_rb_test/pic_rotate_info_{}.txt'.format('test_rb'), np.hstack((data_depth, pic_new)), fmt='%.4f', delimiter='\t', comments='',
    #            header='WELLNAME={}\nSTDEP\t= {}\nENDEP\t= {}\nLEV\t= {:.4f}\nUNIT\t= meter\nCURNAMES= {}\n\n#DEPTH\t{}'.format(
    #                'test_1', data_depth[0, 0], data_depth[-1, 0], data_depth[1, 0] - data_depth[0, 0], 'img_rotate', 'img_rotate'))


    # data_img_shifted = pic_dim_shift(data_img)
    # show_Pic([data_img, data_img_shifted], pic_order='12', pic_str=['原始成像', '色域偏移成像'], save_pic=False, path_save='')
    # x_range, y_range = image_hist(data_img)
    # x_range_shifted, y_range_shifted = image_hist(data_img_shifted)
    # print(x_range, y_range)
    # # 全部的线段风格
    # styles = ['c:s', 'y:8', 'r:^', 'r:v', 'g:D', 'm:X', 'b:p', ':>'] # 其他可用风格 ':<',':H','k:o','k:*','k:*','k:*'
    # # # 获取全部的图例
    # # columns = [i[:-2] for i in data.columns]
    # # n,m = data.shape
    # plt.figure(figsize=(10, 7))
    #
    # # 设置字体
    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams.update({'font.size': 22})
    # plt.rc('legend', fontsize=15)
    #
    # plt.plot(x_range, y_range, styles[2], markersize=4, label='img_org')
    #
    # # 设置图片的x,y轴的限制，和对应的标签
    # plt.xlim([0, 256])
    # plt.ylim([0, 0.04])
    # plt.xlabel("pixel value distribution")
    # plt.ylabel("percentage")
    #
    # # 设置图片的方格线和图例
    # plt.grid()
    # plt.legend(loc='lower right', framealpha=0.7)
    # plt.tight_layout()
    # # plt.show()
    #
    # # 如果想保存图片，请把plt.show注释掉，然后把下面这行代码打开注释
    # plt.savefig("img.png", dpi=800)
    # plt.plot(x_range_shifted, y_range_shifted, styles[0], markersize=4, label='img_shifted')
    #
    # # 设置图片的x,y轴的限制，和对应的标签
    # plt.xlim([0, 256])
    # plt.ylim([0, 0.04])
    # plt.xlabel("pixel value distribution")
    # plt.ylabel("percentage")
    #
    # # 设置图片的方格线和图例
    # plt.grid()
    # plt.legend(loc='lower right', framealpha=0.7)
    # plt.tight_layout()
    # # plt.show()
    #
    # # 如果想保存图片，请把plt.show注释掉，然后把下面这行代码打开注释
    # plt.savefig("img2.png", dpi=800)