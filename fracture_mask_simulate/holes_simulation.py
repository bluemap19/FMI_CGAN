import copy
import math
import cv2
import numpy as np
import random
from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic, binary_pic
from PIL import Image, ImageDraw
import random
np.set_printoptions(suppress=True)
# 根据单缝图库，模拟多缝生成的文件


def pic_rorate_random(pic, Rotate_Angle=np.random.randint(10, 350)):
    """
    pic rorate around well wall
    :param pic: pic to process
    :param Rotate_Angle:  angle the pic to process
    :return: the result rotated pic
    """
    pic_NEW = copy.deepcopy(pic)
    pic_NEW[:, -Rotate_Angle:] = pic[:, 0:Rotate_Angle]
    pic_NEW[:, 0:-Rotate_Angle] = pic[:, Rotate_Angle:]

    return pic_NEW, Rotate_Angle


# 定义一个随机增加 随机的膨胀、腐蚀、开闭操作
def pic_open_close_random(pic):
    # # 噪声去除
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))            # 矩形
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))           # 交叉形
    x_k_size = np.random.randint(1, 2) *2 +1
    y_k_size = np.random.randint(1, 2) *2 +1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*x_k_size+1, 2*y_k_size+1))  # 椭圆形
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆形

    # # cv2.MORPH_CLOSE 闭运算(close)，先膨胀后腐蚀的过程。闭运算可以用来排除小黑洞。
    # # cv2.MORPH_OPEN  开运算(open) ,先腐蚀后膨胀的过程。开运算可以用来消除小黑点，在纤细点处分离物体、平滑较大物体的边界的 同时并不明显改变其面积。
    pic = cv2.morphologyEx(pic, cv2.MORPH_CLOSE, kernel, iterations=1)
    pic = cv2.morphologyEx(pic, cv2.MORPH_OPEN, kernel, iterations=1)

    # 二值化，将所有裂缝区域进行 最大值化
    _, pic = cv2.threshold(pic, 5, 255, cv2.THRESH_BINARY)  # cv2.THRESH_BINARY：当像素值大于阈值时，赋予最大值；否则为0。
    return pic


# 生成随机形状,  用来组成单孔洞的基本单元，包括 圆形"circle", 方形"rectangle", 椭圆形"ellipse", 多边形"polygon"
def random_shape(width, height):
    # 创建一个空白图像
    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)

    # 随机选择形状类型
    shape_type = random.choice(["circle", "rectangle", "ellipse", "polygon"])

    # 随机生成形状的位置和大小
    bedding_len = int(0.1 * width)
    x1 = random.randint(0+bedding_len, width-2*bedding_len)
    y1 = random.randint(0+bedding_len, height-2*bedding_len)
    x_l = np.random.randint(2, int(0.7 * (width - x1)))
    y_l = np.random.randint(2, int(0.7 * (height - y1)))
    x2 = x1 + x_l
    y2 = y1 + y_l

    # print(shape_type, x1, y1, x2, y2)
    # 根据形状类型绘制形状
    if shape_type == "circle":
        # 绘制圆
        draw.ellipse((x1, y1, x2, y2), fill=255)
    elif shape_type == "rectangle":
        # 绘制矩形
        draw.rectangle((x1, y1, x2, y2), fill=255)
    elif shape_type == "polygon":
        # draw.polygon((x1, y1, x2, y2), fill=255)
        draw.regular_polygon(((int((x1+x2)*(0.3+random.random()*0.4)), int((y1+y2)*(0.3+random.random()*0.4))), int((y2-y1)*(0.3+random.random()*0.4))+2),
                             n_sides=np.random.randint(3, 8),
                             rotation=np.random.randint(10, 350),
                             fill=255)
    else:
        # 绘制椭圆
        draw.ellipse((x1, y1, x2, y2), fill=255)

    return img


# 生成随机的孔洞结构，通过生成多个随机形状,并将这些随机的形状拼接成随机孔洞结构
def generate_shapes(num_shapes, width, height, time_repetition=30):
    result = np.zeros((width, height), dtype=np.uint8)

    # 生成多个不同形状的随机 矩形、椭圆、多边形  并将其进行叠加
    for i in range(num_shapes):
        for j in range(time_repetition):
            if np.sum(result) == 0:
                img = random_shape(width, height)
                img = np.array(img)
                result |= img
            else:
                img = random_shape(width, height)
                img = np.array(img)
                part_and = img & result
                part_and_ratio = np.sum(part_and)/(np.sum(img)+1)
                if part_and_ratio > 0.1 and np.random.random() < 0.5:
                    result |= img
                    break
                else:
                    continue

    # 随机的 开闭 操作，溶蚀一下图片信息，使得到的孔洞结构更加平滑
    result = pic_open_close_random(result)

    # 图像二值化
    ret, result = cv2.threshold(result, 10, 255, cv2.THRESH_BINARY)
    return result


def get_random_vugs_pic(vug_num_p=1):
    """
    # 获取随机个数的孔洞图形
    :param vug_num_p: num you want to ge the sample of holes
    :return: the list of holes
    """

    # 定义 对图像添加多少随机的孔洞信息
    vugs_num = vug_num_p
    vugs_list = []
    for i in range(vugs_num):
        # 生成随机的孔洞结构信息 generate_shapes(基础图形的个数, 图像的长, 图像的宽)
        vugs_img = generate_shapes(7, 50, 50)
        vugs_list.append(vugs_img)

    return vugs_list


# 在图像上添加 孔洞特征
def try_add_hole(pic, time_repetition=30, ratio_repetition=0.02, vugs_shape_configuration=[[2, 27], [2, 27]]):
    """
    :param pic: 待添加的图像数据
    :param time_repetition: 尝试进行孔洞添加时，可能会存在重复区域，重复过高会重新尝试添加，这个设置最高可重复次数，次数越大成功几率越高
    :param ratio_repetition: 尝试进行孔洞添加时，可能会存在重复区域，这个配置不同新旧孔洞最大可重复百分比，超过这个百分比就会重复尝试，直到添加成功或者是time_repetition耗尽
    :param vugs_shape_configuration:孔洞形状上下限,只支持形状2*2的list，[[长度下限,长度上限], [宽度下限,宽度上限]]
    :return:
    """

    location_t = None
    for i in range(time_repetition):
        # 将孔洞结构进行随机的缩放
        # 生成随机的孔洞结构信息 generate_shapes(基础图形的个数, 图像的长, 图像的宽)
        img_hole = generate_shapes(11, 80, 80)

        vugs_len = np.random.randint(vugs_shape_configuration[0][0], vugs_shape_configuration[0][1]) * 2
        vugs_high = np.random.randint(vugs_shape_configuration[1][0], vugs_shape_configuration[1][1]) * 2
        img_hole = cv2.resize(img_hole, (vugs_high, vugs_len))

        h, w = img_hole.shape
        max_edge = np.max(np.array([h, w]))
        img_hole_rotate = np.zeros((max_edge, max_edge), dtype=np.uint8)
        img_hole_rotate[max_edge//2-h//2:max_edge//2+h//2, max_edge//2-w//2:max_edge//2+w//2] = img_hole

        # 定义一个随机的旋转操作，对生成的孔洞结构进行旋转操作
        # 原图像的高、宽、通道数
        h_r, w_r = img_hole_rotate.shape
        # 旋转参数：旋转中心，旋转角度， scale
        M = cv2.getRotationMatrix2D((w_r/2, h_r/2), np.random.randint(10, 350), 1)
        # 参数：原始图像，旋转参数，元素图像宽高
        img_hole = cv2.warpAffine(img_hole_rotate, M, (w_r, h_r))

        # 生成随机的 孔洞结构 位置
        x_vugs_location = np.random.randint(5, pic.shape[0] - img_hole.shape[0] - 5)
        y_vugs_location = np.random.randint(5, pic.shape[1] - img_hole.shape[1] - 5)

        hole_intersection = pic[x_vugs_location:x_vugs_location + img_hole.shape[0], y_vugs_location:y_vugs_location + img_hole.shape[1]] & img_hole
        s2_intersection = np.sum(hole_intersection)//255
        s2_hole = np.sum(img_hole)//255 + 1
        if (s2_intersection/s2_hole < ratio_repetition):
            pic[x_vugs_location:x_vugs_location + h_r, y_vugs_location:y_vugs_location + w_r] |= img_hole
            location_t = [x_vugs_location, y_vugs_location, vugs_len, vugs_high]
            break
        else:
            # print('holes intersection part is too large:{}'.format(s2_intersection/s2_hole))
            continue

    # 图像二值化
    ret, pic = cv2.threshold(pic, 5, 255, cv2.THRESH_BINARY)
    return pic, location_t



class holes_simulation(object):
    def __init__(self):

        pass

    # 定义一个随机增加 随机孔洞结构的函数
    def add_vugs_random(self, pic, vug_num_p=np.random.randint(5, 20), ratio_repetition=0.01, vugs_shape_configuration=[[2, 27], [2, 27]]):
        """
        :定义一个随机增加 随机孔洞结构的函数
        :param pic: the pic to add single vug and rotate pic
        :param vug_num_p: the num of vugs to add
        :return: the result of added vugs pic
        """

        # 定义 对图像添加多少随机的孔洞信息
        vugs_num = vug_num_p
        location_info = []
        for i in range(vugs_num):
            Angle = np.random.randint(1, 359)
            pic, Rotate_Angle = pic_rorate_random(pic, Rotate_Angle=Angle)
            location_info.append(Rotate_Angle)

            pic, location_t = try_add_hole(pic, time_repetition=50, ratio_repetition=ratio_repetition, vugs_shape_configuration=vugs_shape_configuration)
            location_info.append(location_t)

            pic, _ = pic_rorate_random(pic, -Rotate_Angle)

        return pic, location_info

    def genrate_new_blocks(self, config_blocks={}):
        # 1. 设置默认参数
        default_config = {
            'width': 256,
            'height': 256,
            'holes_num': 10,
            'ratio_repetition': 0.1,
            'vugs_shape_configuration':[[3, 26], [3, 26]]
        }
        # 2. 合并配置参数
        config_blocks = {**default_config, **config_blocks}


        # 3. 验证参数
        if not isinstance(config_blocks['width'], int) or config_blocks['width'] <= 0:
            raise ValueError("width必须是正整数")
        if not isinstance(config_blocks['height'], int) or config_blocks['height'] <= 0:
            raise ValueError("height必须是正整数")
        if not 0.0 <= config_blocks['ratio_repetition'] <= 1.0:
            raise ValueError("ratio_repetition，为可容忍的最大重叠度 必须在[0.0, 1.0]范围内")
        if not isinstance(config_blocks['holes_num'], int) or config_blocks['holes_num'] <= 0:
            raise ValueError("holes_num 必须为整数，代表了 孔洞添加的个数")

        # 4. 孔洞模拟部分，参数读取
        width = config_blocks['width']
        height = config_blocks['height']
        holes_num = config_blocks['holes_num']
        ratio_repetition = config_blocks['ratio_repetition']
        vugs_shape_configuration = config_blocks['vugs_shape_configuration']

        # 5. 空白图像初始化
        pic_empty = np.zeros((height, width), dtype=np.uint8)

        # 6. 根据参数生成新孔洞块状图像
        pic_blocks, location_info = self.add_vugs_random(pic_empty, vug_num_p=holes_num, ratio_repetition=ratio_repetition, vugs_shape_configuration=vugs_shape_configuration)
        return pic_blocks

if __name__ == '__main__':
    HS = holes_simulation()

    IMG_LIST = []
    for i in range(9):
        block_image = HS.genrate_new_blocks(config_blocks={
                'width': 256,
                'height': 256,
                'holes_num': 10,
                'ratio_repetition': 0.1,
                'vugs_shape_configuration':[[3, 32], [3, 32]]
            })
        IMG_LIST.append(block_image)


        block_image, location_info = HS.add_vugs_random(block_image, vug_num_p=20, ratio_repetition=0.01, vugs_shape_configuration=[[2, 27], [2, 27]])
        IMG_LIST.append(block_image)

    show_Pic(IMG_LIST, pic_order='36', figure=(20, 10))
