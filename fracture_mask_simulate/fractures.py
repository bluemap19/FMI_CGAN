import copy
import cv2
import numpy as np
import random
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic
from PIL import Image, ImageDraw
import random

# 根据单缝图库，模拟多缝生成的文件


def pic_rorate_random(pic, Rotate_Angle=np.random.randint(10, 350)):
    """
    pic rorate around well wall
    :param pic: pic to process
    :param Rotate_Angle:  angle the pic to process
    :return: the result rotated pic
    """

    # 裂缝绕井壁旋转操作
    pic_NEW = copy.deepcopy(pic)
    pic_NEW[:, 0:Rotate_Angle] = pic[:, -Rotate_Angle:]
    pic_NEW[:, Rotate_Angle:] = pic[:, 0:-Rotate_Angle]

    return pic_NEW, Rotate_Angle



# 定义一个随机增加 随机的膨胀、腐蚀、开闭操作
def pic_open_close_random(pic):
    # # 噪声去除
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))            # 矩形
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))           # 交叉形
    x_k_size = np.random.randint(1, 2)
    y_k_size = np.random.randint(1, 2)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*x_k_size+1, 2*y_k_size+1))  # 椭圆形
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆形

    # # cv2.MORPH_CLOSE 闭运算(close)，先膨胀后腐蚀的过程。闭运算可以用来排除小黑洞。
    # # cv2.MORPH_OPEN  开运算(open) ,先腐蚀后膨胀的过程。开运算可以用来消除小黑点，在纤细点处分离物体、平滑较大物体的边界的 同时并不明显改变其面积。
    # # iterations – 操作次数，默认为1
    pic = cv2.morphologyEx(pic, cv2.MORPH_CLOSE, kernel, iterations=1)
    pic = cv2.morphologyEx(pic, cv2.MORPH_OPEN, kernel, iterations=1)

    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            if pic[i][j] > 0:
                pic[i][j] = 255

    return pic


# 生成随机形状,  用来组成基本的单孔洞特征
def random_shape(width, height):
    # 创建一个空白图像
    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)

    # 随机选择形状类型
    shape_type = random.choice(["circle", "rectangle", "ellipse", "polygon"])

    # 随机生成形状的位置和大小
    bedding_len = int(0.1 * width)
    x1 = random.randint(0+bedding_len, width-5*bedding_len)
    y1 = random.randint(0+bedding_len, height-5*bedding_len)
    x_l = np.random.randint(1, int(0.8 * (width - x1)))
    y_l = np.random.randint(1, int(0.8 * (height - y1)))
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
        draw.regular_polygon(((int((x1+x2)*0.5), int((y1+y2)*0.5)), int((y2-y1)*0.5)+2),
                             n_sides=np.random.randint(3, 8),
                             rotation=np.random.randint(10, 350),
                             fill=255)
    else:
        # 绘制椭圆
        draw.ellipse((x1, y1, x2, y2), fill=255)

    return img


# 生成多个随机形状,并将这些随机的形状拼接成随机的孔洞结构
def generate_shapes(num_shapes, width, height):
    result = Image.new("L", (width, height), color=0)
    result = np.array(result)

    # 生成多个不同形状的随机 矩形、椭圆、多边形  并将其进行叠加
    for i in range(num_shapes):
        img = random_shape(width, height)
        img = np.array(img)
        result |= img

    # 定义一个随机的旋转操作，对生成的孔洞结构进行旋转操作
    # 原图像的高、宽、通道数
    rows, cols = result.shape
    # 旋转参数：旋转中心，旋转角度， scale
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), np.random.randint(10, 350), 1)
    # 参数：原始图像，旋转参数，元素图像宽高
    result = cv2.warpAffine(result, M, (cols, rows))

    # 图像二值化
    ret, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
    # show_Pic([result], pic_order='11')

    # 随机的 开闭 操作，溶蚀一下图片信息，使得到的孔洞结构更加平滑
    result = pic_open_close_random(result)
    # show_Pic([result], pic_order='11')

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


def try_add_hole(pic, time_repetition=30, ratio_repetition=0.02):
    # 生成随机的孔洞结构信息 generate_shapes(基础图形的个数, 图像的长, 图像的宽)
    img_hole = generate_shapes(7, 50, 50)


    Flag_Add_hole = False
    location_t = None
    for i in range(time_repetition):
        # 将孔洞结构进行随机的缩放
        vugs_len = np.random.randint(2, 45)
        vugs_high = int((1 - 0.6 * np.random.random() + 0.3) * vugs_len)
        img_hole = cv2.resize(img_hole, (vugs_high, vugs_len))

        # 生成随机的 孔洞结构 位置
        x_vugs_location = np.random.randint(5, pic.shape[0] - img_hole.shape[0] - 5)
        y_vugs_location = np.random.randint(5, pic.shape[1] - img_hole.shape[1] - 5)
        # print('x,y location:({},{})\tsize:({},{})'.format(x_vugs_location, y_vugs_location, vugs_len, vugs_high))

        hole_intersection = pic[x_vugs_location:x_vugs_location + img_hole.shape[0], y_vugs_location:y_vugs_location + img_hole.shape[1]] & img_hole
        s2_intersection = np.sum(hole_intersection)
        s2_hole = np.sum(img_hole)
        if (s2_intersection/s2_hole < ratio_repetition):
            Flag_Add_hole = True
            pic[x_vugs_location:x_vugs_location + vugs_len, y_vugs_location:y_vugs_location + vugs_high] |= img_hole
            location_t = [x_vugs_location, y_vugs_location, vugs_len, vugs_high]
            break
        else:
            print('holes intersection part is too large:{}'.format(s2_intersection/s2_hole))
            pass

    return pic, location_t





# 定义一个随机增加 随机孔洞结构的函数
def add_vugs_random(pic, vug_num_p=np.random.randint(5, 10), ratio_repetition=0.01):
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
        print('vugs num:{}->{}'.format(vugs_num, i))
        pic, location_t = try_add_hole(pic, time_repetition=30, ratio_repetition=ratio_repetition)
        location_info.append(location_t)

        pic, Rotate_Angle = pic_rorate_random(pic)
        location_info.append(Rotate_Angle)

    show_Pic([pic], pic_order='11')

    return pic, location_info


# 随机的绕井壁旋转，随机的拉伸、缩放
def single_fracture_pic_preprocess_random(pic):
    # # 裂缝绕井壁旋转操作
    # Rotate_Angle = np.random.randint(10, 350)
    # pic_NEW = copy.deepcopy(pic)
    # pic_NEW[:, :Rotate_Angle] = pic[:, -Rotate_Angle:]
    # pic_NEW[:, Rotate_Angle:] = pic[:, :-Rotate_Angle]
    # # show_Pic([pic, pic_NEW], pic_order='12')
    pic, Rotate_Angle = pic_rorate_random(pic)


    # 裂缝伸缩操作
    Random_PIC_Len = np.random.randint(100, 400)
    pic_NEW = cv2.resize(pic, (pic.shape[1], Random_PIC_Len))

    return pic_NEW



# 水平缝的随机预处理 随机的绕井壁旋转，随机的拉伸、缩放
def single_horizontal_fracture_pic_preprocess_random(pic):
    """
    图像的缩放 以及随机绕井壁旋转
    :param pic: the pic to rorate
    :return:
    """

    # 裂缝绕井壁旋转操作
    pic, Rotate_Angle = pic_rorate_random(pic)

    # 裂缝伸缩操作
    Random_PIC_Len = int((1-0.2*np.random.random())*pic.shape[0])
    pic_NEW = cv2.resize(pic, (pic.shape[1], Random_PIC_Len))
    # show_Pic([pic, pic_NEW], pic_order='12')

    return pic_NEW




def try_add_fracture_random(img_background, fracture_location_depth, fracture_t, fracture_process_fun, time_repetition=20, ratio_repetition=0.05):
    s2_background = np.sum(img_background)

    Flag_Add_fractur = False
    fracture_location = None
    for i in range(time_repetition):
        # 根据裂缝预处理方法，预处理一下，裂缝信息
        fracture_n = fracture_process_fun(fracture_t)
        s2_fracture = np.sum(fracture_n)

        start_range = 0
        max_fracture_width = fracture_n.shape[0]  # 520像素对应的是 70°的高角度缝
        # 计算裂缝样本生成的 起始以及结束位置
        end_range = img_background.shape[0] - max_fracture_width - 5
        # 生成随机的裂缝位置信息
        fracture_location_depth = np.random.randint(start_range, end_range)

        # 判断裂缝部分 与 地层本来存在的裂缝部分 存在交叉部分的大小
        intersection_part_array = img_background[fracture_location_depth:fracture_location_depth + fracture_n.shape[0], :] & fracture_n
        s2_intersection = np.sum(intersection_part_array)

        if (s2_intersection/s2_fracture > ratio_repetition):
            print('fracture intersection part is too large:{}'.format(s2_intersection/s2_fracture))
        else:
            Flag_Add_fractur = True
            img_background[fracture_location_depth:fracture_location_depth + fracture_n.shape[0], :] |= fracture_n
            fracture_location = [fracture_location_depth, fracture_location_depth + fracture_n.shape[0]]
            break

    return img_background, fracture_location





# 手动孔洞缝模拟生成类
class fractures():
    # 孔洞缝素材初始化，孔洞缝模拟生成的资源库
    def __init__(self, path=r'D:\Data\img_fractures_split\other_fracture'):
        # 有一定角度的裂缝资源库的读取
        path_list = traverseFolder(path)
        print('fractures num is:{}'.format(len(path_list)))
        self.fractures_list = []
        for i in range(len(path_list)):
            pic, depth = get_ele_data_from_path(path_list[i])
            pic2 = cv2.resize(pic, (250, pic.shape[0]))
            self.fractures_list.append(pic2)
            # print(pic.shape, pic2.shape)
        # 水平缝 素材的初始化
        self.fractures_horizontal_list = []
        path_horizontal = r'D:\Data\img_fractures_split\horizontal_fracture'
        path_horizontal_list = traverseFolder(path_horizontal)
        for i in range(len(path_horizontal_list)):
            pic, depth = get_ele_data_from_path(path_horizontal_list[i])
            pic2 = cv2.resize(pic, (250, pic.shape[0]))
            self.fractures_horizontal_list.append(pic2)

        print('current folder normal fracture num is:{}, horizontal fracture num is :{}'.format(len(self.fractures_list), len(self.fractures_horizontal_list)))

    # 多缝样本的生成 随机的方式，效果可能不是那么好，存在裂缝重复的情况
    def create_fractures_random(self, fracture_Num, img_pix_len=600):
        start_range = 0
        end_range = len(self.fractures_list) - 1

        # 记录并处理抽取到的随机的裂缝
        pic_shape_list = []
        fractures_temp = []


        # 判断是否添加水平单缝
        if np.random.random() > 0.8:
            # 添加一条水平缝

            print('use horizontal fracture')
            index_t = np.random.randint(0, len(self.fractures_horizontal_list))
            fracture_hor_t = self.fractures_horizontal_list[index_t]
            # 水平缝的随机预处理
            fracture_hor_t = single_horizontal_fracture_pic_preprocess_random(fracture_hor_t)

            fractures_temp.append(fracture_hor_t)
            # pic_shape_list.append(fracture_hor_t.shape)

            # 再添加 N-1 条 其它类型的裂缝
            for i in range(fracture_Num-1):
                random_index = np.random.randint(0, len(self.fractures_list))
                fracture_t = single_fracture_pic_preprocess_random(self.fractures_list[random_index])
                fractures_temp.append(fracture_t)
                # pic_shape_list.append(fracture_t.shape)
        elif np.random.random() > 0.7:
            print('use two horizontal fracture')
            # 添加两条水平缝
            for i in range(2):
                index_t = np.random.randint(0, len(self.fractures_horizontal_list))
                fracture_hor_t = self.fractures_horizontal_list[index_t]
                # 水平缝的随机预处理
                fracture_hor_t = single_horizontal_fracture_pic_preprocess_random(fracture_hor_t)

                fractures_temp.append(fracture_hor_t)
                # pic_shape_list.append(fracture_hor_t.shape)

            # 再添加 N-2 条 其它类型的裂缝
            for i in range(fracture_Num-2):
                random_index = np.random.randint(0, len(self.fractures_list))
                fracture_t = single_fracture_pic_preprocess_random(self.fractures_list[random_index])
                fractures_temp.append(fracture_t)
                # pic_shape_list.append(fracture_t.shape)
        else:
            # 直接全用 N 条 有角度的裂缝
            for i in range(fracture_Num):
                random_index = np.random.randint(0, len(self.fractures_list))
                fracture_t = single_fracture_pic_preprocess_random(self.fractures_list[random_index])
                fractures_temp.append(fracture_t)
                # pic_shape_list.append(fracture_t.shape)
        # print(pic_shape_list)


        start_range = 0
        # 计算裂缝样本生成的 起始以及结束位置
        end_range = img_pix_len - np.max(np.array(pic_shape_list)) - 1


        # 生成裂缝样本背景图像
        img_background = np.zeros((img_pix_len, 250), dtype=np.uint8)
        # print(random_index, random_pixel_start, end_range, img_background.shape)


        # 将 每一个被选中的裂缝进行 合并 并将 每一个裂缝详细的位置信息 记录下来
        location_info = []
        for i in range(fracture_Num):
            # 生成随机的裂缝位置信息
            fracture_location_depth = np.random.randint(start_range, end_range)

            pic_temp = fractures_temp[i]
            img_background[fracture_location_depth:fracture_location_depth+pic_temp.shape[0], :] |= pic_temp

            location_info.append([fracture_location_depth, fracture_location_depth+pic_temp.shape[0]])
        # 多缝样本中也随即添加几个孔洞样本，增加真实性
        # show_Pic([img_background], pic_order='11')
        img_background, location_info = add_vugs_random(img_background, vug_num_p=8)
        # show_Pic([img_background], pic_order='11')

        return img_background

    def create_fractures_random_logical(self, fracture_Num, img_pix_len=800, vugs_num=40):
        start_range = 0
        end_range = len(self.fractures_list) - 1

        # 生成裂缝样本背景图像
        img_background = np.zeros((img_pix_len, 250), dtype=np.uint8)

        start_range = 0
        max_fracture_width = 520        # 520像素对应的是 70°的高角度缝
        # 计算裂缝样本生成的 起始以及结束位置
        end_range = img_pix_len - max_fracture_width - 1


        # 将 每一个被选中的裂缝进行 合并 并将 每一个裂缝详细的位置信息 记录下来
        location_info = []
        for i in range(fracture_Num):
            # 生成随机的裂缝位置信息
            fracture_location_depth = np.random.randint(start_range, end_range)

            # 尝试在背景图像上进行随机的裂缝添加
            # 尝试添加随机的水平缝
            if np.random.random() > 0.85:
                print('use horizontal fracture')
                index_t = np.random.randint(0, len(self.fractures_horizontal_list))
                fracture_hor_t = self.fractures_horizontal_list[index_t]
                img_background, location_t = try_add_fracture_random(img_background, fracture_location_depth, fracture_hor_t, single_horizontal_fracture_pic_preprocess_random)
            else:
                print('use angled fracture')
                index_t = np.random.randint(0, len(self.fractures_list))
                fracture_t = self.fractures_list[index_t]
                img_background, location_t = try_add_fracture_random(img_background, fracture_location_depth, fracture_t, single_fracture_pic_preprocess_random)

            location_info.append(location_t)
            # img_background[fracture_location_depth:fracture_location_depth + pic_temp.shape[0], :] |= pic_temp
            # location_info.append([fracture_location_depth, fracture_location_depth + pic_temp.shape[0]])
            show_Pic([img_background], pic_order='11')

        # 多缝样本中也随即添加几个孔洞样本，增加真实性
        img_background, location_info = add_vugs_random(img_background, vug_num_p=vugs_num)
        # show_Pic([img_background], pic_order='11')


        return img_background

    # 展示 裂缝mask 样本
    def show_fracture_lib(self, num = 16):
        fracture_show = []

        for i in range(num):
            if i < len(self.fractures_list):
                f_t = cv2.resize(self.fractures_list[i], (80, 80))
                fracture_show.append(f_t)
            else:
                j = i - len(self.fractures_list)
                f_t = cv2.resize(self.fractures_horizontal_list[j], (80, 20))
                fracture_show.append(f_t)

        # show_Pic(fracture_show, pic_order='44')

        # for i in range(len(fracture_show)):
        #     cv2.imwrite('fracture_{}.png'.format(i), fracture_show[i])

    # 展示 孔洞mask 样本
    def show_random_vugs_effect(self, num=49):
        vugs_list = get_random_vugs_pic()

        # show_Pic(vugs_list, pic_order='66')

        # for i in range(len(vugs_list)):
        #     cv2.imwrite('fvug_{}.png'.format(i), vugs_list[i])

    # 生成随机 孔洞mask 样本
    def create_random_vugs_effect(self, num=np.random.randint(50, 150), img_pix_len=400):
        img_background = np.zeros((img_pix_len, 250), dtype=np.uint8)
        img_background, location_info = add_vugs_random(img_background, vug_num_p=num)

        return img_background

    def show_one_fracture_create_process(self):
        fracture = self.fractures_list[0]
        fracture = cv2.resize(fracture, (240, 240))
        fracture = np.where(fracture[..., :] < 245, 0, 255)

        fracture_2 = np.zeros_like(fracture, dtype=np.uint8)
        # fracture_3 = np.zeros_like(fracture, dtype=np.uint8)

        for j in range(fracture.shape[1]):
            if (j>0)&(j<fracture.shape[1]):
                for i in range(fracture.shape[0]):
                    if (fracture[i][j] != fracture[i-1][j]):
                        fracture_2[i][j] = 255

        # show_Pic([fracture, fracture_2], pic_order='12')
        cv2.imwrite('fracture_1_temp.png', fracture)
        cv2.imwrite('fracture_1_temp_split.png', fracture_2)

if __name__ == '__main__':
    b = fractures()
    # b.show_one_fracture_create_process()
    for i in range(10):
        fs = b.create_fractures_random_logical(7, img_pix_len=1600, vugs_num=1500)
        cv2.imwrite('fracture_moni_{}.png'.format(i+3), fs)
    # b.create_fractures_random(3)
    # b.create_fractures_random(2)
    # b.create_fractures_random(1)
    # b.show_fracture_lib()
    # b.show_random_vugs_effect()

    # multi_fracture_list = []
    # multi_fracture_openAclose_list = []
    # num_random_fracture = 16
    # for i in range(num_random_fracture):
    #     fracture_t = b.create_fractures_random(2)
    #     fracture_t = cv2.resize(fracture_t, (250, 250))
    #     multi_fracture_list.append(fracture_t)
    #     multi_fracture_openAclose_list.append(pic_open_close_random(fracture_t))
    # # show_Pic(multi_fracture_list, pic_order='44')
    # # show_Pic(multi_fracture_openAclose_list, pic_order='44')
    # for i in range(len(multi_fracture_list)):
    #     cv2.imwrite('multi_fracure_{}_7.png'.format(i), multi_fracture_list[i])
    #     # cv2.imwrite('multi_fracure_process_{}_4.png'.format(i), multi_fracture_openAclose_list[i])

    # vugs_list = []
    # num_vugs_pic = 16
    # for i in range(num_vugs_pic):
    #     vugs_p_t = b.create_random_vugs_effect(num=10)
    #     vugs_p_t = cv2.resize(vugs_p_t, (250, 250))
    #     vugs_list.append(vugs_p_t)
    #
    #     cv2.imwrite('vugs_effect_{}.png'.format(i), vugs_p_t)
    #
    # show_Pic(vugs_list, pic_order='44')
    #
    # exit(0)