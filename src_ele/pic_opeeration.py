import copy
import math
import random
import numpy as np
import cv2
import os
# from fracture_mask_split.fractures import pic_open_close_random
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from src_ele.file_operation import get_test_ele_data, get_ele_data_from_path, get_random_fmi


# import seaborn as sns
# sns.set()

# 图片的一些操作
# 1.show_Pic(pic_list, pic_order='12', pic_str=[], save_pic=False, path_save='')
# 展示图片，无返回

# 2.WindowsDataZoomer(SinglePicWindows, ExtremeRatio=0.02)
# 数据缩放，把电阻的数据域映射到图像的数据域，返回原图片数组大的图片数组[m,n] int

# 3.GetPicContours(PicContours, threshold = 4000)
# 对图片进行分割，threshold代表了目标区域需要保留的最小面积大小
# 返回的 contours_Conform, contours_Drop, contours_All 代表了目标轮廓信息list，被丢掉的轮廓信息list，总的轮廓信息list
# 轮廓信息包括，轮廓面积数值，轮廓描述（即是轮廓的存放），轮廓的质心[x, y]

# 4.GetBinaryPic(ProcessingPic)
# 有点问题，别用这个函数

# 5.pic_enhence(input, windows_shape = 7, ratio_top = 0.33, ratio_migration = 5/6)
# 图片的增强函数，用的是局部梯度偏移

# 6.pic_enhence_random(input, windows_shape=3, ratio_top=0.2, ratio_migration=0.6, random_times=3)
# 图片的增强函数，用的是随机局部梯度偏移

# 7.pic_scale(input, windows_shape=3, center_ratio=0.5, x_size=100.0, y_size=100.0, ratio_top=0.1)
# 图片缩放函数，用的是局部梯度增强缩放

# 8.test_pic_enhance_effect()
# 测试图片的增强效果

# 9.adjust_gamma(image, gamma=1.0)
# 图片的 伽马增强

# 10.save_img_data(dep, data, path='')
# 保存图片

# 11.pic_smooth_effect_compare()
# 图片的增强效果对比函数，对比上面的随机局部梯度偏移、伽马增强、直方图均衡增强效果


# 图像色度反转
def traverse_pic(img):
    if np.max(img) < 1.2:
        return np.clip(1-img, 0, 1)
    else:
        return np.clip(255-img, 0, 255)

# 图像二值化
def binary_pic(pic, threshold):
    _, mask = cv2.threshold(pic, threshold, 255, cv2.THRESH_BINARY)
    return mask

# 图像hist获取
def get_pic_distribute(pic=np.random.randint(1,256,(2,2)), dist_length=9, min_V=0, max_V=256):
    pic_mean = np.mean(pic)
    pic_s2 = np.var(pic)

    if len(pic.shape)==2:
        step = (max_V-min_V)/dist_length
        pic_dist = np.zeros(dist_length)
        for i in range(pic.shape[0]):
            for j in range(pic.shape[1]):
                index_t = math.floor((pic[i][j]-min_V)/step)
                pic_dist[index_t] += 1

        pic_dist = pic_dist/pic.size
        return pic_dist
    else:
        print('wrong pic shape:{}'.format(pic.shape))
        exit(0)


def show_Pic(pic_list, pic_order=None, pic_str=[], path_save='', title='title', figure=(16, 9), show=True):
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
    # 设置后端为Agg避免PyCharm后端问题
    import matplotlib
    print(matplotlib.matplotlib_fname())
    matplotlib.use('Qt5Agg')  # 使用非交互式后端
    from matplotlib import pyplot as plt

    plt.rcParams['font.family'] = 'SimHei'

    if pic_order is None:
        num_pic = len(pic_list)
        # 使用字典简化配置
        size_config = {
            4: ('22', (9, 9)),
            6: ('23', (12, 9)),
            8: ('24', (14, 9)),
            9: ('33', (9, 9)),
            10: ('25', (18, 9)),
            12: ('34', (12, 9)),
            14: ('27', (18, 6)),
            15: ('35', (15, 9))
        }

        if num_pic in size_config:
            pic_order, figure = size_config[num_pic]
        else:
            pic_order = f'1{num_pic}'
            figure = (num_pic, 1)

    if len(pic_order) != 2:
        print(f'pic order error: {pic_order}')
        return

    # 计算图像总数
    rows, cols = map(int, pic_order)
    num = rows * cols

    if num != len(pic_list):
        print(f'pic order num is not equal to pic_list num: {len(pic_list)} vs {pic_order}')
        return

    # 自动生成标题
    pic_str += [f'Image {i + 1}' for i in range(len(pic_list) - len(pic_str))]

    # 预处理图像
    processed_pics = []
    for pic in pic_list:
        # 归一化处理
        if np.max(pic) < 4.01:
            pic = 255 * pic
        # 确保数据类型正确
        pic = np.clip(pic, 0, 255).astype(np.uint8)

        # 通道顺序调整
        if len(pic.shape) == 3 and pic.shape[0] == 3:
            pic = np.transpose(pic, (1, 2, 0))

        processed_pics.append(pic)

    plt.close('all')
    fig, axes = plt.subplots(rows, cols, figsize=figure)
    fig.suptitle(title, fontsize=18)

    # 展平轴数组以便迭代
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    for i, (ax, pic, title_str) in enumerate(zip(axes, processed_pics, pic_str)):
        ax.set_title(title_str)
        ax.axis('off')

        if len(pic.shape) == 3 and pic.shape[-1] == 3:
            ax.imshow(pic)
        else:
            ax.imshow(pic, cmap='hot')  # 使用热力图显示单通道图像

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 为标题留出空间

    if path_save:
        plt.savefig(path_save, bbox_inches='tight')

    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"显示图像时出错: {e}")
            print("尝试保存图像到临时文件...")
            temp_path = "temp_plot.png"
            plt.savefig(temp_path)
            print(f"图像已保存到: {temp_path}")

    plt.close()



# 图像归一化缩放，输入是图像矩阵pic,主要特点是极值可选择取上下限的数据分布
def WindowsDataZoomer(pic, ExtremeRatio=0.02, USE_EXTRE=False, Max_V=-1, Min_V=-1):
    bigTop = np.max(pic)
    smallTop = np.min(pic)

    if USE_EXTRE:
        ExtremePointNum = int(pic.size * ExtremeRatio)
        bigTop = np.mean(np.sort(pic.reshape(1, -1)[0])[-ExtremePointNum:])
        smallTop = np.mean(np.sort(pic.reshape(1, -1)[0])[:ExtremePointNum])

    if Max_V > 0:
        bigTop = Max_V
    if Min_V > 0:
        smallTop = Min_V

    if bigTop - smallTop < 0.0001:
        print("Error........bigTop == smallTop")
        exit(0)

    pic_new = np.clip((pic-smallTop)/(bigTop-smallTop), 0, 1).astype(np.float32)

    return pic_new


# 数据缩放，把电阻的数据域映射到图像的数据域
def WindowsDataZoomer(SinglePicWindows, ExtremeRatio=0.02):
    """
    数据缩放，把电阻的数据域映射到图像的数据域
    通过计算5%的极大值、极小值来完成，会修改原本的数组，数组依旧是小数
    修改原数据
    :param SinglePicWindows:2d np.array
    :return:no change original data
    """
    # print('Windows Data Zoomer......')
    # tem = np.argsort(SinglePicWindows.reshape(1, -1)[0], axis=-1, kind='quicksort', order=None)
    # print(np.sort(SinglePicWindows.reshape(1, -1)[0]))
    ExtremePointNum = int(SinglePicWindows.size*ExtremeRatio)
    # print('缩放的最大最小值的窗口大小：%s'%ExtremePointNum)
    # bigTop = np.mean(np.sort(SinglePicWindows.reshape(1, -1)[0])[-ExtremePointNum:])
    bigTop = np.max(SinglePicWindows)
    # print('大的一段%.5f'%bigTop)
    # smallTop = np.mean(np.sort(SinglePicWindows.reshape(1, -1)[0])[:ExtremePointNum])
    smallTop = np.min(SinglePicWindows)
    # print('小的一段%.5f'%smallTop)
    if bigTop - smallTop < 0.000001:
        print("Error........bigTop == smallTop")
        exit(0)
    Step = 256 / (bigTop - smallTop)
    # print('缩放的倍数：%.5f'%Step)
    # print(SinglePicWindows[:5, :5])
    # print('缩放前子图平均数：%.5f'%(np.mean(SinglePicWindows)))
    SinglePicWindows_new = np.copy(SinglePicWindows)
    for j in range(SinglePicWindows.shape[0]):
        for k in range(SinglePicWindows.shape[1]):
            SinglePicWindows_new[j][k] = (SinglePicWindows[j][k] - smallTop) * Step
            if SinglePicWindows_new[j][k] < 0:
                SinglePicWindows_new[j][k] = 0
            elif SinglePicWindows_new[j][k] > 255:
                SinglePicWindows_new[j][k] = 255
    # print(SinglePicWindows[:5, :5])
    # print('缩放后子图平均数：%.5f'%(np.mean(SinglePicWindows)))
    # SinglePicWindows = np.array(SinglePicWindows, dtype=np.int)
    # SinglePicWindows = np.array(SinglePicWindows, dtype=np.float)
    return SinglePicWindows_new, Step, smallTop


def GetPicContours(PicContours, threshold = 4000):
    # findContours函数第二个参数表示轮廓的检索模式
    # cv2.RETR_EXTERNAL 表示只检测外轮廓
    # cv2.RETR_LIST     检测的轮廓不建立等级关系
    # cv2.RETR_CCOMP    建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
    # cv2.RETR_TREE     建立一个等级树结构的轮廓。
    # 第三个参数method为轮廓的近似办法
    # cv2.CHAIN_APPROX_NONE     存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1 - x2），abs（y2 - y1）） == 1
    # cv2.CHAIN_APPROX_SIMPLE   压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    # cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh - Chinl chain近似算法
    contours, hierarchy = cv2.findContours(PicContours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )

    contours_Conform = [[], [], []]       # 存储符合要求的轮廓 顺序为 # 面积，轮廓，质心
    contours_Drop = [[], [], []]          # 存储不符合要求的轮廓
    contours_All = [[], [], []]           # 存储所有轮廓
    for i in range(len(contours)):
        # contour_S 为轮廓面积
        contour_S = cv2.contourArea(contours[i])
        M = cv2.moments(contours[i])
        # mc为质心
        mc = [int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])]

        if contour_S > threshold:         # 筛选出面积大于4000的轮廓
            # print('第%d个轮廓面积：' % i + str(temp))
            contours_Conform[0].append(contour_S)
            contours_Conform[1].append(contours[i])
            contours_Conform[2].append(mc)
        else:                           # 剩下的为不合格的轮廓
            # print('第%d个轮廓面积：'%i + str(temp))
            contours_Drop[0].append(contour_S)
            contours_Drop[1].append(contours[i])
            contours_Drop[2].append(mc)

        # 记录全部的轮廓信息
        contours_All[0].append(contour_S)
        contours_All[1].append(contours[i])
        contours_All[2].append(mc)
    return contours_Conform, contours_Drop, contours_All


def GetBinaryPic(ProcessingPic):
    Blur_Average = cv2.blur(ProcessingPic, (7, 5))
    Blur_Gauss = cv2.GaussianBlur(ProcessingPic, (7, 5), 0)
    Blur_Median = cv2.medianBlur(ProcessingPic, 5)

    ProcessingPic = Blur_Gauss
    firstLevel = 40
    ret, img_binary_Level1 = cv2.threshold(ProcessingPic, firstLevel, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level2 = cv2.threshold(ProcessingPic, firstLevel + 10, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level3 = cv2.threshold(ProcessingPic, firstLevel + 20, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level4 = cv2.threshold(ProcessingPic, firstLevel + 30, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level5 = cv2.threshold(ProcessingPic, firstLevel + 40, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level6 = cv2.threshold(ProcessingPic, firstLevel + 50, 255, cv2.THRESH_BINARY)

    ProcessingPic = img_binary_Level3
    Kernel_Rect = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))  # 生成形状为矩形5x5的卷积核
    Kernel_Ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 3))  # 椭圆形9x9
    kernel = np.ones((5, 5), np.uint8)
    targetKernel = Kernel_Ellipse
    Pic_erosion = cv2.erode(ProcessingPic, targetKernel, iterations=1)
    Pic_dilation = cv2.dilate(ProcessingPic, targetKernel, iterations=1)
    Pic_opening = cv2.morphologyEx(ProcessingPic, cv2.MORPH_OPEN, targetKernel)
    Pic_closing = cv2.morphologyEx(ProcessingPic, cv2.MORPH_CLOSE, targetKernel)
    Pic_opening_closing = cv2.morphologyEx(Pic_opening, cv2.MORPH_CLOSE, targetKernel)
    Pic_closing_opening = cv2.morphologyEx(Pic_closing, cv2.MORPH_CLOSE, targetKernel)
    ProcessingPic = Pic_opening_closing
    t, Pic_To_Count_Contours = cv2.threshold(ProcessingPic, 0, 255, cv2.THRESH_BINARY_INV)  # 通过阀值将其反色为白图 二值化图像反转

    return Pic_To_Count_Contours


def process_pix(index_x, index_y, input, windows_shape, max_pixel, ratio_top, ratio_migration):
    # 寻找窗口的index
    start_index_x = max(index_x-windows_shape//2, 0)
    end_index_x = min(index_x+windows_shape//2 + 1, input.shape[0])
    start_index_y = max(index_y-windows_shape//2, 0)
    end_index_y = min(index_y+windows_shape//2 + 1, input.shape[1])

    # 根据窗口index 获得窗口的 数据
    data_windows = copy.deepcopy(input[start_index_x:end_index_x, start_index_y:end_index_y]).ravel()
    value = input[index_x][index_y]

    # 根据窗口周边数据情况，计算像素移动方向， 正的为 增大，负的为 减小
    direction = -1
    if (np.sum(data_windows)-value) > (max_pixel/2) * (windows_shape*windows_shape-1):
        direction = 1
    # direction = ((np.sum(data_windows)-value)//(windows_shape*windows_shape-1))-(max_pixel//2)

    # ordered_list = sorted(data_windows)
    # small_top = np.mean(ordered_list[:int(len(ordered_list)*ratio_top)])
    # big_top = np.mean(ordered_list[-int(len(ordered_list)*ratio_top):])
    # print(small_top, big_top)
    small_top = np.min(data_windows)
    big_top = np.max(data_windows)

    if direction < 0:
        return (value - (value - small_top)*ratio_migration)
    else:
        return (value + (big_top - value)*ratio_migration)



def pic_enhence(input, windows_shape = 7, ratio_top = 0.33, ratio_migration = 5/6):
    max_pixel = np.max(input)
    data_new = copy.deepcopy(input)
    if (windows_shape%2) != 1:
        print('windows shape error...........')
        exit()

    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            data_new[i][j] = process_pix(i, j, input, windows_shape, max_pixel, ratio_top, ratio_migration)

    return data_new


# 洗牌算法随机一个数组
def shuffle(lis):
    for i in range(len(lis) - 1, 0, -1):
        p = random.randrange(0, i + 1)
        lis[i], lis[p] = lis[p], lis[i]
    return lis



# 图像的随机偏移图像增强
def pic_enhence_random(input, windows_shape=3, ratio_top=0.2, ratio_migration=0.6, random_times=3):
    if ((windows_shape % 2) != 1) | (windows_shape < 0):
        print('windows shape error...........')
        exit()
    if len(input.shape) >= 3:
        print('转换成灰度图再运行')
        exit()

    max_pixel = np.max(input)
    data_new = copy.deepcopy(input)
    all_times = input.shape[0] * input.shape[1]

    a = list(range(all_times))
    random_index_list = shuffle(a)

    for j in range(random_times):
        for i in random_index_list:
            x = i // input.shape[1]
            y = i % input.shape[1]

            data_new[x][y] = process_pix(x, y, input, windows_shape, max_pixel, ratio_top, ratio_migration)

    return data_new

# 图像缩放
def pic_scale(input, windows_shape=3, center_ratio=0.5, x_size=100.0, y_size=100.0, ratio_top=0.1):
    if x_size <= 1.0:
        x_size = int(x_size * input.shape[0])
    else:
        x_size = int(x_size)
    if y_size <= 1.0:
        y_size = int(y_size * input.shape[1])
    else:
        y_size = int(y_size)

    pic_new = np.zeros((x_size, y_size)).astype('uint8')

    if x_size>input.shape[0] | y_size>input.shape[1]:
        print('size error pic processing..............')
        exit()

    if windows_shape%2 != 1:
        print('windows shape error, must be single......')
        exit()


    for i in range(x_size):
        for j in range(y_size):
            index_x = int(i/x_size * (input.shape[0] - 1))
            index_y = int(j/y_size * (input.shape[1] - 1))

            # 寻找窗口的index
            start_index_x = max(index_x - windows_shape // 2, 0)
            end_index_x = min(index_x + windows_shape // 2 + 1, input.shape[0])
            start_index_y = max(index_y - windows_shape // 2, 0)
            end_index_y = min(index_y + windows_shape // 2 + 1, input.shape[1])

            # 根据窗口index 获得窗口的 数据
            data_windows = copy.deepcopy(input[start_index_x:end_index_x, start_index_y:end_index_y]).ravel()

            value = input[index_x][index_y]

            windows_mean = np.mean(data_windows)

            ordered_list = sorted(data_windows)
            small_top = np.mean(ordered_list[:int(len(ordered_list) * ratio_top)])
            big_top = np.mean(ordered_list[-int(len(ordered_list) * ratio_top):])

            if windows_mean > int(center_ratio * 256):
                pic_new[i][j] = int(max(value, big_top))
            elif windows_mean < int(center_ratio * 256):
                pic_new[i][j] = int(min(value, small_top))

    return pic_new


# 图像缩放
def pic_scale_normal(input, shape=(196, 196)):
    if len(input.shape) == 2:
        if (shape[0] < input.shape[0]) | (shape[1] < input.shape[1]):
            print('pic scale fun error:shape {}&{}'.format(shape,input.shape))
            exit(0)

        pic_new = np.zeros(shape).astype('uint8')

        for i in range(shape[0]):
            for j in range(shape[1]):
                index_x = int(i/shape[0] * (input.shape[0] - 1))
                index_y = int(j/shape[1] * (input.shape[1] - 1))

                pic_new[i][j] = input[index_x, index_y]

        return pic_new
    elif len(input.shape) == 3:
        img_tar = []
        for i in range(input.shape[0]):
            img_tar.append(cv2.resize(input[i], shape))
        return np.array(img_tar)
    else:
        print('error shape:{}'.format(input.shape))
        exit(0)



def test_pic_random_enhance_effect():
    data_img, data_depth = get_test_ele_data()
    print(data_img.shape, data_depth.shape)

    processing_pic = data_img[0:600, :]
    pic_EH = pic_enhence_random(processing_pic, windows_shape=3, ratio_top=0.1, ratio_migration=0.3, random_times=1)
    pic_equalizeHist = cv2.equalizeHist(processing_pic)  # 直方图均衡化

    show_Pic([processing_pic, pic_EH], save_pic=False, pic_order='12', pic_str=['pic_org', 'pic_enhance'])

    # hist_o = cv2.calcHist([np.uint8(processing_pic)], [0], None, [256], [0, 256])
    # hist_EH = cv2.calcHist([np.uint8(pic_EH)], [0], None, [256], [0, 256])
    # plt.subplot(2, 2, 1)
    # plt.plot(hist_o/processing_pic.size, label="原图灰度直方图", linestyle="--", color='g')
    # plt.legend()
    # plt.subplot(2, 2, 2)
    # plt.plot(hist_EH/pic_EH.size, label="增强后灰度直方图", linestyle="--", color='r')
    # plt.legend()
    # # plt.show()
    #
    # plt.subplot(2, 2, 3)
    # plt.imshow(processing_pic)
    # plt.subplot(2, 2, 4)
    # plt.imshow(pic_EH)
    # plt.savefig("pic_enhance_effect.jpg")
    # plt.show()



def save_img_data(dep=None, data=None, path=''):
    if dep is None:
        dep = np.arange(data.shape[0])

    # assert dep.size == data.shape[0]
    dep = np.reshape(dep, (-1, 1))
    data = np.hstack((dep, data))

    np.savetxt(path, data, fmt='%.4f', delimiter='\t', comments='',
               header='WELLNAME={}\nSTDEP\t= {}\nENDEP\t= {}\nLEV\t= {:.4f}\nUNIT\t= meter\nCURNAMES= {}\n#DEPTH\t{}'.format(
                   'Temp_well', dep[0, 0], dep[-1, 0], dep[1, 0]-dep[0, 0], 'Img_data', 'Img_data'))


# 线性变换的原理是对所有像素值乘上一个扩张因子 factor
# 像素值大的变得越大，像素值小的变得越小，从而达到图像增强的效果，这里利用 Numpy 的数组进行操作；
def line_trans_img(img,coffient):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    out = 2*img
    #像素截断；；；
    out[out>255] = 255
    out = np.around(out)
    return out


# 图像对比度计算
def contrast(img1):
    m, n = img1.shape
    # 图片矩阵向外扩展一个像素
    img1_ext = cv2.copyMakeBorder(img1,1,1,1,1,cv2.BORDER_REPLICATE) / 1.0   # 除以1.0的目的是uint8转为float型，便于后续计算
    rows_ext,cols_ext = img1_ext.shape
    b = 0.0
    for i in range(1,rows_ext-1):
        for j in range(1,cols_ext-1):
            b += ((img1_ext[i,j]-img1_ext[i,j+1])**2 + (img1_ext[i,j]-img1_ext[i,j-1])**2 +
                    (img1_ext[i,j]-img1_ext[i+1,j])**2 + (img1_ext[i,j]-img1_ext[i-1,j])**2)

    cg = b/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4) #对应上面48的计算公式
    # cg = b/(m*n)
    # print(cg)
    return cg

#计算峰值信噪比
def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# 计算图像信息熵
def comentropy(img):
    # img = cv2.imread('20201210_3.bmp',0)
    # img = np.zeros([16,16]).astype(np.uint8)
    img = np.array(img).astype(np.uint8)
    m, n = img.shape

    hist_cv = cv2.calcHist([img], [0], None, [256], [0, 256])  # [0,256]的范围是0~255.返回值是每个灰度值出现的次数

    P = hist_cv / (m * n)  # 概率
    E = np.sum([p * np.log2(1 / p) for p in P if p>0])
    return E


# from skimage.metrics import structural_similarity
# from skimage.metrics import peak_signal_noise_ratio

def pic_smooth_effect_compare():

    # import cv2 as cv
    # import numpy as np
    from matplotlib import pyplot as plt
    # %matplotlib inline
    # img = cv.imread('opencv-logo-white.png')
    os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
    plt.rcParams['font.family'] = 'SimHei'
    data_img_dyna, data_img_stat, data_depth = get_test_ele_data()
    print('data_image shape:{}'.format(data_img_dyna.shape))

    data_img = cv2.resize(data_img_dyna,(256, 256))

    # index_1, index_2 = 1400, 1800
    # img = np.uint8(data_img[index_1:index_2, :])
    # dep_img = data_depth[index_1:index_2, :]

    img = np.uint8(data_img)
    ret, img = cv2.threshold(img, 200+np.random.randint(0, 20)-10, 255, cv2.THRESH_BINARY_INV)
    print(img.shape)
    avg_blur = cv2.blur(img, (5, 5))
    guass_blur = cv2.GaussianBlur(img, (5, 5), 0)
    median_blur = cv2.medianBlur(img, 5)
    pic_bilateral_filter = cv2.bilateralFilter(img, 9, 75, 75)

    windows_shape = [3, 5, 7, 9]
    ratio_mig = [0.4, 0.6, 0.6, 0.6]
    random_times = [1, 1, 1, 1]

    pic_EH_3 = pic_enhence_random(img, windows_shape=windows_shape[0], ratio_migration=ratio_mig[0], random_times=random_times[0])
    # pic_EH_5 = pic_enhence_random(img, windows_shape=windows_shape[1], ratio_migration=ratio_mig[1], random_times=random_times[1])
    # pic_EH_7 = pic_enhence_random(img, windows_shape=windows_shape[2], ratio_migration=ratio_mig[2], random_times=random_times[2])
    # pic_EH_9 = pic_enhence_random(img, windows_shape=windows_shape[3], ratio_migration=ratio_mig[3], random_times=random_times[3])

    # 对比不同参数 随机偏移 的图像增强效果
    # show_Pic([img, pic_EH_3, pic_EH_5, pic_EH_7], pic_order='22',
    #          pic_str=['原始电成像图像', '像素值偏移增图像效果:n=5', '像素值偏移增图像效果:n=7', '像素值偏移增图像效果:n=9'])

    # 直方图均衡化
    pic_equalizeHist = cv2.equalizeHist(img)

    # 对图像进行局部直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))  # 对图像进行分割，10*10
    pic_local_equalizeHist = clahe.apply(img)  # 进行直方图均衡化

    # gama 伽马变换
    imgGrayNorm = img / 255
    gamma = 0.8
    pic_gamma_transf = (np.power(imgGrayNorm, gamma) * 256).astype(np.uint8)

    # print(contrast(img), contrast(pic_EH_3), contrast(pic_gamma_transf), contrast(pic_equalizeHist))
    # print(psnr(img, img), psnr(img, pic_EH_3), psnr(img, pic_gamma_transf), psnr(img, pic_equalizeHist))
    # print(comentropy(img), comentropy(pic_EH_3), comentropy(pic_gamma_transf), comentropy(pic_equalizeHist))

    # num_sp = 20
    # pixel_per_window = img.shape[0]//num_sp
    # E1 = []
    # E2 = []
    # E3 = []
    # E4 = []
    # for i in range(num_sp):
    #     for j in range(num_sp):
    #         pic_temp = img[i*pixel_per_window:(i+1)*pixel_per_window]
    #         E1.append(comentropy(pic_temp))
    #         pic_temp = pic_EH_3[i*pixel_per_window:(i+1)*pixel_per_window]
    #         E2.append(comentropy(pic_temp))
    #         pic_temp = pic_gamma_transf[i*pixel_per_window:(i+1)*pixel_per_window]
    #         E3.append(comentropy(pic_temp))
    #         pic_temp = pic_equalizeHist[i*pixel_per_window:(i+1)*pixel_per_window]
    #         E4.append(comentropy(pic_temp))
    #
    # print(np.mean(np.array(E1)), np.mean(np.array(E2)), np.mean(np.array(E3)), np.mean(np.array(E4)))

    # print(comentropy(img), comentropy(pic_EH_3), comentropy(pic_gamma_transf), comentropy(pic_equalizeHist))

    pic_equalizeHist = pic_open_close_random(pic_equalizeHist)
    pic_local_equalizeHist = pic_open_close_random(pic_local_equalizeHist)
    pic_bilateral_filter = pic_open_close_random(pic_bilateral_filter)
    pic_EH_3 = pic_open_close_random(pic_EH_3)
    pic_gamma_transf = pic_open_close_random(pic_gamma_transf)

    # cv2.imwrite('pic_equalizeHist.png', traverse_pic(pic_equalizeHist))
    # # cv2.imwrite('pic_local_equalizeHist.png', traverse_pic(pic_local_equalizeHist))
    # # cv2.imwrite('pic_bilateral_filter.png', traverse_pic(pic_bilateral_filter))
    # cv2.imwrite('pic_EH_3.png', traverse_pic(pic_EH_3))
    # # cv2.imwrite('pic_gamma_transf.png', traverse_pic(pic_gamma_transf))

    show_Pic([traverse_pic(data_img), traverse_pic(pic_equalizeHist), traverse_pic(pic_local_equalizeHist),
              traverse_pic(pic_gamma_transf), traverse_pic(pic_bilateral_filter), traverse_pic(pic_EH_3)], pic_order='23',
             pic_str=['原始图像', '直方图均衡', '局部直方图均衡', '伽马变换', '双边滤波', '随机偏移增强'])


    # cv2.calcHist(images, channels, mask, histSize, ranges, hist, accumulate)
    # mask: 掩模图像。要统计整幅图像的直方图就把它设为 None。但是如 果你想统计图像某一部分的直方图的话，你就需要制作一个掩模图像，并 使用它。（后边有例子）
    # histSize：BIN 的数目。也应该用中括号括起来，例如：[256]。 5. ranges: 像素值范围，通常为 [0，256]
    # hist：是一个 256x1 的数组作为返回值，每一个值代表了与次灰度值对应的像素点数目。
    # accumulate：是一个布尔值，用来表示直方图是否叠加。
    # hist_org = cv2.calcHist([img], [0], None, [256], [0, 256])/img.size
    # hist_equalize = cv2.calcHist([pic_equalizeHist], [0], None, [256], [0, 256])/img.size
    # hist_local_equalize = cv2.calcHist([pic_local_equalizeHist], [0], None, [256], [0, 256])/img.size
    # hist_gama = cv2.calcHist([pic_gamma_transf], [0], None, [256], [0, 256])/img.size
    # hist_bil_blur = cv2.calcHist([pic_bilateral_filter], [0], None, [256], [0, 256])/img.size
    # hist_random_shift = cv2.calcHist([pic_EH_3], [0], None, [256], [0, 256])/img.size


    # # Draw Plot
    # # cut：参数表示绘制的时候，切除带宽往数轴极限数值的多少(默认为3)
    # # cumulative ：是否绘制累积分布，默认为False
    # # fill：若为True，则在kde曲线下面的区域中进行阴影处理，color控制曲线及阴影的颜色
    # # vertical：表示以X轴进行绘制还是以Y轴进行绘制
    # # label="原始成像"
    # # plt.figure(figsize=(10, 8), dpi=80)
    # # sns.kdeplot(img.ravel(), cut=0, fill=True, color="#01a2d9", alpha=.7).set(xlabel="Percentage", ylabel="Pixel distribution")
    # # sns.kdeplot(pic_equalizeHist.ravel(), fill=True, color="#dc2624", label="Cyl=5", alpha=.7)
    # # sns.kdeplot(pic_local_equalizeHist.ravel(), fill=True, color="#C89F91", label="Cyl=6", alpha=.7)
    # # sns.kdeplot(pic_gamma_transf.ravel(), fill=True, color="#649E7D", label="Cyl=8", alpha=.7)
    # # sns.kdeplot(pic_bilateral_filter.ravel(), fill=True, color="#649E7D", label="Cyl=8", alpha=.7)
    # # sns.kdeplot(pic_EH_3.ravel(), fill=True, color="#649E7D", label="Cyl=8", alpha=.7)
    #
    # sns.set_style(style="white")
    # sns.despine(top=True, right=True, left=False, bottom=False)
    # # sns.distplot(img.ravel(), bins=20).set(xlabel="Percentage", ylabel="Pixel distribution")
    # # sns.distplot(pic_EH_3.ravel(), bins=20).set(xlabel="Percentage", ylabel="Pixel distribution")
    # # sns.distplot(pic_gamma_transf.ravel(), bins=20).set(xlabel="Percentage", ylabel="Pixel distribution")
    # sns.distplot(pic_equalizeHist.ravel(), bins=20).set(xlabel="Percentage", ylabel="Pixel distribution")
    #
    # # sns.histplot(img.ravel(), color='#01a2d9').set(xlabel="Percentage", ylabel="Pixel distribution")
    # # sns.displot(img.ravel(), color='#01a2d9').set(xlabel="Percentage", ylabel="Pixel distribution")
    #
    # # plt.title('Density Plot of City Mileage by n_Cylinders', fontsize=18)
    # # plt.legend('legend')
    # plt.xlim([0, 256])
    # plt.ylim([0, 0.01])
    # plt.show()

    # # plt.subplot(2, 2, 1)
    # plt.subplot(1, 1, 1)
    # # plt.plot(hist_o, label="原始电成像图像灰度直方图", linestyle="--", color='g')
    # plt.plot(hist_org, linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.show()

    # # plt.subplot(2, 2, 2)
    # plt.subplot(1, 1, 1)
    # # plt.plot(hist_EH, label="随即迁移增强后的灰度直方图", linestyle="--", color='r')
    # plt.plot(hist_EH, linestyle="--", color='r')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.show()

    # # plt.subplot(2, 2, 3)
    # plt.subplot(1, 1, 1)
    # # plt.plot(hist_gamma, label="伽马变换增强后的灰度直方图", linestyle="--", color='g')
    # plt.plot(hist_gamma, linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.show()

    # # plt.subplot(2, 2, 4)
    # plt.subplot(1, 1, 1)
    # # plt.plot(hist_equalize_hist, label="直方图均衡增强后的灰度直方图", linestyle="--", color='g')
    # plt.plot(hist_equalize_hist, linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.savefig("pic_enhance_effect.jpg")
    # plt.show()



    # hist_o = cv2.calcHist([np.uint8(img)], [0], None, [256], [0, 256])/img.size
    # hist_EH = cv2.calcHist([np.uint8(pic_EH_3)], [0], None, [256], [0, 256])/img.size
    # hist_gamma = cv2.calcHist([np.uint8(pic_gamma_transf)], [0], None, [256], [0, 256])/img.size
    # # hist_gamma_1 = cv2.calcHist([np.uint8(pic_gamma_transf_1)], [0], None, [256], [0, 256])/img.size
    # hist_equalize_hist = cv2.calcHist([np.uint8(pic_equalizeHist)], [0], None, [256], [0, 256])/img.size

    # plt.subplot(2, 2, 1)
    # plt.plot(hist_o, label="原始电成像图像灰度直方图", linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.subplot(2, 2, 2)
    # plt.plot(hist_EH, label="随机迁移增强后的灰度直方图", linestyle="--", color='r')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.subplot(2, 2, 3)
    # plt.plot(hist_gamma, label="伽马变换增强后的灰度直方图", linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.subplot(2, 2, 4)
    # plt.plot(hist_equalize_hist, label="直方图均衡增强后的灰度直方图", linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # # plt.savefig("pic_enhance_effect.jpg")
    # plt.show()

# pic_smooth_effect_compare()




# 对二维数据 进行 简单的数据缩放
def pic_scale_simple(pic=np.array([]), pic_shape=[0,0]):
    if len(pic.shape) >= 3:
        print('only process two dim pic& pic shape is:{}'.format(pic.shape))
        exit(0)

    if pic_shape[0] <= 0:
        print('shape error...:{}'.format(pic_shape))
        exit(0)
    elif (pic_shape[0]>1) & (pic_shape[1]>1):
        x_size = pic_shape[0]
        y_size = pic_shape[1]
    elif (pic_shape[0] <= 1) & (pic_shape[0] > 0) & (pic_shape[1] <= 1) & (pic_shape[1] > 0):
        x_size = int(pic_shape[0] * pic.shape[0])
        y_size = int(pic_shape[1] * pic.shape[1])
    elif (pic_shape[0] > pic.shape[0]) | (pic_shape[1] > pic.shape[1]):
        print('target pic shape is {},org shape is {}'.format(pic_shape, pic.shape))
        exit(0)
    else:
        print('pic shape error:{}'.format(pic_shape))
        exit(0)

    pic_new = np.zeros((x_size, y_size))
    for i in range(x_size):
        for j in range(y_size):
            index_x = int(i/x_size*pic.shape[0])
            index_y = int(j/y_size*pic.shape[1])
            pic_new[i][j] = pic[index_x][index_y]

    return pic_new


def pic_repair_normal(pic, windows_l=5):
    PicDataWhiteStripe = np.zeros_like(pic)

    # 手动空白带提取
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            if pic[i][j] <= 0.09:
                PicDataWhiteStripe[i][j] = 255
            else:
                PicDataWhiteStripe[i][j] = 0
    # 空白带提取
    # ret, PicDataWhiteStripe = cv2.threshold(pic, 0, 1, cv2.THRESH_BINARY_INV)

    PicDataWhiteStripe = np.uint8(PicDataWhiteStripe)
    pic = np.uint8(pic)

    # TELEA 图像修复
    PIC_Repair_dst_TELEA = cv2.inpaint(pic, PicDataWhiteStripe, windows_l, cv2.INPAINT_TELEA)
    # NS 图像修复
    PIC_Repair_dst_NS = cv2.inpaint(pic, PicDataWhiteStripe, windows_l, cv2.INPAINT_NS)

    return PIC_Repair_dst_TELEA, PIC_Repair_dst_NS, PicDataWhiteStripe


def pic_seg_by_kai_bi():
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG701_126_5183.0000_5183.6600_dyna.png'

    pic = cv2.imread(path_in, cv2.IMREAD_GRAYSCALE)
    print(pic.shape)

    ProcessingPic = copy.deepcopy(pic)

    Blur_Average = cv2.blur(ProcessingPic, (7, 5))
    Blur_Gauss = cv2.GaussianBlur(ProcessingPic, (7, 5), 0)
    Blur_Median = cv2.medianBlur(ProcessingPic, 5)

    firstLevel = 40
    ret, img_binary_Level1 = cv2.threshold(ProcessingPic, firstLevel, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level2 = cv2.threshold(ProcessingPic, firstLevel + 10, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level3 = cv2.threshold(ProcessingPic, firstLevel + 20, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level4 = cv2.threshold(ProcessingPic, firstLevel + 30, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level5 = cv2.threshold(ProcessingPic, firstLevel + 40, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level6 = cv2.threshold(ProcessingPic, firstLevel + 130, 255, cv2.THRESH_BINARY)
    ProcessingPic = img_binary_Level6

    # cv2.THRESH_BINARY：二值阈值处理，只有大于阈值的像素值为最大值，其他像素值为最小值。
    # cv2.THRESH_BINARY_INV：反二值阈值处理，只有小于阈值的像素值为最大值，其他像素值为最小值。
    # cv2.THRESH_TRUNC：截断阈值处理，大于阈值的像素值被赋值为阈值，小于阈值的像素值保持原值不变。
    # cv2.THRESH_TOZERO：置零阈值处理，只有大于阈值的像素值被置为0，其他像素值保持原值不变。
    # cv2.THRESH_TOZERO_INV：反置零阈值处理，只有小于阈值的像素值被置为0，其他像素值保持原值不变。

    Kernel_Rect = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # 生成形状为矩形5x5的卷积核
    Kernel_Rect2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    Kernel_Rect3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    Kernel_Ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 3))  # 椭圆形9x9
    kernel = np.ones((5, 5), np.uint8)
    targetKernel = Kernel_Rect3


    Pic_erosion = cv2.erode(ProcessingPic, targetKernel, iterations=1)
    Pic_dilation = cv2.dilate(ProcessingPic, targetKernel, iterations=1)
    Pic_opening = cv2.morphologyEx(ProcessingPic, cv2.MORPH_OPEN, targetKernel)
    Pic_closing = cv2.morphologyEx(ProcessingPic, cv2.MORPH_CLOSE, targetKernel)
    Pic_opening_closing = cv2.morphologyEx(Pic_opening, cv2.MORPH_CLOSE, targetKernel)
    Pic_closing_opening = cv2.morphologyEx(Pic_closing, cv2.MORPH_OPEN, targetKernel)

    ProcessingPic = Pic_opening_closing

    contours_Conform, contours_Drop, contours_All = GetPicContours(ProcessingPic, threshold=500)
    img_white = np.zeros((ProcessingPic.shape[0], ProcessingPic.shape[1]), np.uint8)
    img_white2 = np.zeros((ProcessingPic.shape[0], ProcessingPic.shape[1]), np.uint8)
    # img_white = np.zeros_like(ProcessingPic).astype(np.uint8).fill(0)
    # img_white = copy.deepcopy(ProcessingPic).astype(np.uint8).fill(0)
    print(ProcessingPic.shape[0], ProcessingPic.shape[1])
    # print(img_white)
    cv2.drawContours(img_white, contours_Conform[1], -1, 255, thickness=-1)
    # print(img_mask)

    # show_Pic([pic, ProcessingPic, img_white, img_white2], pic_order='14', pic_str=[], save_pic=False)

    cv2.imwrite(path_in.replace('dyna', 'mask2'), img_white)
    cv2.imwrite(path_in.replace('dyna', 'mask'), ProcessingPic)


def cal_pic_generate_effect(pic_org, pic_repair):
    # print(pic_org.shape, pic_repair.shape)
    # 计算PSNR：
    PSNR = peak_signal_noise_ratio(pic_org, pic_repair)
    # 计算SSIM
    SSIM = structural_similarity(pic_org, pic_repair)
    # 计算MSE 、 RMSE、 MAE、r2
    mse = np.sum((pic_org - pic_repair) ** 2) / pic_org.size
    rmse = math.sqrt(mse)
    mae = np.sum(np.absolute(pic_org - pic_repair)) / pic_org.size
    r2 = 1 - mse / np.var(pic_org)  # 均方误差/方差

    Entropy_org = comentropy(pic_org)
    Entropy_vice = comentropy(pic_repair)

    Con_org = contrast(pic_org)
    Con_vice = contrast(pic_repair)

    return PSNR, SSIM, mse, rmse, mae, r2, Entropy_org, Entropy_vice, Con_org, Con_vice






# 根据空白条带参数config设置，获得随机的图像空白带掩码mask
def get_pic_mask_random(pic_shape=(256, 256), mask_ratio=0.2, num_belt = np.random.randint(2, 4) * 2):
    """
    生成随机的电成像空白条带掩码

    参数:
    - pic_shape: 目标图像形状 (高度, 宽度)
    - mask_ratio: 空白区域占总宽度的比例 (0-1)
    - num_belt: 空白条带数量 (极板数量)，如果为None则随机生成

    返回:
    - mask: 空白条带掩码，1表示保留区域，0表示空白区域

    算法说明:
    1. 将图像宽度均匀分配给各个极板
    2. 在每个极板上创建指定宽度的空白区域
    3. 生成对应的掩码矩阵

    数据检查点:
    - 验证输入参数的有效性
    - 检查生成的掩码形状是否正确
    - 确认空白区域比例符合预期
    """
    # 数据检查1: 验证输入参数
    if not isinstance(pic_shape, (tuple, list)) or len(pic_shape) != 2:
        raise ValueError(f"pic_shape应为包含两个元素的元组或列表，当前为: {pic_shape}")

    if not (0 <= mask_ratio <= 1):
        raise ValueError(f"mask_ratio应在0-1范围内，当前为: {mask_ratio}")

    # 如果未指定极板数量，则随机生成
    if num_belt is None:
        num_belt = np.random.randint(2, 4) * 2  # 生成偶数个极板: 4, 6, 8

    # 数据检查2: 验证极板数量有效性
    if num_belt <= 0 or num_belt > pic_shape[1]:
        raise ValueError(f"极板数量应在1到图像宽度之间，当前为: {num_belt}, 图像宽度: {pic_shape[1]}")

    num_mask = int(mask_ratio * pic_shape[-1])      # 256*0.25=64   64像素的空白带

    # 数据检查3: 确保空白像素数合理
    if num_mask <= 0:
        print(f"警告: 空白像素数为0，mask_ratio可能过小: {mask_ratio}")
        return np.ones(pic_shape, dtype='float32')

    if num_mask >= pic_shape[1]:
        print(f"警告: 空白像素数超过图像宽度，将生成全空白掩码")
        return np.zeros(pic_shape, dtype='float32')

    pix_skip = pic_shape[-1]//num_belt               # 每个极板上 一共256/8=32个像素点
    mask_belt_width = num_mask//num_belt                 # 每个极板上 一共64/8=8个的空白像素点

    num_belt_para = []                              # 极板空白带配置
    for i in range(num_belt):
        index_start = i * pix_skip
        index_end = i*pix_skip+mask_belt_width
        num_belt_para.append([index_start, index_end])

    # print('all mask pixel:{}, num belt:{}, pixel per skip:{}, mask width per belt:{}'.format(num_mask, num_belt, pix_skip, mask_belt_width))

    mask = np.ones(pic_shape, dtype='float32')
    for i in range(len(num_belt_para)):
        mask[:, num_belt_para[i][0]: num_belt_para[i][1]] = 0

    return mask


# 获取随机的方位角曲线，为了接下来的进行图像绕井壁旋转
def get_random_RB_curve(depth, start_angle=None):
    """
        生成随机的方位角(RB)曲线

        参数:
        - depth: 深度数据，用于确定曲线长度
        - start_angle: 起始方位角，如果为None则随机生成

        返回:
        - Rb_random: 随机生成的方位角曲线

        算法说明:
        1. 从随机起始角度开始
        2. 在每个深度点进行随机角度变化（受限变化幅度）
        3. 确保角度在[-180, 180]范围内

        数据检查点:
        - 验证深度数据有效性
        - 检查生成的曲线角度范围
        - 确认曲线长度与深度数据匹配
    """
    # 数据检查1: 验证深度数据
    if depth is None or len(depth) == 0:
        raise ValueError("深度数据不能为空")

    # 随机生成起始角度（如果未提供）
    if start_angle is None:
        start_angle = np.random.randint(-60, 60)

    max_rotate_angle = 1                # 最大单步旋转角度
    Rb_random = np.zeros(depth.shape)
    for i in range(depth.shape[0]):
        # 生成随机旋转角度（-0.5到0.5度）
        rotate_angle = (np.random.random()-0.5)*max_rotate_angle
        if i == 0:
            Rb_random[i][0] = start_angle
        else:
            # 计算新角度并限制在[-180, 180]范围内
            new_angle = Rb_random[i-1][0] + rotate_angle
            Rb_random[i][0] = max(min(new_angle, 180), -180)

    return Rb_random

# 根据RB曲线进行图像旋转
def pic_rotate_by_Rb(pic=np.zeros((10, 10)), Rb=np.zeros((10, 1))):
    """
        根据方位角曲线旋转图像

        参数:
        - pic: 输入图像 (高度, 宽度)
        - Rb: 方位角曲线 (高度, 1)

        返回:
        - pic_new: 旋转后的图像

        算法说明:
        1. 将方位角转换为像素位移量
        2. 对每一行图像进行循环平移
        3. 实现绕井壁的旋转效果

        数据检查点:
        - 验证图像和方位角曲线尺寸匹配
        - 检查旋转后的图像数据完整性
        - 确认旋转操作不会丢失图像信息
    """
    # 数据检查1: 验证输入数据形状匹配
    if pic.shape[0] != Rb.shape[0]:
        error_msg = f"图像行数 {pic.shape[0]} 与方位角曲线长度 {Rb.shape[0]} 不匹配"
        print(f"错误: {error_msg}")
        raise ValueError(error_msg)

    # 数据检查2: 验证图像和方位角数据有效性
    if pic.size == 0 or Rb.size == 0:
        print("警告: 输入图像或方位角曲线为空")
        return pic.copy() if pic.size > 0 else pic

    # 创建输出图像
    pic_new = np.zeros(pic.shape)
    temp = 360/pic.shape[1]         # 每个像素对应的角度
    # 对每一行应用旋转
    for i in range(pic.shape[0]):
        # 计算像素位移量（四舍五入到最近的整数）
        pixel_rotate = int(round(Rb[i][0] / temp))

        # 数据检查3: 验证旋转像素数合理性
        if abs(pixel_rotate) > pic.shape[1]:
            print(f"警告: 行 {i} 的旋转像素数 {pixel_rotate} 过大，图像宽度: {pic.shape[1]}")
            pixel_rotate = pixel_rotate % pic.shape[1]  # 取模限制在合理范围

        # 应用循环平移
        if pixel_rotate != 0:
            pic_new[i, pixel_rotate:] = pic[i, :-pixel_rotate]
            pic_new[i, :pixel_rotate] = pic[i, -pixel_rotate:]
        else:
            pic_new[i, :] = pic[i, :]

    return pic_new

# 图像 生成随机RB曲线 并旋转
def pic_rotate_random(pic=np.zeros((5, 5)), depth=None, ratio=None):
    """
        随机旋转图像

        参数:
        - pic: 输入图像
        - depth: 深度数据（用于生成方位角曲线）
        - ratio: 旋转概率，如果为None则随机生成

        返回:
        - pic_new: 旋转后的图像（可能与原图相同）
        - rb_random: 使用的方位角曲线

        数据检查点:
        - 验证输入图像有效性
        - 检查深度数据与图像匹配
        - 确认旋转操作的正确性
    """
    # 数据检查1: 验证输入图像
    if pic is None or pic.size == 0:
        print("警告: 输入图像为空")
        return pic, np.zeros((0, 1)) if depth is None else np.zeros_like(depth)

    # 处理默认参数
    if depth is None:
        depth = np.zeros((pic.shape[0], 1))
    if ratio is None:
        ratio = np.random.random()

    # 数据检查2: 验证深度数据与图像行数匹配
    if depth.shape[0] != pic.shape[0]:
        print(f"警告: 深度数据长度 {depth.shape[0]} 与图像行数 {pic.shape[0]} 不匹配")
        # 调整深度数据长度
        if depth.shape[0] > pic.shape[0]:
            depth = depth[:pic.shape[0]]
        else:
            depth = np.pad(depth, ((0, pic.shape[0] - depth.shape[0]), (0, 0)), mode='edge')

    # 根据概率决定是否旋转
    if ratio < 0.8:
        rb_random = get_random_RB_curve(depth)
        pic_new = pic_rotate_by_Rb(pic, rb_random)
    else:
        rb_random = np.zeros((pic.shape[0], 1))
        pic_new = pic.copy()  # 创建副本，避免修改原图

    pic_new = pic_rotate_by_Rb(pic, rb_random)
    return pic_new, rb_random


# 输入图像list，但是这些图像的形状必须完全一样，否则行不通，并对图像进行添加随机的mask操作
def pic_list_add_random_stripe(image_list):
    """
        为图像列表添加随机空白条带

        参数:
        - image_list: 图像列表，所有图像必须具有相同形状

        返回:
        - masked_list: 添加空白条带后的图像列表
        - mask_list: 空白条带掩码列表

        数据检查点:
        - 验证输入图像列表有效性
        - 检查所有图像形状一致性
        - 确认生成的掩码有效性
        - 验证处理后的图像数据完整性
    """
    # 数据检查1: 验证输入参数
    if not image_list or len(image_list) == 0:
        print("警告: 图像列表为空")
        return [], []

    # 检查所有图像形状是否一致
    image_shape_list = []
    for i, image in enumerate(image_list):
        if image is None:
            print(f"错误: 图像列表中的第 {i} 个元素为None")
            return [], []
        image_shape_list.append(image.shape)

    # 数据检查2: 验证所有图像形状一致
    if len(set(image_shape_list)) > 1:
        print(f"警告: 图像形状不一致: {image_shape_list}")
        raise ValueError("图像形状不一致")
        # 这里可以添加图像调整逻辑或抛出异常

    first_image_shape = image_list[0].shape

    # 空白率设置
    ratio_empty = np.random.choice([0.25, 0.3, 0.3, 0.35, 0.35, 0.4, 0.4, 0.45, 0.45])
    # 极板个数设置
    num_belt = np.random.choice([6, 6, 6, 8, 8, 10])
    # 生成空白条带的掩码图像， [0, 1]
    mask = get_pic_mask_random(pic_shape=first_image_shape, mask_ratio=ratio_empty, num_belt=num_belt)

    # 数据检查4: 验证生成的掩码
    if mask is None or mask.shape != first_image_shape:
        print(f"错误: 生成的掩码形状 {mask.shape if mask is not None else 'None'} 与预期 {first_image_shape} 不匹配")
        # 创建默认掩码（全1）
        mask = np.ones(first_image_shape, dtype='float32')

    # 空白条带掩码图像的随机绕井壁旋转,对掩码进行随机旋转
    mask, rb_random = pic_rotate_random(mask)

    masked_list = []
    mask_list = []
    for i, image in enumerate(image_list):
        # 数据检查6: 验证单个图像
        if image is None:
            print(f"警告: 第 {i} 个图像为None，跳过处理")
            masked_list.append(None)
            mask_list.append(None)
            continue

        if image.shape != first_image_shape:
            print(f"警告: 第 {i} 个图像形状 {image.shape} 与第一个图像 {first_image_shape} 不一致")
            # 这里可以添加图像调整逻辑

        # 这个是图像被遮盖后的图像 # 应用掩码：保留区域 = 原图 × 掩码
        image_masked = mask * image
        # 这个是图像遮盖的部分    # 空白区域 = 原图 × (1 - 掩码)
        image_mask = (1 - mask) * image
        masked_list.append(image_masked)
        mask_list.append(image_mask)

    return masked_list, mask_list

if __name__ == '__main__':
    dyna_data, stat_data, depth = get_random_fmi()

    dyna_stripe_list, image_mask_list = pic_list_add_random_stripe([dyna_data])
    show_Pic([dyna_data, dyna_stripe_list[0], image_mask_list[0]])