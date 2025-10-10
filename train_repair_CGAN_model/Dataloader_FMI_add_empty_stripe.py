import cv2
import numpy as np
from torch.utils.data import Dataset
from src_ele.dir_operation import get_all_file_paths
from src_ele.pic_opeeration import show_Pic


# 获取随机的方位角曲线，为了接下来的进行图像绕井壁旋转
def get_random_RB_curve_1(depth, start_angle=None):
    if start_angle is None:
        start_angle = np.random.randint(-60, 60)

    max_rotate_angle = 1
    Rb_random = np.zeros(depth.shape)
    for i in range(depth.shape[0]):
        rotate_angle = (np.random.random()-0.5)*max_rotate_angle
        if i == 0:
            Rb_random[i][0] = start_angle
        else:
            Rb_random[i][0] = max(min(Rb_random[i-1][0] + rotate_angle, 180), -180)
    return Rb_random

# 根据RB曲线进行图像旋转
def pic_rotate_by_Rb(pic=np.zeros((10, 10)), Rb=np.zeros((10, 1))):
    if pic.shape[0] != Rb.shape[0]:
        print('pic length is not equal to depth length:{}, {}'.format(pic.shape, Rb.shape))
        exit(0)

    pic_new = np.zeros(pic.shape)
    temp = 360/pic.shape[1]
    for i in range(pic.shape[0]):
        pixel_rotate = int(Rb[i][0] / temp)
        if pixel_rotate != 0:
            pic_new[i, pixel_rotate:] = pic[i, :-pixel_rotate]
            pic_new[i, :pixel_rotate] = pic[i, -pixel_rotate:]
        else:
            pic_new[i, :] = pic[i, :]

    return pic_new

# 图像 生成随机RB曲线 并旋转
def pic_rotate_random(pic=np.zeros((5, 5)), depth=None, ratio=None):
    if depth is None:
        depth = np.zeros((pic.shape[0], 1))
    if ratio is None:
        ratio = np.random.random()

    if ratio < 0.8:
        rb_random = get_random_RB_curve_1(depth)
    else:
        rb_random = np.zeros((pic.shape[0], 1))
        return pic, rb_random

    pic_new = pic_rotate_by_Rb(pic, rb_random)
    return pic_new, rb_random

# 根据空白条带参数config设置，获得随机的图像空白带掩码mask
def get_pic_mask_random(pic_shape=(256, 256), mask_ratio=0.2, num_belt = np.random.randint(3, 4) * 2):
    num_mask = int(mask_ratio * pic_shape[-1])      # 256*0.25=64   64像素的空白带

    pix_skip = pic_shape[-1]//num_belt               # 每个极板上 一共256/8=32个像素点
    mask_belt_width = num_mask//num_belt                 # 每个极板上 一共64/8=8个的空白像素点

    num_belt_para = []                              # 极板空白带配置
    for i in range(num_belt):
        index_start = i * pix_skip
        index_end = i*pix_skip+mask_belt_width
        num_belt_para.append([index_start, index_end])

    print('all mask pixel:{}, num belt:{}, pixel per skip:{}, mask width per belt:{}'.format(num_mask, num_belt, pix_skip, mask_belt_width))

    mask = np.ones(pic_shape)
    for i in range(len(num_belt_para)):
        mask[:, num_belt_para[i][0]: num_belt_para[i][1]] = 0

    return mask

# 电成像FMI图像，左右两边指定像素的Padding
def FMI_padding(image, padding=32):
    image_n = np.zeros((image.shape[0], image.shape[1]+2*padding), dtype=np.uint8)
    image_n[:, padding: -padding] = image
    image_n[:, :padding] = image[:, -padding:]
    image_n[:, -padding:] = image[:, :padding]
    return image_n

# 随机的图像裁剪，随机裁剪图像的一部分作为数据输入
def pic_crop_random(pic, pic_shape_ratio=None):
    # 设置随机裁剪的图像大小
    if pic_shape_ratio is None:
        pic_shape_ratio = 0.6 + 0.4 * np.random.random()

    pic_h = int(pic.shape[0] * pic_shape_ratio)
    pic_w = int(pic.shape[1] * pic_shape_ratio)
    start_h = np.random.randint(0, pic.shape[0]-pic_h-1)
    start_w = np.random.randint(0, pic.shape[1]-pic_w-1)

    return pic[start_h:start_h+pic_h, start_w:start_w+pic_w]


# 实井电成像+模拟电成像+常规图像数据的 空白条带图像生成
class dataloader_padding_striped(Dataset):
    def __init__(self, path=r'F:\DeepLData\target_stage1_small_big_mix', len_pic=256, padding=16):
        super().__init__()
        self.target_list_file_path = get_all_file_paths(path)
        self.length = len(self.target_list_file_path)
        self.pic_shape = (len_pic, len_pic)
        self.padding = padding

    def __getitem__(self, index):
        path_t = self.target_list_file_path[index]
        # 常规图片的 空白条带图像生成
        if path_t.__contains__('REALL_WORD_PIC'):
            image_origin = cv2.imread(self.target_list_file_path[index], cv2.IMREAD_COLOR_RGB)

            # 常规的图像的话，随机选择一个channel，不使用所有的channel
            channel = np.random.randint(0, 3)
            image_origin = image_origin[:, :, channel]
            image_origin = pic_crop_random(image_origin)

            # 空白率设置
            ratio_empty = np.random.choice([0.25, 0.3, 0.3, 0.35, 0.35, 0.4, 0.4, 0.45, 0.45])
            # 极板个数设置
            num_belt = np.random.choice([6, 6, 6, 8, 8, 10])
            # 生成空白条带的掩码图像， [0, 1]
            mask = get_pic_mask_random(pic_shape=image_origin.shape, mask_ratio=ratio_empty, num_belt=num_belt)
            # 空白条带掩码图像的随机绕井壁旋转
            mask, rb_random = pic_rotate_random(mask)
            image_masked = mask * image_origin
            image_to_repaired = (1 - mask) * image_origin

        # 电成像数据的空白条带图像生成
        elif path_t.__contains__('SIMULATED_FMI_SPLIT') or path_t.__contains__('ZG_FMI_SPLIT'):
            image_origin = cv2.imread(self.target_list_file_path[index], cv2.IMREAD_GRAYSCALE)

            # 空白率设置
            ratio_empty = np.random.choice([0.25, 0.3, 0.3, 0.35, 0.35, 0.4, 0.4, 0.45, 0.45])
            # 极板个数设置
            num_belt = np.random.choice([6, 6, 6, 8, 8, 10])
            # 生成空白条带的掩码图像， [0, 1]
            mask = get_pic_mask_random(pic_shape=image_origin.shape, mask_ratio=ratio_empty, num_belt=num_belt)
            # 空白条带掩码图像的随机绕井壁旋转
            mask, rb_random = pic_rotate_random(mask)
            # 生成模型输入数据
            image_masked = mask * image_origin
            # 这个是遮盖的部分，也是模型要预测的一部分
            image_to_repaired = (1 - mask) * image_origin

            # 电成像，要添加左右的 padding
            image_origin = FMI_padding(image_origin, self.padding)
            mask = FMI_padding(mask, self.padding)
            image_masked = FMI_padding(image_masked, self.padding)
            image_to_repaired = FMI_padding(image_to_repaired, self.padding)
        else:
            print('NO SUCH TYPE IMAGE')
            exit(0)

        image_origin = cv2.resize(image_origin, self.pic_shape)
        mask = cv2.resize(mask, self.pic_shape)
        image_masked = cv2.resize(image_masked, self.pic_shape)
        image_to_repaired = cv2.resize(image_to_repaired, self.pic_shape)

        return image_origin/256, 1-mask, image_masked/256, image_to_repaired/256

    def __len__(self):
        return self.length



# # 用来进行图像修复的dataloader，使用模型进行修复时，使用此dataloader
# class repair_dataloader_long_FMI(Dataset):
#     def __init__(self, path=r'D:\Data\pic_repair_paper_effect\width_258_windows_43_belt_6-15pix_empty\LG7-12_1_5190.0_5201.0__pic_masked.png',
#                  len_pic=256, padding=32, len_windows=300, step_windows=100, masked=False):
#         super().__init__()
#         # self.pic_path = path
#         self.pic_all = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         self.len_windows = len_windows
#         self.step_windows = step_windows
#         self.length = (self.pic_all.shape[0]-self.len_windows)//self.step_windows + 2
#
#         # print(self.length, self.ratio, sum(self.ratio))
#         self.pic_shape = (len_pic, len_pic)
#         self.padding = padding
#
#         if masked:
#             ratio_empty = np.random.choice([0.1, 0.15, 0.2, 0.25, 0.3])
#             # 极板个数设置
#             num_belt = np.random.choice([6, 6, 6, 8, 8, 10])
#             mask = get_pic_mask_random(pic_shape=self.pic_all.shape, mask_ratio=ratio_empty, num_belt=num_belt, pic_rorate=False)
#             self.pic_all = mask * self.pic_all
#
#         # print('sub_windows len:{}, step:{}'.format(self.len_windows, self.step_windows))
#
#     def __getitem__(self, index):
#         if index != self.length - 1:
#             image_masked = self.pic_all[index*self.step_windows:index*self.step_windows+self.len_windows, :]
#         else:
#             image_masked = self.pic_all[-self.len_windows:, :]
#
#         image_masked = cv2.resize(image_masked, self.pic_shape)
#         image_masked = FMI_padding(image_masked, self.padding)
#         return image_masked/256
#
#     def __len__(self):
#         return self.length


if __name__ == '__main__':
    a = dataloader_padding_striped(len_pic=128)
    for i in range(40):
        index = np.random.randint(0, a.__len__())
        b, c, d, e = a[index]
        print(b.shape, c.shape, d.shape, e.shape)
        print(np.max(b), np.max(c), np.max(d), np.max(e))
        show_Pic([b, c, d, e], pic_order='22')



