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

    # print('all mask pixel:{}, num belt:{}, pixel per skip:{}, mask width per belt:{}'.format(num_mask, num_belt, pix_skip, mask_belt_width))

    mask = np.ones(pic_shape, dtype='float32')
    for i in range(len(num_belt_para)):
        mask[:, num_belt_para[i][0]: num_belt_para[i][1]] = 0

    return mask

# 电成像FMI图像，左右两边指定像素的Padding
def FMI_padding(image, padding=16):
    image_n = np.zeros((image.shape[0], image.shape[1]+2*padding), dtype=np.uint8)
    image_n[:, padding: -padding] = image
    image_n[:, :padding] = image[:, -padding:]
    image_n[:, -padding:] = image[:, :padding]
    return image_n

# 随机的图像裁剪，随机裁剪图像的一部分作为数据输入
def pic_crop_random(pic, pic_shape_ratio=None):
    # 设置随机裁剪的图像大小
    if pic_shape_ratio is None:
        pic_shape_ratio = 0.6 + 0.38 * np.random.random()

    pic_h = int(pic.shape[0] * pic_shape_ratio)
    pic_w = int(pic.shape[1] * pic_shape_ratio)
    start_h = np.random.randint(0, max(pic.shape[0]-pic_h-1, 10))
    start_w = np.random.randint(0, max(pic.shape[1]-pic_w-1, 10))

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
        if not (path_t.lower().endswith(('.jpg', '.png', '.jpeg'))):
            print(f"跳过非图片文件: {path_t}")
            return self._get_dummy_item()

        # 常规图片的 空白条带图像生成
        if path_t.__contains__('REAL_WORLD_PIC'):
            image_origin = cv2.imread(self.target_list_file_path[index], cv2.IMREAD_COLOR_RGB)

            # 添加检查
            if image_origin is None:
                print(f"无法读取图像: {self.target_list_file_path[index]}")
                # 返回一个替代图像或跳过
                return self._get_dummy_item()

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
            print('NO SUCH TYPE IMAGE:{}'.format(path_t))
            exit(0)

        image_origin = cv2.resize(image_origin, self.pic_shape)
        mask = cv2.resize(mask, self.pic_shape)
        image_masked = cv2.resize(image_masked, self.pic_shape)
        image_to_repaired = cv2.resize(image_to_repaired, self.pic_shape)

        # 转换为 float32 并归一化
        image_origin = image_origin.astype(np.float32)/255.0
        mask = mask.astype(np.float32)/1.0
        image_masked = image_masked.astype(np.float32)/255.0
        image_to_repaired = image_to_repaired.astype(np.float32)/255.0

        # 确保所有图像都有通道维度
        if image_origin.ndim == 2:  # 如果是二维灰度图
            image_origin = np.expand_dims(image_origin, axis=0)  # 添加通道维度
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)
        if image_masked.ndim == 2:
            image_masked = np.expand_dims(image_masked, axis=0)
        if image_to_repaired.ndim == 2:
            image_to_repaired = np.expand_dims(image_to_repaired, axis=0)

        return {"real_all":image_origin, "mask":1-mask, "real_input":image_masked, "real_deduction":image_to_repaired}

    def __len__(self):
        return self.length

    def _get_dummy_item(self):
        """创建一个替代图像项"""
        dummy_img = np.zeros(self.pic_shape, dtype=np.uint8)
        dummy_mask = np.zeros(self.pic_shape, dtype=np.uint8)

        # 添加通道维度
        dummy_img = np.expand_dims(dummy_img, axis=0)
        dummy_mask = np.expand_dims(dummy_mask, axis=0)

        return {
            "real_all": dummy_img,
            "mask": dummy_mask,
            "real_input": dummy_img,
            "real_deduction": dummy_img
        }

# 用来进行图像修复的dataloader，使用模型进行修复时，使用此dataloader
class dataloader_FMI_logging(Dataset):
    def __init__(self, path=r'F:\DeepLData\FMI_SIMULATION\simu_FMI\0_fmi_dyna.png', padding=16, len_windows=256, step_windows=10, pic_length_target=128, mask_config={'ratio_empty':0.2, 'num_belt':6, 'rotate_rb':True}):
        super().__init__()
        """
        path:输入的文件路径
        padding:左右两边padding长度大小
        len_windows:图像分割的窗长大小
        step_windows:步长，窗口遍历的步长
        pic_length_target:图像输出的窗长大小
        mask_config:掩码配置
        """
        self.pic_origin = cv2.imread(path, cv2.IMREAD_GRAYSCALE)            # 图像读取
        self.pic_origin_padding = FMI_padding(self.pic_origin, padding)     # 图像加padding，用来进行辅助修复
        self.len_windows = len_windows                                      # 图像遍历窗口长度设置
        self.length_target = pic_length_target                              # 输出图像边长，方形的
        self.pic_shape_target = (pic_length_target, pic_length_target)      # 输出图像形状
        self.step_windows = step_windows                                    # 遍历的窗口步长
        self.length = (self.pic_origin.shape[0]-len_windows)//self.step_windows+2           # 数据集的长度
        self.pic_shape_windows_org = (self.pic_origin_padding.shape[1], len_windows)        # 原始的图像形状
        self.padding = padding                                              # 左右padding length 设置

        if mask_config:
            mask_config_default = {'ratio_empty':np.random.choice([0.1, 0.15, 0.2, 0.25, 0.3]), 'num_belt':np.random.choice([6, 6, 6, 8, 8, 10]), 'rotate_rb':np.random.choice([True, False])}
            # 合并配置参数
            mask_config = {**mask_config_default, **mask_config}
            # 极板个数初始化
            num_belt = mask_config['num_belt']
            # 空白率初始化
            ratio_empty = mask_config['ratio_empty']
            # 是否进行绕井壁旋转
            rotate_rb = mask_config['rotate_rb']
            # 获得掩码图像
            mask = get_pic_mask_random(pic_shape=self.pic_origin.shape, mask_ratio=ratio_empty, num_belt=num_belt)
            self.mask_padding = FMI_padding(mask, self.padding)
            if rotate_rb:
                self.mask_padding, rb_random = pic_rotate_random(self.mask_padding)
            self.pic_masked_padding = self.pic_origin_padding * self.mask_padding
        else:
            self.mask_padding = cv2.threshold(self.pic_origin_padding, 10, 255, cv2.THRESH_BINARY)
            self.pic_masked_padding = self.pic_origin_padding * self.mask_padding


    def __getitem__(self, index):
        if index != self.length-1:
            image_masked = self.pic_masked_padding[index*self.step_windows:index*self.step_windows+self.len_windows, :]
            image_origin = self.pic_origin[index*self.step_windows:index*self.step_windows+self.len_windows, :]
            image_mask = self.mask_padding[index*self.step_windows:index*self.step_windows+self.len_windows, :]
        else:
            image_masked = self.pic_masked_padding[-self.len_windows:, :]
            image_origin = self.pic_origin[-self.len_windows:, :]
            image_mask = self.mask_padding[-self.len_windows:, :]

        # 形状变换
        image_masked = cv2.resize(image_masked, self.pic_shape_target)
        image_origin = cv2.resize(image_origin, self.pic_shape_target)
        image_mask = cv2.resize(image_mask, self.pic_shape_target)

        # 转换为 float32 并归一化
        image_masked = image_masked.astype(np.float32)/255.0
        image_origin = image_origin.astype(np.float32)/255.0
        image_mask = image_mask.astype(np.float32)

        # 确保所有图像都有通道维度
        if image_masked.ndim == 2:  # 如果是二维灰度图
            image_masked = np.expand_dims(image_masked, axis=0)  # 添加通道维度
        if image_origin.ndim == 2:
            image_origin = np.expand_dims(image_origin, axis=0)
        if image_mask.ndim == 2:
            image_mask = np.expand_dims(image_mask, axis=0)
        return {'img_masked':image_masked, 'img_origin':image_origin, 'img_mask':1-image_mask}

    def __len__(self):
        return self.length

    # 把修复后的图像进行合并成与输入图像一样的图像格式、形状
    def combine_pic_list(self, pic_array_list, path_target_folder='', str_charter='SIMU_1'):
        """
        pic_array_list: array1.shape=[357,1,128,128] ----> 1000*256
        """
        pic_result = np.zeros_like(self.pic_origin_padding, dtype=np.float32)   # 结果保存数据
        pic_weight = np.zeros_like(pic_result, dtype=np.float32)                # 预测结果的权重数据

        # 图像加权拼接
        for i in range(pic_array_list.shape[0]):
            image_t = pic_array_list[i, 0, :, :]
            image_resize = cv2.resize(image_t, self.pic_shape_windows_org)
            if i == 0:
                pic_result[i*self.step_windows:i*self.step_windows+self.len_windows, :] += image_resize
                pic_weight[i*self.step_windows:i*self.step_windows+self.len_windows, :] += 1
            elif i != self.length-1:
                pic_result[i*self.step_windows+self.step_windows:i*self.step_windows+self.len_windows-self.step_windows, :] += image_resize[self.step_windows:-self.step_windows, :]
                pic_weight[i*self.step_windows+self.step_windows:i*self.step_windows+self.len_windows-self.step_windows, :] += 1
            else:
                pic_result[-self.len_windows:, :] += image_resize
                pic_weight[-self.len_windows:, :] += 1
            # show_Pic([image_resize, image_t])

        # 图像的权重剔除，并进行图像阈值恢复，将图像缩放到0-255范围内
        pic_result = (pic_result/pic_weight)*255

        # 只保留图像修复后的部分 加上 图像原始的已存在的部分，这个是剔除了模型输出的图像不需要修复的部分
        pic_result_target = self.pic_origin_padding*self.mask_padding + pic_result*(1-self.mask_padding)

        # 图像裁剪，才调用来进行辅助恢复的左右两边的padding范围，并进行数值类型改变
        pic_result = pic_result[:, self.padding:-self.padding].astype(np.uint8)
        pic_result_target = pic_result_target[:, self.padding:-self.padding].astype(np.uint8)
        pic_mask = self.mask_padding[:, self.padding:-self.padding].astype(np.uint8) * 255

        # 图像保存
        if path_target_folder != '':
            print('path_save:{}'.format(path_target_folder))
            cv2.imwrite(path_target_folder+'\\{}_target_result.png'.format(str_charter), pic_result_target)
            cv2.imwrite(path_target_folder+'\\{}_model_result.png'.format(str_charter), pic_result)
            cv2.imwrite(path_target_folder+'\\{}_mask.png'.format(str_charter), pic_mask)
            cv2.imwrite(path_target_folder+'\\{}_org.png'.format(str_charter), self.pic_origin)

        return pic_result, pic_result_target


if __name__ == '__main__':
    # a = dataloader_padding_striped(len_pic=128)
    # for i in range(40):
    #     index = np.random.randint(0, a.__len__())
    #     img_list = a[index]
    #     b, c, d, e = img_list['mask'], img_list['real_all'], img_list['real_input'], img_list['real_deduction']
    #     print(b.shape, c.shape, d.shape, e.shape)
    #     print(np.max(b), np.max(c), np.max(d), np.max(e))
    #     show_Pic([b, c, d, e], pic_order='22')

    dFl = dataloader_FMI_logging()
    for i in range(10):
        img_list = dFl[i]
        b, c, d = img_list['img_masked'], img_list['img_origin'], img_list['img_mask']
        print(b.shape, c.shape, d.shape)
        show_Pic([b[0, :, :], c[0, :, :], d[0, :, :]])
