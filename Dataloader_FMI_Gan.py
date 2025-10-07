import numpy as np
from torch.utils.data import Dataset
from pic_preprocess import pic_binary_random
from src_ele.dir_operation import get_all_file_paths
from src_ele.file_operation import get_ele_data_from_path
import cv2
from src_ele.pic_opeeration import show_Pic

# 返回的是，文件夹下的，随机一张图片的，经过处理后的两张图像原始照片，以及其相对应的mask图像
class dataloader_fmi_mask_create(Dataset):
    def __init__(self, path=r'D:\Data\target_uns_class', x_l=256, y_l=256, pic_rorate=True):
        super().__init__()
        self.list_all_file = get_all_file_paths(path)
        self.length = len(self.list_all_file)//2
        self.x_l = x_l
        self.y_l = y_l
        self.pic_rorate = pic_rorate

    def __getitem__(self, index):
        path_temp = self.list_all_file[index*2]
        path_temp_stat = ''
        # print(path_temp)

        if path_temp.__contains__('dyna'):
            path_temp_stat = path_temp.replace('dyna', 'stat')
        elif path_temp.__contains__('stat'):
            path_temp_stat = path_temp
            path_temp = path_temp_stat.replace('stat', 'dyna')

        pic_dyna, depth = get_ele_data_from_path(path_temp)
        pic_stat, depth = get_ele_data_from_path(path_temp_stat)
        pic_dyna = cv2.resize(pic_dyna, (self.x_l, self.y_l))
        pic_stat = cv2.resize(pic_stat, (self.x_l, self.y_l))

        # show_Pic([pic_dyna, pic_stat], pic_order='12', pic_str=['', ''], save_pic=False, path_save='')

        dyna_kThreshold_shift = 1.3
        pic_dyna_mask, pic_dyna = pic_binary_random(pic_dyna, kThreshold_shift=dyna_kThreshold_shift)
        # stat_kThreshold_shift = np.random.randint(22, 28)
        stat_kThreshold_shift = 1.2
        pic_stat_mask, pic_stat = pic_binary_random(pic_stat, kThreshold_shift=stat_kThreshold_shift)
        # print('kThreshold_shift dyna and stat is:{}, {}'.format(dyna_kThreshold_shift, stat_kThreshold_shift))

        pic_dyna = pic_dyna.reshape((1, self.x_l, self.y_l))/256
        pic_stat = pic_stat.reshape((1, self.x_l, self.y_l))/256
        pic_all_org = np.append(pic_dyna, pic_stat, axis=0)

        pic_dyna_mask = pic_dyna_mask.reshape((1, self.x_l, self.y_l))/256
        pic_stat_mask = pic_stat_mask.reshape((1, self.x_l, self.y_l))/256
        pic_all_mask = np.append(pic_dyna_mask, pic_stat_mask, axis=0)
        return pic_all_org, pic_all_mask


    def __len__(self):
        return self.length

        # 或者return len(self.trg), src和trg长度一样


if __name__ == '__main__':
    a = dataloader_fmi_mask_create(path=r'D:\DeepLData\target_stage1_small_big_mix')
    index_random = np.random.randint(0, 200)
    print(index_random)
    pic_all_org, pic_all_mask = a[index_random]
    print(pic_all_mask.shape, pic_all_org.shape)
    show_Pic([1-pic_all_org[0], 1-pic_all_org[1], 1-pic_all_mask[0], 1-pic_all_mask[1]], pic_order='22')


    for i in range(a.length):
        temp = a[i]
        show_Pic([temp[0][0]*256, temp[0][1]*256, temp[1][0]*256, temp[1][1]*256], pic_order='22', pic_str=[], save_pic=False, path_save='')
    # print(a.length)
    # print(a[0][0].shape)

    (a.length)
    print(a[0][0].shape)

