import glob
import random
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pic_preprocess import pic_binary_random, pic_rorate_random, pic_tailor
from src_ele.dir_operation import get_all_file_paths
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic


class ImageDataset_FMI(Dataset):
    def __init__(self, path=r'D:\Data\target_uns_class', x_l=256, y_l=256, random_crop=True, erode=True):
        super().__init__()
        self.list_all_file = get_all_file_paths(path)
        self.length = len(self.list_all_file)//2
        self.x_l = x_l
        self.y_l = y_l
        self.random_crop = random_crop
        self.erode = erode

    def __getitem__(self, index):
        path_temp_dyna = self.list_all_file[index*2]
        path_temp_stat = ''

        if path_temp_dyna.__contains__('dyna'):
            path_temp_stat = path_temp_dyna.replace('dyna', 'stat')
        elif path_temp_dyna.__contains__('stat'):
            path_temp_stat = path_temp_dyna
            path_temp_dyna = path_temp_stat.replace('stat', 'dyna')

        pic_dyna, depth = get_ele_data_from_path(path_temp_dyna)
        pic_stat, depth = get_ele_data_from_path(path_temp_stat)

        pic_all_org = np.array([pic_dyna, pic_stat])
        pic_style = pic_tailor(pic_all_org, pic_shape_ex=[40, 40])
        pic_dyna_style = pic_style[0]
        pic_stat_style = pic_style[1]

        if self.random_crop:
            crop_ratio = random.uniform(0.3, 1)
            pic_all_org = pic_tailor(pic_all_org, pic_shape_ex=[crop_ratio, crop_ratio])
            pic_dyna = pic_all_org[0]
            pic_stat = pic_all_org[1]

        pic_dyna_style = cv2.resize(pic_dyna_style, (self.x_l, self.y_l))
        pic_stat_style = cv2.resize(pic_stat_style, (self.x_l, self.y_l))
        pic_dyna = cv2.resize(pic_dyna, (self.x_l, self.y_l))
        pic_stat = cv2.resize(pic_stat, (self.x_l, self.y_l))
        pic_all_org = np.array([pic_dyna, pic_stat])

        dyna_kThreshold_shift = 1.0
        pic_dyna_mask, pic_dyna = pic_binary_random(pic_dyna, kThreshold_shift=dyna_kThreshold_shift, erode=self.erode)

        stat_kThreshold_shift = 1.0
        pic_stat_mask, pic_stat = pic_binary_random(pic_stat, kThreshold_shift=stat_kThreshold_shift, erode=self.erode)

        pic_dyna_16 = cv2.resize(cv2.resize(pic_dyna, (16, 16)), (self.x_l, self.y_l))
        pic_stat_16 = cv2.resize(cv2.resize(pic_stat, (16, 16)), (self.x_l, self.y_l))
        pic_dyna_8 = cv2.resize(cv2.resize(pic_dyna, (8, 8)), (self.x_l, self.y_l))
        pic_stat_8 = cv2.resize(cv2.resize(pic_stat, (8, 8)), (self.x_l, self.y_l))
        pic_dyna_4 = cv2.resize(cv2.resize(pic_dyna, (4, 4)), (self.x_l, self.y_l))
        pic_stat_4 = cv2.resize(cv2.resize(pic_stat, (4, 4)), (self.x_l, self.y_l))
        pic_all_mask = np.array([pic_dyna_mask, pic_stat_mask, pic_dyna_8, pic_stat_8, pic_dyna_4, pic_stat_4, pic_dyna_style, pic_stat_style])

        # print(np.max(pic_all_mask), np.min(pic_all_mask), np.mean(pic_all_mask), np.max(pic_all_org), np.min(pic_all_org), np.mean(pic_all_org))

        return {"A": pic_all_org/256, "B": pic_all_mask/256}

    def __len__(self):
        return self.length


if __name__ == '__main__':
    a = ImageDataset_FMI(r'D:\DeepLData\target_stage1_small_big_mix')

    for i in range(20):
        b = a[np.random.randint(0, a.length-1)]
        print(b['A'].shape, b['B'].shape)
        show_Pic([1-b['A'][0,:,:], 1-b['A'][1,:,:],
                  1-b['B'][0,:,:], 1-b['B'][1,:,:],
                  1-b['B'][2,:,:], 1-b['B'][3,:,:],
                  1-b['B'][4,:,:], 1-b['B'][5,:,:],
                  1-b['B'][6,:,:], 1-b['B'][7,:,:]],
                 pic_order='25', figuresize=(16, 8))
