import random

import cv2
import numpy as np
from simulation_mask.cracks_simulation import cracks_simulation
from simulation_mask.holes_simulation import holes_simulation
from src_ele.pic_opeeration import show_Pic


if __name__ == '__main__':
    CS = cracks_simulation()

    # # 背景初始化
    # BASE_BACKGROUND = np.zeros((512, 256), dtype=np.uint8)
    # LIST_IMG_MULTI = []
    # for i in range(9):
    #     IMG_T = CS.generate_random_multi_cracks(config_multi_fractures={'crack_x_shift': 0.1 * i})
    #     LIST_IMG_MULTI.append(IMG_T)
    # show_Pic(LIST_IMG_MULTI, pic_order='33', figure=(9, 16))
    #
    # LIST_IMG_SINGLE = []
    # for i in range(9):
    #     IMG_T = CS.genrate_random_single_crack(config_crack={'crack_x_shift' :0.1 *i})
    #     LIST_IMG_SINGLE.append(IMG_T)
    # show_Pic(LIST_IMG_SINGLE, pic_order='33', figure=(9, 9))


    HS = holes_simulation()
    # IMG_LIST = []
    # for i in range(9):
    #     block_image = HS.genrate_new_blocks(config_blocks={
    #             'width': 256,
    #             'height': 256,
    #             'holes_num': 10,
    #             'ratio_repetition': 0.1,
    #             'vugs_shape_configuration':[[3, 32], [3, 32]]
    #         })
    #     IMG_LIST.append(block_image)
    #
    #     block_image, location_info = HS.add_vugs_random(block_image, vug_num_p=20, ratio_repetition=0.01, vugs_shape_configuration=[[2, 27], [2, 27]])
    #     IMG_LIST.append(block_image)
    # show_Pic(IMG_LIST, pic_order='36', figure=(20, 10))

    for i in range(10):
        IMG_BACKGROUND = np.zeros((10000, 256), dtype=np.uint8)
        end_index = 0
        min_crack_height = 100
        max_crack_height = 400
        crack_x_shift = random.random()
        file_path_save = r'F:\FMI_SIMULATION\simu_cracks'           # 必须全英文

        PIC_LIST = []
        while (end_index < IMG_BACKGROUND.shape[0]-min_crack_height):
            print(end_index)
            mode_random = random.random()
            if mode_random < 0.5:
                # 为地层新增多缝
                crack = CS.generate_random_multi_cracks(config_multi_fractures={'crack_x_shift': crack_x_shift*(0.8+0.4*random.random())})
            else:
                # 为地层新增单缝
                crack = CS.genrate_random_single_crack(config_crack={'crack_x_shift': crack_x_shift*(0.8+0.4*random.random())})
            # 把产生的裂缝信息，随机压缩，并添加到地层背景上
            height_random = random.randint(min_crack_height, min(max_crack_height, IMG_BACKGROUND.shape[0]-end_index))
            crack = cv2.resize(crack, (256, height_random))
            IMG_BACKGROUND[end_index:end_index + height_random, :] = crack
            end_index += height_random
            # end_index += random.randint(0, int(0.1*IMG_BACKGROUND.shape[0]))
            end_index += random.randint(0, 20)
        PIC_LIST.append(IMG_BACKGROUND.copy())
        cv2.imwrite(file_path_save+'\\'+str(i)+'_crack_mask.png', IMG_BACKGROUND)

        # 添加随机的地层孔洞信息
        IMG_BACKGROUND, holes_location = HS.add_vugs_random(IMG_BACKGROUND, vug_num_p=np.random.randint(IMG_BACKGROUND.shape[0]//25, IMG_BACKGROUND.shape[0]//25*2), ratio_repetition=0.05, vugs_shape_configuration=[[2, 40], [2, 40]])
        PIC_LIST.append(IMG_BACKGROUND.copy())
        cv2.imwrite(file_path_save+'\\'+str(i)+'_background_mask.png', IMG_BACKGROUND)

        # show_Pic(PIC_LIST, pic_order='12', figure=(9, 18))