import random

import cv2
from src_ele.dir_operation import search_files_by_criteria

if __name__ == '__main__':
    path_input_folder = r'F:\DeepLData\FMI_SIMULATION'
    target_folder = r'F:\DeepLData\simulate_fmi_small_big_mix'
    WINDOWS_LENGTH = [300, 500]
    WINDOWS_STEP = 150

    for i in range(20):
        index = str(i)
        dyna_path = search_files_by_criteria(path_input_folder, name_keywords=[index, 'dyna'], file_extensions=['png'])
        stat_path = search_files_by_criteria(path_input_folder, name_keywords=[index, 'stat'], file_extensions=['png'])
        mask_all_path = search_files_by_criteria(path_input_folder, name_keywords=[index, 'background_mask'])
        mask_crack_path = search_files_by_criteria(path_input_folder, name_keywords=[index, 'crack_mask'])

        data_dyna = cv2.imread(dyna_path[-1], cv2.IMREAD_GRAYSCALE)
        data_stat = cv2.imread(stat_path[-1], cv2.IMREAD_GRAYSCALE)
        data_mask_all = cv2.imread(mask_all_path[-1], cv2.IMREAD_GRAYSCALE)
        data_mask_crack = cv2.imread(mask_crack_path[-1], cv2.IMREAD_GRAYSCALE)
        data_mask_hole = data_mask_all.copy() - data_mask_crack.copy()
        ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        data_mask_hole = cv2.morphologyEx(data_mask_hole.copy(), cv2.MORPH_OPEN, ellipse_kernel)

        NUM_SPLIT = (data_dyna.shape[0]-WINDOWS_LENGTH[0])//WINDOWS_STEP
        print(dyna_path[-1], stat_path[-1], mask_all_path[-1], mask_crack_path[-1])
        print(data_dyna.shape, data_stat.shape, data_mask_all.shape, data_mask_crack.shape, WINDOWS_LENGTH, WINDOWS_STEP, NUM_SPLIT)
        for j in range(NUM_SPLIT):
            START_INDEX = j * WINDOWS_STEP
            END_INDEX = j * WINDOWS_STEP + random.randint(WINDOWS_LENGTH[0], WINDOWS_LENGTH[1])

            window_dyna = data_dyna[START_INDEX:END_INDEX, :]
            window_stat = data_stat[START_INDEX:END_INDEX, :]
            window_mask_all = data_mask_all[START_INDEX:END_INDEX, :]
            window_mask_crack = data_mask_crack[START_INDEX:END_INDEX, :]
            window_mask_hole = data_mask_hole[START_INDEX:END_INDEX, :]

            cv2.imwrite(target_folder+'\\dyna_stat_fmi\\{}_{}_{}_{}_{}.png'.format(i, j, 'dyna', 0, 1), window_dyna)
            cv2.imwrite(target_folder+'\\dyna_stat_fmi\\{}_{}_{}_{}_{}.png'.format(i, j, 'stat', 0, 1), window_stat)
            cv2.imwrite(target_folder+'\\total_mask\\{}_{}_{}_{}_{}.png'.format(i, j, 'mask-all', 0, 1), window_mask_all)
            cv2.imwrite(target_folder+'\\crack_mask\\{}_{}_{}_{}_{}.png'.format(i, j, 'mask-crack', 0, 1), window_mask_crack)
            cv2.imwrite(target_folder+'\\hole_mask\\{}_{}_{}_{}_{}.png'.format(i, j, 'mask-hole', 0, 1), window_mask_hole)



