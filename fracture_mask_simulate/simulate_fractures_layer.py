import random
import cv2
import numpy as np
import pandas as pd

from fracture_mask_simulate.cracks_simulation import cracks_simulation
from fracture_mask_simulate.holes_simulation import holes_simulation
from src_ele.pic_opeeration import show_Pic
from src_plot.plot_logging import visualize_well_logs

import pandas as pd
import numpy as np


def df_background_para_adjust_final(df, window_length=5):
    """
    测井曲线窗口处理接口

    参数:
    df: 输入DataFrame，包含以下列:
        ['crack_length', 'crack_width', 'crack_area', 'crack_angle',
         'crack_inclination', 'crack_density', 'hole_area', 'hole_density', 'hole_area_ratio']
    window_length: 窗口长度（默认5）

    返回:
    处理后的DataFrame
    """
    # 1. 输入验证
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是DataFrame")

    required_columns = [
        'depth', 'crack_length', 'crack_width', 'crack_area', 'crack_angle',
        'crack_inclination', 'crack_density', 'hole_area', 'hole_density', 'hole_area_ratio'
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要列: {', '.join(missing_columns)}")
    if window_length < 1:
        raise ValueError("窗口长度必须大于0")

    # 2. 预处理 crack_width 列
    # 先计算 crack_width / crack_density（如果 crack_density 不为零）
    df['crack_width_adjusted'] = df.apply(
        lambda row: row['crack_width'] / row['crack_density'] if row['crack_density'] != 0 else 0,
        axis=1
    )

    # 3. 创建窗口索引
    n = len(df)
    window_indices = np.arange(0, n, 1)

    # 4. 准备结果容器
    result_data = []

    # 5. 处理每个窗口
    for start_idx in window_indices:
        end_idx = min(start_idx + window_length//2, n)
        start_idx_read = max(start_idx-window_length//2, 0)
        # 正常窗口内的孔洞缝参数数据
        window_df = df.iloc[start_idx_read:end_idx]
        # 小窗口内的孔洞缝参数数据，这个主要用来计算 crack_density 以及 crack_width_adjusted、crack_width 参数的
        length_windows = window_df.shape[0]//4
        window_df_small = df.iloc[start_idx_read+length_windows:end_idx-length_windows]

        # 5.1 累加列
        crack_length_sum = window_df['crack_length'].sum() * 0.00515
        crack_area_sum = window_df['crack_area'].sum() * 0.00515 * 0.0025
        hole_area_sum = window_df['hole_area'].sum() * 0.00515 * 0.0025
        hole_density_sum = window_df['hole_density'].sum()
        hole_area_ratio_sum = window_df['hole_area_ratio'].sum()
        crack_density = window_df_small['crack_density'].max()

        # 5.2 计算 crack_width_sum（所有不为零行的均值）
        # 获取 crack_width_adjusted 列中不为零的值
        non_zero_values = window_df_small['crack_width_adjusted'][window_df_small['crack_width_adjusted'] != 0]
        # 计算非零值 的 平均值
        if len(non_zero_values) > 0:
            crack_width_mean = non_zero_values.mean() * 0.0025
        else:
            crack_width_mean = 0

        index = window_df.shape[0]//2
        depth = df['depth'].iloc[start_idx] if len(window_df) > 0 else 0
        crack_angle = window_df['crack_angle'].iloc[index] if len(window_df) > 0 else 0
        crack_inclination = window_df['crack_inclination'].iloc[index] if len(window_df) > 0 else 0
        # crack_density = window_df['crack_density'].iloc[index] if len(window_df) > 0 else 0

        # 5.3 添加到结果
        result_data.append({
            'depth': depth,
            'crack_length': crack_length_sum,
            'crack_width': crack_width_mean,
            'crack_area': crack_area_sum,
            'crack_angle': crack_angle,
            'crack_inclination': crack_inclination,
            'crack_density': crack_density,
            'hole_area': hole_area_sum,
            'hole_density': hole_density_sum,
            'hole_area_ratio': hole_area_ratio_sum
        })

    # 6. 创建结果DataFrame
    result_df = pd.DataFrame(result_data)
    return result_df


def add_depth_column(image_FMI):
    """
    为电成像图像数组添加索引列
    参数:
    stat_image: 原始图像数组 (10000×256)
    返回:
    添加索引列后的新数组 (10000×257)
    """
    # 1. 验证输入
    if not isinstance(image_FMI, np.ndarray):
        raise TypeError("输入必须是numpy数组")

    # 2. 创建深度列 (0-9999的等差数列)
    depth_column = np.arange(0, image_FMI.shape[0], dtype=np.float32).reshape(-1, 1)

    # 3. 将索引列添加到原始数组的第一列
    # 使用np.hstack水平拼接
    # result = np.hstack((depth_column, stat_image))

    # 或者使用np.c_更简洁
    result = np.c_[depth_column, image_FMI.astype(np.float32)]

    return result


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

    for i in range(20):
        IMG_BACKGROUND_CRACKS = np.zeros((5000, 256), dtype=np.uint8)

        # 存放9个孔洞缝参数，分别是 裂缝密度、裂缝张开度、裂缝长度、裂缝有效面积、面孔率、孔洞密度、孔洞面积
        # 'crack_length', 'crack_width', 'crack_area', 'crack_angle', 'crack_inclination', 'crack_density', 'hole_area', 'hole_density', 'hole_area_ratio'
        crack_hole_parameter = np.zeros((IMG_BACKGROUND_CRACKS.shape[0], 10), dtype=np.float64)
        depth_array = np.linspace(start=0, stop=IMG_BACKGROUND_CRACKS.shape[0] - 1, num=IMG_BACKGROUND_CRACKS.shape[0])
        df_background_para = pd.DataFrame(crack_hole_parameter, columns=['depth', 'crack_length', 'crack_width', 'crack_area', 'crack_angle', 'crack_inclination', 'crack_density', 'hole_area', 'hole_density', 'hole_area_ratio'])
        df_background_para['depth'] = depth_array
        df_background_para.astype(np.float64)

        cols_cracks = ['crack_length', 'crack_width', 'crack_area', 'crack_angle', 'crack_inclination', 'crack_density']
        cols_holes = ['hole_area', 'hole_density', 'hole_area_ratio']

        end_index = 0
        min_crack_height = 50
        max_crack_height = 500
        crack_x_shift = random.random()
        # file_path_save = r'F:\DeepLData\FMI_SIMULATION\simu_cracks_2'           # 必须全英文
        file_path_save = r'F:\DeepLData\FMI_SIMULATION\simu_cracks'           # 必须全英文

        while (end_index < IMG_BACKGROUND_CRACKS.shape[0] - min_crack_height):
            mode_random = random.random()
            if mode_random < 0.5:
                # 为地层新增多缝
                crack, config_crack, df_cracks_para = CS.generate_random_multi_cracks(config_multi_fractures={'crack_x_shift': crack_x_shift*(0.5+1.0*random.random())})
            else:
                # 为地层新增单缝
                crack, config_crack, df_cracks_para = CS.genrate_random_single_crack(config_crack={'height_background': np.random.randint(100, 400), 'crack_x_shift': crack_x_shift*(0.5+1.0*random.random())})

            # 把产生的裂缝信息， 随机压缩， 并添加到地层背景上
            height_random = random.randint(min_crack_height, min(max_crack_height, IMG_BACKGROUND_CRACKS.shape[0] - end_index))
            crack = cv2.resize(crack, (256, height_random))
            config_crack_n, df_cracks_para_n = CS.adjust_crack_para_by_target_window_height(config_crack, df_cracks_para, (height_random, 256))

            # 图像合并
            IMG_BACKGROUND_CRACKS[end_index:end_index + height_random, :] = crack
            # 裂缝的参数合并
            df_background_para.loc[end_index:end_index + height_random - 1, cols_cracks] += df_cracks_para_n[cols_cracks].values
            end_index += height_random

            end_index += random.randint(100, 500)

        # 添加随机的地层孔洞信息
        IMG_BACKGROUND_CRACKS_HOLES, holes_location, df_hole_para = HS.add_vugs_random(IMG_BACKGROUND_CRACKS, vug_num_p=np.random.randint(IMG_BACKGROUND_CRACKS.shape[0] // 25, IMG_BACKGROUND_CRACKS.shape[0] // 25 * 2), ratio_repetition=0.05, vugs_shape_configuration=[[2, 40], [2, 40]])

        IMG_BACKGROUND_HOLES = IMG_BACKGROUND_CRACKS_HOLES.copy() - IMG_BACKGROUND_CRACKS.copy()
        ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        IMG_BACKGROUND_HOLES = cv2.morphologyEx(IMG_BACKGROUND_HOLES.copy(), cv2.MORPH_OPEN, ellipse_kernel)
        _, IMG_BACKGROUND_HOLES = cv2.threshold(IMG_BACKGROUND_HOLES, 5, 255, cv2.THRESH_BINARY)  # cv2.THRESH_BINARY：当像素值大于阈值时，赋予最大值；否则为0。
        df_background_para[cols_holes] = df_hole_para[cols_holes]

        index_save = i

        ####### 裂缝mask保存
        # name_cracks = str(index_save)+'_cracks_mask'
        # cv2.imwrite(file_path_save +'\\' + name_cracks +'.png', IMG_BACKGROUND_CRACKS)
        # IMG_BACKGROUND_depthed = add_depth_column(IMG_BACKGROUND_CRACKS)
        # np.savetxt(file_path_save+'\\'+name_cracks+'.txt', IMG_BACKGROUND_depthed, fmt='%.2f', header=f'{name_cracks}\n\n', comments='', delimiter='    ')

        ####### 背景mask保存
        # name_background = str(index_save)+'_background_mask'
        # cv2.imwrite(file_path_save+'\\'+name_background+'.png', IMG_BACKGROUND_CRACKS_HOLES)
        # IMG_BACKGROUND_CRACKS_HOLES_depthed = add_depth_column(IMG_BACKGROUND_CRACKS_HOLES)
        # np.savetxt(file_path_save+'\\'+name_background+'.txt', IMG_BACKGROUND_CRACKS_HOLES_depthed, fmt='%.2f', header=f'{name_background}\n\n', comments='', delimiter='    ')

        ####### 孔洞mask保存
        # name_holes = str(index_save)+'_holes_mask'
        # cv2.imwrite(file_path_save+'\\'+name_holes+'.png', IMG_BACKGROUND_HOLES)
        # IMG_BACKGROUND_HOLES_depthed = add_depth_column(IMG_BACKGROUND_HOLES)
        # np.savetxt(file_path_save+'\\'+name_holes+'.txt', IMG_BACKGROUND_HOLES_depthed, fmt='%.2f', header=f'{name_holes}\n\n', comments='', delimiter='    ')

        df_background_para_final = df_background_para_adjust_final(df_background_para, window_length=200)
        # df_background_para.to_csv(file_path_save + '\\' + str(index_save) + '_background_para_origin.csv', index=False)
        # df_background_para_final.to_csv(file_path_save+'\\'+str(index_save)+'_background_para_processed.csv', index=False)

        visualize_well_logs(
            data=df_background_para,
            depth_col='depth',
            curve_cols=cols_cracks+cols_holes,
        )
        visualize_well_logs(
            data=df_background_para_final,
            depth_col='depth',
            curve_cols=cols_cracks+cols_holes,
        )
        show_Pic([IMG_BACKGROUND_CRACKS, IMG_BACKGROUND_CRACKS_HOLES, IMG_BACKGROUND_HOLES], pic_order='13', figure=(9, 18))