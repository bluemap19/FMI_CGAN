import pandas as pd
from fracture_mask_simulate.simulate_fractures_layer import df_background_para_adjust_final
from src_ele.dir_operation import search_files_by_criteria
pd.set_option("display.max_rows", 1000)#可显示1000行
pd.set_option("display.max_columns", 1000)#可显示1000列


if __name__ == '__main__':

    path_folder = r'F:\DeepLData\FMI_SIMULATION\simu_FMI'
    target_path_list = search_files_by_criteria(search_root=path_folder, name_keywords=['_background_para_origin'], file_extensions=['.csv'])
    print(target_path_list)

    cols_target = ['crack_length', 'crack_width', 'crack_area', 'crack_density', 'hole_area', 'hole_density', 'hole_area_ratio']
    for path in target_path_list:
        print(path)
        df_paras_origin = pd.read_csv(path)
        print(df_paras_origin.columns, df_paras_origin[cols_target].describe())

        df_processed = df_background_para_adjust_final(df_paras_origin, window_length=200)

        print(df_processed[cols_target].describe())
        df_processed.to_csv(path.replace('_background_para_origin.csv', '_background_para_processed.csv'), index=False)