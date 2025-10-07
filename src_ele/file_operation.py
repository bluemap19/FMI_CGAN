import os
import numpy as np
import cv2


def get_ele_data_from_path(strname = r'D:\Data\target\107S\YS107_FMI_BorEID_FA.txt', depth = [-1.0, -1.0]):
    img_data = np.array([])
    depth_data = np.array([])
    if (strname.endswith('.png') | strname.endswith('.jpg')) :
        img_data = np.array(cv2.imread(strname, cv2.IMREAD_GRAYSCALE))
        if img_data.shape[0] == 0:
            print('image in data is empty:{}'.format(strname))
            exit(0)
        depth_data = np.zeros((img_data.shape[0], 1))
        if len(strname.split('_')) < 4:
            print(f'file {strname} dont contain depth information')
            return img_data, depth_data
        startdep = float(strname.split('\\')[-1].split('/')[-1].split('_')[2])
        enddep = float(strname.split('\\')[-1].split('/')[-1].split('_')[3])
        Step = (enddep-startdep)/img_data.shape[0]
        for i in range(img_data.shape[0]):
            depth_data[i] = i*Step + startdep
    elif (strname.endswith('.txt')):
        AllData = np.loadtxt(strname, delimiter='\t', skiprows=8, encoding='GBK')
        if AllData.shape[0] ==0:
            print('text in data is empty:{}'.format(strname))
            exit(0)
        img_data = AllData[:, 1:]
        depth_data = AllData[:, 0].reshape((AllData.shape[0], 1))
    elif (strname.endswith('.csv')):
        print('CsvData')

    # 从深度方向截取数据
    if (depth[0] < 0) & (depth[1] < 0):
        pass
    else:
        Step = (depth_data[-1] - depth_data[0])/depth_data.shape[0]
        start_index = 0
        end_index = 0
        if depth[0] <= 0:
            start_index = 0
        if depth[1] <= 0:
            end_index = img_data.shape[0]-1
        for i in range(depth_data.shape[0]):
            if abs((depth_data[i]-depth[0])) <= Step/2:
                start_index = i
            if abs((depth_data[i]-depth[1])) <= Step/2:
                end_index = i

        return img_data[start_index:end_index, :], depth_data[start_index:end_index, :]

    return img_data, depth_data



def get_test_ele_data():
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建数据目录路径
    data_dir = os.path.join(script_dir, "get_random_data")

    # 定义文件名
    # dyna_file = "lg7-4_163_5183.3027_5183.9277_dyna.png"
    # stat_file = "lg7-4_163_5183.3027_5183.9277_stat.png"
    dyna_file = r'D:\DeepLData\target_stage1_small_big_mix\lg7-4_165_5184.3027_5184.9277_dyna.png'
    stat_file = r'D:\DeepLData\target_stage1_small_big_mix\lg7-4_165_5184.3027_5184.9277_stat.png'

    # 构建完整路径
    path_test1 = os.path.join(data_dir, dyna_file)
    path_test2 = os.path.join(data_dir, stat_file)

    # 检查文件是否存在
    if not os.path.exists(path_test1):
        raise FileNotFoundError(f"文件不存在: {path_test1}")
    if not os.path.exists(path_test2):
        raise FileNotFoundError(f"文件不存在: {path_test2}")

    # 获取数据
    data_img_dyna, data_depth = get_ele_data_from_path(path_test1)
    data_img_stat, data_depth = get_ele_data_from_path(path_test2)

    return data_img_dyna, data_img_stat, data_depth

if __name__ == '__main__':
    img_data, depth_data = get_ele_data_from_path(r'D:\DeepLData\target_stage1_small_big_mix\lg7-4_166_5184.8027_5186.0527_dyna.png')
    print(img_data.shape, depth_data.shape)