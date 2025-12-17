import os
import numpy as np
import cv2
import pandas as pd


def get_ele_data_from_path(strname=r'D:\Data\target\107S\YS107_FMI_BorEID_FA.txt', depth=[-1.0, -1.0]):
    """
    电成像测井资料通用读取接口

    功能: 从文件路径读取电成像数据，支持多种格式（PNG/JPG图像、TXT文本、CSV表格）
          自动提取深度信息，支持按深度范围截取数据

    参数:
    - strname: 文件路径，支持 .png/.jpg/.txt/.csv 格式
    - depth: 深度范围 [起始深度, 结束深度]，负值表示不进行深度截取

    返回:
    - img_data: 电成像数据矩阵 (深度点数 × 电极数)
    - depth_data: 对应的深度数据 (深度点数 × 1)

    原理说明:
    1. 根据文件扩展名判断数据类型
    2. 图像文件: 从文件名解析深度信息，生成等间隔深度序列
    3. 文本文件: 读取格式化数据，第一列为深度，其余为电成像值
    4. 支持按指定深度范围截取数据子集
    """

    # 初始化返回变量
    img_data = np.array([])  # 电成像数据矩阵
    depth_data = np.array([])  # 深度数据向量

    # ============================================================================
    # 1. 处理图像文件 (.png, .jpg)
    # ============================================================================
    if strname.endswith('.png') or strname.endswith('.jpg'):
        # 读取灰度图像
        img_data = np.array(cv2.imread(strname, cv2.IMREAD_GRAYSCALE))

        # 数据检查: 验证图像是否成功加载
        if img_data.shape[0] == 0:
            print(f'图像数据为空: {strname}')
            exit(0)  # 严重错误，直接退出程序

        # 创建深度数据占位符 (与图像行数相同)
        depth_data = np.arange(0, img_data.shape[0], dtype=np.float32).reshape(-1, 1)

        # 尝试从文件名解析深度信息
        # 文件名格式示例: wellname_startDepth_endDepth_suffix.png
        if len(strname.split('_')) < 4:
            print(f'文件 {strname} 不包含深度信息，使用默认深度序列')
            return img_data, depth_data

        try:
            # 提取文件名（去除路径）
            filename = strname.split('\\')[-1].split('/')[-1]
            # 解析深度信息: 假设格式为 ..._startDepth_endDepth_...
            parts = filename.split('_')
            startdep = float(parts[2])  # 起始深度
            enddep = float(parts[3])  # 结束深度

            # 计算深度步长并生成深度序列
            Step = (enddep - startdep) / img_data.shape[0]
            for i in range(img_data.shape[0]):
                depth_data[i] = i * Step + startdep

        except Exception as e:
            # 深度解析失败，使用默认深度序列 (0, 1, 2, ...)
            print(f'文件 {strname} 深度信息解析失败: {e}，使用默认深度序列')
            depth_data = np.arange(0, img_data.shape[0]).reshape(-1, 1)

    # ============================================================================
    # 2. 处理文本文件 (.txt) - 电成像标准数据格式
    # ============================================================================
    elif strname.endswith('.txt'):
        # 读取文本数据，跳过前8行头文件，使用制表符分隔
        AllData = np.loadtxt(strname, delimiter='\t', skiprows=8, encoding='GBK')

        # 数据检查: 验证数据是否成功加载
        if AllData.shape[0] == 0:
            print(f'文本数据为空: {strname}')
            exit(0)  # 严重错误，直接退出程序

        # 文本文件格式: 第一列为深度，其余列为电成像测量值
        img_data = AllData[:, 1:]  # 电成像数据 (排除深度列)
        depth_data = AllData[:, 0].reshape((AllData.shape[0], 1))  # 深度数据 (第一列)

    # ============================================================================
    # 3. 处理CSV文件 (.csv) - 待实现功能
    # ============================================================================
    elif strname.endswith('.csv'):
        print('CSV数据格式待实现')
        try:
            # 读取CSV文件，使用pandas处理
            df = pd.read_csv(strname)

            # 数据检查: 验证CSV文件是否成功加载
            if df.empty:
                print(f'CSV数据为空: {strname}')
                exit(0)
            # 验证列数足够（至少2列：深度列+至少1个数据列）
            if df.shape[1] < 2:
                print(f'CSV文件列数不足: {strname}，需要至少2列（深度列+数据列）')
                exit(0)

            # 提取第一列作为深度数据
            depth_data = df.iloc[:, 0].values.reshape(-1, 1)
            # 提取其余列作为电成像数据
            img_data = df.iloc[:, 1:].values
            # 数据验证: 检查深度数据是否单调递增
            depth_diff = np.diff(depth_data.flatten())

            if np.any(depth_diff <= 0):
                print(f'警告: CSV文件深度数据非单调递增，可能存在异常: {strname}')
            # 数据验证: 检查电成像数据范围
            if np.any(np.isnan(img_data)):
                print(f'警告: CSV文件包含NaN值: {strname}')
            if np.any(np.isinf(img_data)):
                print(f'警告: CSV文件包含无穷大值: {strname}')
            print(f'成功加载CSV文件: {strname}，数据形状: {img_data.shape}')
        except Exception as e:
            print(f'读取CSV文件失败: {strname}，错误: {e}')
            exit(0)

    # ============================================================================
    # 4. 深度范围截取功能
    # ============================================================================
    # 检查是否需要按深度范围截取数据
    if depth[0] < 0 and depth[1] < 0:
        # 深度参数为负值，不进行截取，返回全部数据
        pass
    else:
        # 计算深度步长 (用于深度匹配)
        Step = (depth_data[-1, 0] - depth_data[0, 0]) / depth_data.shape[0]

        # 初始化截取索引
        start_index = 0
        end_index = 0

        # 处理起始深度参数
        if depth[0] <= 0:
            start_index = 0  # 起始深度参数无效，从0开始
        # 处理结束深度参数
        if depth[1] <= 0:
            end_index = img_data.shape[0] - 1  # 结束深度参数无效，取到最后

        # 查找最接近指定深度的数据点索引
        for i in range(depth_data.shape[0]):
            # 使用半步长容差匹配深度值
            if abs(depth_data[i] - depth[0]) <= Step / 2 + 0.0001:
                start_index = i
            if abs(depth_data[i] - depth[1]) <= Step / 2 + 0.0001:
                end_index = i

        # 返回截取后的数据子集
        return img_data[start_index:end_index, :], depth_data[start_index:end_index, :]

    # 返回完整数据 (未截取)
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


def get_random_fmi():
    """
        随机获取FMI电成像数据

        功能: 从两个预定义的FMI数据路径中随机选择一个，加载动态和静态电成像数据

        返回:
        - dyna_data: 动态电成像数据
        - stat_data: 静态电成像数据
        - depth: 深度数据

        数据检查点:
        - 检查文件路径是否存在
        - 验证加载的数据形状一致性
        - 确认深度数据与图像数据行数匹配
    """
    path_dyna_1 = r'F:\DeepLData\target_stage1_small_big_mix\FMI_IMAGE\ZG_FMI_SPLIT\guan17-11_333_3746.5025_3747.1275_dyna.png'
    path_stat_1 = path_dyna_1.replace('dyna', 'stat')

    path_dyna_2 = r'F:\DeepLData\target_stage1_small_big_mix\FMI_IMAGE\ZG_FMI_SPLIT\lg7-4_120_5161.8027_5163.0527_dyna.png'
    path_stat_2 = path_dyna_2.replace('dyna', 'stat')

    # 数据检查1: 验证文件是否存在
    for path in [path_dyna_1, path_stat_1, path_dyna_2, path_stat_2]:
        try:
            with open(path, 'r'):
                pass
        except FileNotFoundError:
            print(f"警告: 文件不存在 - {path}")
            # 这里可以添加默认图像生成逻辑或抛出异常

    # 加载电成像数据 (假设get_ele_data_from_path函数已定义)
    dyna_data_1, depth1 = get_ele_data_from_path(path_dyna_1)
    stat_data_1, depth1 = get_ele_data_from_path(path_stat_1)
    dyna_data_2, depth2 = get_ele_data_from_path(path_dyna_2)
    stat_data_2, depth2 = get_ele_data_from_path(path_stat_2)

    # 数据检查2: 打印数据形状用于调试
    print(f"数据集1 - 深度形状: {depth1.shape}, 动态数据形状: {dyna_data_1.shape}, 静态数据形状: {stat_data_1.shape}")
    print(f"数据集2 - 深度形状: {depth2.shape}, 动态数据形状: {dyna_data_2.shape}, 静态数据形状: {stat_data_2.shape}")

    # show_Pic([dyna_data_1, stat_data_1, dyna_data_2, stat_data_2])
    r = np.random.random()
    if r < 0.5:
        return dyna_data_1, stat_data_1, depth1
    else:
        return dyna_data_2, stat_data_2, depth2


def fmi_data_save(fmi_image, depth_config=None, path_save=r''):
    """
        电成像数据保存函数

        功能: 将电成像数据和深度信息保存为多种格式（CSV、图像、文本）
              支持不同的深度配置方式

        参数:
        - fmi_image: 电成像数据矩阵 (深度点数 × 电极数)
        - depth_config: 深度配置，支持多种格式:
            - None: 使用默认深度序列 (0, 1, 2, ...)
            - tuple: (起始深度, 结束深度)，生成等间隔深度序列
            - np.array: 自定义深度序列，长度必须与fmi_image行数匹配
        - path_save: 保存路径，根据文件扩展名确定保存格式

        支持格式:
        - .csv: CSV表格格式，第一列为深度，其余为电成像值
        - .jpg/.png: 图像格式，仅保存电成像数据（不包含深度信息）
        - .txt: 文本格式，制表符分隔，包含深度和电成像数据

        返回:
        - 成功保存返回True，失败返回False
    """
    # ============================================================================
    # 1. 参数验证和预处理
    # ============================================================================

    # 验证输入数据
    if fmi_image is None or fmi_image.size == 0:
        print("错误: 电成像数据为空")
        return False

    if path_save is None or path_save == '':
        print("错误: 保存路径为空")
        return False

    print(path_save)
    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(path_save), exist_ok=True)

    # ============================================================================
    # 2. 深度数据生成
    # ============================================================================
    depth_data = None
    # 情况1: 无深度配置，使用默认序列 (0, 1, 2, ...)
    if depth_config is None:
        depth_data = np.arange(0, fmi_image.shape[0]).reshape(-1, 1)
    # 情况2: 深度配置为元组 (起始深度, 结束深度)
    elif isinstance(depth_config, tuple) and len(depth_config) == 2:
        start_depth, end_depth = depth_config
        depth_data = np.linspace(start_depth, end_depth, num=fmi_image.shape[0],
                                 dtype=np.float32).reshape(-1, 1)
        print(f"使用等间隔深度序列: {start_depth} 到 {end_depth}")
    # 情况3: 深度配置为自定义数组
    elif isinstance(depth_config, np.ndarray):
        # 验证深度数据形状匹配
        if depth_config.shape[0] != fmi_image.shape[0]:
            print(f"错误: 深度数据行数 {depth_config.shape[0]} 与图像行数 {fmi_image.shape[0]} 不匹配")
            return False
        depth_data = depth_config.reshape(-1, 1)  # 确保为列向量
        print(f"使用自定义深度序列: {depth_data[0, 0]} 到 {depth_data[-1, 0]}")
    elif isinstance(depth_config, pd.DataFrame):
        if depth_config.shape[0] != fmi_image.shape[0]:
            print(f"错误: 深度数据行数 {depth_config.shape[0]} 与图像行数 {fmi_image.shape[0]} 不匹配")
            return False
        depth_data = depth_config[depth_config.columns[0]].values
        depth_data = depth_data.reshape(-1, 1)  # 确保为列向量
        print(f"使用自定义深度序列: {depth_data[0, 0]} 到 {depth_data[-1, 0]}")


    # ============================================================================
    # 3. 数据合并和预处理
    # ============================================================================

    # 合并深度数据和电成像数据
    data_all = np.hstack([depth_data, fmi_image])
    # 数据验证
    if np.any(np.isnan(data_all)):
        print("警告: 数据包含NaN值，可能影响保存结果")
    if np.any(np.isinf(data_all)):
        print("警告: 数据包含无穷大值，可能影响保存结果")

    # ============================================================================
    # 4. 根据文件格式进行保存
    # ============================================================================
    try:
        # CSV格式保存
        if path_save.endswith('.csv'):
            return _save_csv_format(data_all, path_save)
        # JPG格式保存
        elif path_save.endswith('.jpg'):
            return _save_image_format(fmi_image, path_save, 'jpg')
        # PNG格式保存
        elif path_save.endswith('.png'):
            return _save_image_format(fmi_image, path_save, 'png')
        # TXT格式保存
        elif path_save.endswith('.txt'):
            return _save_text_format(data_all, path_save)
        # 不支持的文件格式
        else:
            print(f"错误: 不支持的文件格式: {os.path.splitext(path_save)[1]}")
            return False
    except Exception as e:
        print(f"保存文件时发生错误: {e}")
        return False


def _save_csv_format(data_all, path_save):
    """
    保存为CSV格式

    格式: 第一列为深度，其余列为电成像数据
    使用逗号分隔，包含列标题
    """
    try:
        # 创建列名
        n_cols = data_all.shape[1]
        column_names = ['Depth'] + [f'Electrode_{i}' for i in range(1, n_cols)]
        # 创建DataFrame
        df = pd.DataFrame(data_all, columns=column_names)
        # 设置显示精度
        pd.set_option('display.precision', 6)
        # 保存CSV文件
        df.to_csv(path_save, index=False, float_format='%.6f')
        print(f"成功保存CSV文件: {path_save}, 数据形状: {data_all.shape}")
        return True
    except Exception as e:
        print(f"保存CSV文件失败: {e}")
        return False

def _save_image_format(fmi_image, path_save, format_type):
    """
    保存为图像格式 (JPG/PNG)

    注意: 图像格式只保存电成像数据，不包含深度信息
    自动进行数据归一化到0-255范围
    """
    try:
        # 数据预处理: 归一化到0-255范围
        if fmi_image.dtype != np.uint8:
            # 计算数据范围
            data_min = np.min(fmi_image)
            data_max = np.max(fmi_image)
            if data_max > data_min:
                # 线性归一化
                normalized = (fmi_image - data_min) / (data_max - data_min) * 255
            else:
                # 数据为常数值
                normalized = np.full_like(fmi_image, 128, dtype=np.float32)
            image_8bit = normalized.astype(np.uint8)
        else:
            image_8bit = fmi_image
        # 保存图像
        if format_type == 'jpg':
            cv2.imwrite(path_save, image_8bit, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:  # png
            cv2.imwrite(path_save, image_8bit, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        print(f"成功保存{format_type.upper()}图像: {path_save}, 图像形状: {fmi_image.shape}")
        return True
    except Exception as e:
        print(f"保存{format_type.upper()}图像失败: {e}")
        return False


def _save_text_format(data_all, path_save):
    """
    保存为文本格式

    格式: 制表符分隔，第一列为深度，其余为电成像数据
    包含简单的文件头信息
    """
    try:
        # 创建文件头
        header_lines = [
            f"# 电成像数据文件",
            f"# 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# 数据维度: {data_all.shape[0]} 行 × {data_all.shape[1] - 1} 列 + 深度列",
            f"# 深度范围: {data_all[0, 0]:.4f} 到 {data_all[-1, 0]:.4f}",
            f"# 格式: 深度[制表符]电极1[制表符]电极2...",
            f"# Depth",
            f""
        ]

        # 写入文件
        with open(path_save, 'w', encoding='utf-8') as f:
            # 写入文件头
            for line in header_lines:
                f.write(line + '\n')
            # 写入数据
            np.savetxt(f, data_all, delimiter='\t', fmt='%.6f')
        print(f"成功保存文本文件: {path_save}, 数据形状: {data_all.shape}")
        return True
    except Exception as e:
        print(f"保存文本文件失败: {e}")
        return False


if __name__ == '__main__':
    # img_data, depth_data = get_ele_data_from_path(r'F:\DeepLData\target_stage1_small_big_mix\FMI_IMAGE\ZG_FMI_SPLIT\guan17-11_333_3746.5025_3747.1275_dyna.png')
    img_data, depth_data = get_ele_data_from_path(r'F:\DeepLData\FMI_SIMULATION\simu_FMI_exteact\1_fmi_dyna_hole_mask.png')
    print(img_data.shape, depth_data.shape, '\n', depth_data[:10, :], '\n', np.max(img_data), np.min(img_data))
    fmi_data_save(img_data, path_save=r'F:\DeepLData\FMI_SIMULATION\simu_FMI_exteact\text_save.txt')

