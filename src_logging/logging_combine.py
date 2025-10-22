import numpy as np
from scipy.spatial import cKDTree
import pandas as pd


def data_combine_table2col(data_main, table_vice, drop=True):
    """
    高效数据合并函数 - 优化版

    使用KD树进行最近邻搜索，大幅提升合并效率

    :param data_main: 主数据，n行m列的测井数据，第一列为DEPTH列
    :param table_vice: 表格数据，n*2数组，格式为[depth, Type]
    :param drop: 是否抛弃无匹配的数据
    :return: 合并后的测井数据数组
    """
    # 1. 数据预处理和验证
    # 检查表格数据是否有序
    if not np.all(np.diff(table_vice[:, 0]) >= 0):
        print("警告：表格深度数据未排序，正在自动排序...")
        sorted_indices = np.argsort(table_vice[:, 0])
        table_vice = table_vice[sorted_indices]

    # 提取深度列
    main_depths = data_main[:, 0].astype(np.float64)
    table_depths = table_vice[:, 0].astype(np.float64)
    table_types = table_vice[:, -1].astype(np.float64)

    # 2. 计算分辨率
    depth_table_resolution = get_resolution_by_depth(table_depths)
    depth_resolution = get_resolution_by_depth(main_depths)
    tolerance = depth_table_resolution / 2 + 0.001

    # 3. 使用KD树进行高效最近邻搜索
    # 创建KD树索引
    tree = cKDTree(table_depths.reshape(-1, 1))

    # 查询最近邻
    distances, indices = tree.query(main_depths.reshape(-1, 1), k=1, distance_upper_bound=tolerance)

    # 4. 创建合并结果数组
    # 初始化结果数组
    if drop:
        # 只保留有匹配的行
        valid_mask = distances <= tolerance
        valid_count = np.sum(valid_mask)
        result = np.empty((valid_count, data_main.shape[1] + 1))

        # 填充有效数据
        result[:, :-1] = data_main[valid_mask]
        result[:, -1] = table_types[indices[valid_mask]]
    else:
        # 保留所有行，无匹配的填充NaN
        result = np.empty((data_main.shape[0], data_main.shape[1] + 1))
        result[:, :-1] = data_main

        # 填充匹配的类型值
        valid_mask = distances <= tolerance
        result[:, -1] = np.where(valid_mask, table_types[indices], np.nan)

    # 5. 输出合并统计信息
    valid_count = np.sum(distances <= tolerance)
    print(f'数据合并统计: 主数据行数={data_main.shape[0]}, 表格数据行数={table_vice.shape[0]}')
    print(f'成功匹配行数={valid_count} ({valid_count / data_main.shape[0] * 100:.2f}%)')
    print(f'主数据深度范围=[{main_depths.min():.2f}, {main_depths.max():.2f}]')
    print(f'表格数据深度范围=[{table_depths.min():.2f}, {table_depths.max():.2f}]')
    print(f'合并后数据行数={result.shape[0]}')

    return result


def get_resolution_by_depth(depth_series):
    """
    计算深度序列的分辨率（最小非零间隔）

    :param depth_series: 深度序列（pd.Series）
    :return: 分辨率值
    """
    sorted_depths = depth_series.sort_values().unique()
    if len(sorted_depths) < 2:
        return 0.0

    diffs = np.diff(sorted_depths)
    non_zero_diffs = diffs[diffs > 0]
    non_zero_diffs = np.sort(non_zero_diffs)

    if len(non_zero_diffs) == 0:
        return 0.0

    num_non_zero_diffs = len(non_zero_diffs)
    return np.mean(non_zero_diffs[:int(num_non_zero_diffs*0.4)])

def combine_logging_data(
        data_main: pd.DataFrame = pd.DataFrame(),
        data_vice: list = [],
        depth_col: str = 'DEPTH',
        drop: bool = True
) -> pd.DataFrame:
    """
    合并多个测井数据DataFrame

    参数:
    data_main: 主数据DataFrame，包含深度列和其他测井数据
    data_vice: 副数据列表，每个元素是一个DataFrame，包含深度列和其他数据
    depth_col: 深度列名称，默认为'DEPTH'
    drop: 是否丢弃无法匹配的行
    tolerance: 深度匹配容差

    返回:
    合并后的DataFrame
    """
    # 1. 输入验证
    if not isinstance(data_main, pd.DataFrame) or data_main.empty:
        raise ValueError("主数据必须是非空的DataFrame")
    if not isinstance(data_vice, list):
        raise TypeError("副数据必须是列表")
    if depth_col not in data_main.columns:
        raise ValueError(f"主数据中缺少深度列 '{depth_col}'")

    # 2. 准备主数据
    # 确保深度列是数值类型
    if not pd.api.types.is_numeric_dtype(data_main[depth_col]):
        try:
            data_main[depth_col] = pd.to_numeric(data_main[depth_col], errors='coerce')
        except Exception:
            raise ValueError(f"无法将主数据的深度列 '{depth_col}' 转换为数值类型")

    # 删除深度列中的NaN值
    data_main = data_main.dropna(subset=[depth_col]).copy()

    # 3. 处理每个副数据
    merged_data = data_main.copy()
    all_matched_mask = np.zeros(len(merged_data), dtype=bool)  # 记录所有匹配情况

    for i, df_vice in enumerate(data_vice):
        if not isinstance(df_vice, pd.DataFrame) or df_vice.empty:
            print('\033[31m' + f"警告: 副数据 {i} 为空或不是DataFrame，跳过" + '\033[0m')
            continue

        if not df_vice.columns[0].lower().__contains__('depth'):
            print('\033[31m' + f"警告: 副数据 {i} 第零列为'{df_vice.columns[0]}'，不符合条件，跳过" + '\033[0m')
            continue

        # 确保深度列是数值类型
        if not pd.api.types.is_numeric_dtype(df_vice[df_vice.columns[0]]):
            try:
                df_vice[df_vice.columns[0]] = pd.to_numeric(df_vice[df_vice.columns[0]], errors='coerce')
            except Exception:
                print('\033[31m' + f"警告: 无法将副数据 {i} 的深度列 '{df_vice.columns[0]}' 转换为数值类型，跳过" + '\033[0m')
                continue

        # 删除深度列中的NaN值
        df_vice = df_vice.dropna(subset=[df_vice.columns[0]]).copy()
        tolerance = get_resolution_by_depth(df_vice[df_vice.columns[0]])*0.55
        # 4. 基于深度合并数据
        merged_data, matched_mask = merge_on_depth(
            merged_data,
            df_vice,
            depth_main=depth_col,
            depth_vice=df_vice.columns[0],
            suffix=f"_vice{i}",
            tolerance=tolerance
        )
        # 更新匹配掩码
        all_matched_mask = all_matched_mask | matched_mask

    # 5. 根据drop参数处理未匹配的行
    if drop and not all_matched_mask.all():
        print(f"丢弃 {len(merged_data) - all_matched_mask.sum()} 个未匹配的行")
        merged_data = merged_data[all_matched_mask].copy()

    # 6. 返回合并后的数据
    return merged_data


def merge_on_depth(
        df_main: pd.DataFrame,
        df_vice: pd.DataFrame,
        depth_main: str = 'DEPTH',
        depth_vice: str = 'DEPTH',
        suffix: str = "_vice",
        tolerance: float = 0.001
) -> tuple:
    """
    基于深度列合并两个DataFrame

    参数:
    df_main: 主数据DataFrame
    df_vice: 副数据DataFrame
    depth_col: 深度列名称
    suffix: 列名后缀，用于避免列名冲突
    tolerance: 深度匹配容差

    返回:
    (合并后的DataFrame, 匹配掩码)
    """
    # 提取深度值
    main_depths = df_main[depth_main].values.astype(np.float64)
    vice_depths = df_vice[depth_vice].values.astype(np.float64)

    # 对副数据深度排序
    if not np.all(np.diff(vice_depths) >= 0):
        sorted_idx = np.argsort(vice_depths)
        vice_depths = vice_depths[sorted_idx]
        df_vice = df_vice.iloc[sorted_idx].reset_index(drop=True)

    # 创建KD树进行最近邻搜索
    tree = cKDTree(vice_depths.reshape(-1, 1))

    # 查询最近邻
    distances, indices = tree.query(main_depths.reshape(-1, 1), k=1, distance_upper_bound=tolerance)
    # 创建匹配掩码
    matched_mask = distances <= tolerance

    # 添加副数据列到主数据
    for col in df_vice.columns:
        if col == depth_vice:
            continue                            # 跳过深度列

        # 处理列名冲突
        new_col = col
        if col in df_main.columns:
            new_col = f"{col}{suffix}"

        # 初始化新列为NaN
        df_main[new_col] = np.nan

        # 填充匹配的值
        if matched_mask.any():
            matched_values = df_vice[col].values[indices[matched_mask]]
            df_main.loc[matched_mask, new_col] = matched_values

    valid_count = len(matched_mask)
    print(f'数据合并统计: 主数据行数={df_main.shape[0]}, 副数据行数={df_vice.shape[0]}')
    print(f'成功匹配行数={valid_count} ({valid_count / df_main.shape[0] * 100:.2f}%)')
    print(f'主数据深度范围=[{main_depths.min():.2f}, {main_depths.max():.2f}]')
    print(f'表格数据深度范围=[{vice_depths.min():.2f}, {vice_depths.max():.2f}]')
    print(f'合并后数据行数={df_main.shape[0]}')

    return df_main, matched_mask


def test_combine_logging_data():
    """
    测试多测井数据合并接口
    """
    # 创建主数据
    depth_main = np.arange(100.0, 200.0, 0.5)
    data_main = pd.DataFrame({
        'DEPTH': depth_main,
        'GR': np.random.uniform(20, 150, len(depth_main)),
        'RT': np.random.uniform(0.2, 200, len(depth_main)),
        'DEN': np.random.uniform(1.5, 3.0, len(depth_main))
    })

    # 创建副数据1
    depth_vice1 = np.arange(100.0, 200.0, 1.0)
    data_vice1 = pd.DataFrame({
        'DEPTH': depth_vice1,
        'SP': np.random.uniform(-10, 10, len(depth_vice1)),
        'CALI': np.random.uniform(6.0, 16.0, len(depth_vice1))
    })

    # 创建副数据2
    depth_vice2 = np.arange(95, 180, 0.4)
    data_vice2 = pd.DataFrame({
        'DEPTH': depth_vice2,
        'CNL': np.random.uniform(0.1, 0.4, len(depth_vice2)),
        'PE': np.random.uniform(1.0, 5.0, len(depth_vice2))
    })

    # 创建副数据3（有列名冲突）
    depth_vice3 = np.arange(112, 210, 0.6)
    data_vice3 = pd.DataFrame({
        'DEPTH': depth_vice3,
        'GR': np.random.uniform(10, 100, len(depth_vice3)),  # 与主数据列名冲突
        'AC': np.random.uniform(50, 200, len(depth_vice3))
    })

    # 创建副数据4（无效数据）
    data_vice4 = pd.DataFrame()  # 空DataFrame
    data_vice5 = pd.DataFrame({'TIME': [1, 2, 3], 'VALUE': [4, 5, 6]})  # 缺少深度列

    # 合并数据
    combined_data = combine_logging_data(
        data_main=data_main,
        data_vice=[data_vice1, data_vice2, data_vice3, data_vice4, data_vice5],
        depth_col='DEPTH',
        drop=False
    )

    # 打印结果
    print("合并后的数据:")
    print(combined_data.head())
    print("\n列名:", combined_data.columns.tolist())
    print("形状:", combined_data.shape)

    # 测试drop=True
    print("\n测试drop=True:")
    combined_data_dropped = combine_logging_data(
        data_main=data_main,
        data_vice=[data_vice1, data_vice2, data_vice3],
        depth_col='DEPTH',
        drop=True
    )
    print("形状:", combined_data_dropped.shape)

    # 测试自定义深度列名
    print("\n测试自定义深度列名:")
    data_main_renamed = data_main.rename(columns={'DEPTH': 'Depth'})
    data_vice1_renamed = data_vice1.rename(columns={'DEPTH': 'Depth'})
    data_vice2_renamed = data_vice2.rename(columns={'DEPTH': 'Depth'})

    combined_data_custom = combine_logging_data(
        data_main=data_main_renamed,
        data_vice=[data_vice1_renamed, data_vice2_renamed],
        depth_col='Depth',
        drop=False
    )
    print("列名:", combined_data_custom.columns.tolist())

    return {
        'combined': combined_data,
        'dropped': combined_data_dropped,
        'custom': combined_data_custom
    }


# 运行测试
if __name__ == '__main__':
    test_results = test_combine_logging_data()
    print("\n测试完成!")