import pandas as pd
import numpy as np


def calculate_frequency_distribution(df, value_columns, type_column='TYPE', bins=50, density=False):
    """
    计算每个数值列在不同类别下的频数或概率密度分布

    参数:
    df -- 输入的DataFrame
    value_columns -- 需要计算分布的数值列列表
    type_column -- 分类列名 (默认为'TYPE')
    bins -- 分布区间数量 (默认为50)
    density -- 是否计算概率密度 (False为频数)

    返回:
    一个字典，包含每个数值列和每个类别的分布数据
    """
    # 获取所有类别
    categories = sorted(df[type_column].unique())

    results = {}

    for col in value_columns:
        # 确定全局范围
        min_val = df[col].min()
        max_val = df[col].max()

        # 创建统一的区间边界
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 创建结果DataFrame
        dist_df = pd.DataFrame({
            'Bin_Start': bin_edges[:-1],
            'Bin_End': bin_edges[1:],
            'Bin_Midpoint': bin_midpoints
        })

        # 计算每个类别的分布
        for category in categories:
            # 获取当前类别的数据
            category_data = df[df[type_column] == category][col]

            # 计算频数或概率密度
            hist, _ = np.histogram(category_data, bins=bin_edges, density=density)

            # 添加到结果DataFrame
            dist_df[f'TYPE_{category}_Frequency'] = hist

        results[col] = dist_df

    return results


def save_distribution_to_excel(results, filename):
    """
    将分布数据保存到Excel文件

    参数:
    results -- calculate_frequency_distribution函数返回的结果字典
    filename -- 输出的Excel文件名
    """
    with pd.ExcelWriter(filename) as writer:
        for col_name, df in results.items():
            sheet_name = col_name[:31]  # Excel工作表名称最多31字符
            df.to_excel(writer, sheet_name=sheet_name, index=False)


# 测试用例
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)

    # 创建不同类别下的正态分布数据
    data = {
        'COL1': np.concatenate([
            np.random.normal(100, 10, 100),  # TYPE 0
            np.random.normal(120, 15, 100),  # TYPE 1
            np.random.normal(80, 8, 100),  # TYPE 2
            np.random.normal(150, 20, 100),  # TYPE 3
            np.random.normal(90, 12, 100)  # TYPE 4
        ]),
        'COL2': np.concatenate([
            np.random.normal(50, 5, 100),  # TYPE 0
            np.random.normal(70, 8, 100),  # TYPE 1
            np.random.normal(40, 4, 100),  # TYPE 2
            np.random.normal(90, 10, 100),  # TYPE 3
            np.random.normal(60, 7, 100)  # TYPE 4
        ]),
        'TYPE': np.concatenate([
            np.full(100, 0),
            np.full(100, 1),
            np.full(100, 2),
            np.full(100, 3),
            np.full(100, 4)
        ])
    }

    test_df = pd.DataFrame(data)

    # 计算频数分布
    distribution_results = calculate_frequency_distribution(
        df=test_df,
        value_columns=['COL1', 'COL2'],
        type_column='TYPE',
        bins=32,
        density=False
    )

    # 打印COL1的分布数据
    print("COL1的频数分布数据:")
    print(distribution_results['COL1'].head())

    # 保存所有结果到Excel
    save_distribution_to_excel(distribution_results, "frequency_distribution.xlsx")
    print("\n频数分布数据已保存到 'frequency_distribution.xlsx'")