import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Callable
from pylab import mpl
import seaborn as sns
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False


def plot_clustering_heatmap(
        df: pd.DataFrame,
        target_col: list = None,  # 修改默认值为 None
        y_label: str = '数据',
        title: str = "Clustering Algorithm Accuracy Distribution",
        condition_formatter: Optional[Callable] = None,
        figsize: tuple = (10, 6),
        cmap: str = "YlGnBu",
        font_scale: float = 1.0,
        figure: Optional[plt.Figure] = None
) -> plt.Axes:
    """生成dataframe准确率分布热力图接口"""
    # 参数校验
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须为Pandas DataFrame")

    # 设置默认 target_col
    if target_col is None:
        # 默认选择除索引外的所有列
        target_col = [col for col in df.columns if col != '窗长']

    # 检查列是否存在
    missing_cols = [col for col in target_col if col not in df.columns]
    if missing_cols:
        raise ValueError(f"以下列不存在: {', '.join(missing_cols)}")

    # 创建画布（使用传入的figure或新建）
    if figure is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = figure
        fig.clear()  # 清除之前的绘图内容

    ax = fig.add_subplot(111)

    # 生成热力图核心逻辑
    heatmap = sns.heatmap(
        df[target_col],  # 使用有效的列
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Normed Influence'},
        annot_kws={"size": 12 * font_scale},
        vmin=0,
        vmax=1,
        ax=ax  # 指定在传入的axes上绘制
    )

    # 设置标题
    ax.set_title(title, pad=20, fontsize=14 * font_scale)
    ax.set_xlabel('Features', fontsize=10 * font_scale)
    ax.set_ylabel('Window Length', fontsize=10 * font_scale)

    # 格式化行标签
    if condition_formatter:
        y_labels = [condition_formatter(i) for i in df.index]
    else:
        # 使用索引作为标签
        y_labels = [str(i) for i in df.index]

    x_labels = list(df[target_col].columns)
    heatmap.set_yticklabels(y_labels, rotation=0)  # 水平显示标签
    heatmap.set_xticklabels(x_labels, rotation=45, ha='right')

    # 优化布局
    fig.tight_layout()
    return ax  # 返回axes对象

if __name__ == '__main__':
    # 创建示例数据
    data = {
        '窗长': [200, 220, 240, 260, 280, 300],
        '类别0': [0.85, 0.88, 0.90, 0.89, 0.87, 0.84],
        '类别1': [0.78, 0.82, 0.85, 0.83, 0.80, 0.78],
        '类别2': [0.92, 0.91, 0.93, 0.94, 0.92, 0.90],
        '平均': [0.85, 0.87, 0.89, 0.88, 0.86, 0.84]
    }
    df = pd.DataFrame(data)

    # 定义标签
    plot_labels = {
        'label': '不同窗长下的精度热力图',
        'x': '窗长参数',
        'y': '精度类别',
        'heatmap_feature':'准确率'
    }

    # 创建热力图
    plot_clustering_heatmap(
        df = df,
        title = "Clustering Algorithm Accuracy Distribution"
    )
    plt.show()