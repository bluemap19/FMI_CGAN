import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde, pearsonr
import matplotlib.gridspec as gridspec

from src_plot.plot_chinese_setting import set_ultimate_chinese_font


def plot_correlation_analyze(df, col_names, method='pearson', figsize=(14, 14),
                             return_matrix=True):
    """
    优化的相关性分析函数：
    - 对角线区域：绘制核密度估计图
    - 上三角区域：相关系数热力图
    - 下三角区域：二维核密度估计图（散点图+等高线）

    参数：
    df: 输入的DataFrame
    col_names: 需要分析相关性的列名列表
    method: 相关性计算方法 ('pearson', 'kendall', 'spearman')
    figsize: 图表大小
    return_matrix: 是否返回相关系数矩阵

    返回：
    correlation_matrix: 相关系数矩阵
    """
    # 验证输入
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df必须是pandas DataFrame")

    if not isinstance(col_names, list) or len(col_names) == 0:
        raise ValueError("col_names必须是非空列表")

    # 检查所有列是否存在
    missing_cols = [col for col in col_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"以下列在DataFrame中不存在: {missing_cols}")

    # df 进行Peason相关性系数计算，必须全部使用float格式的数据，这里把类别转换成float类型数据
    df[col_names] = df[col_names].astype(float)

    # 设置中文字体
    chinese_font_prop = set_ultimate_chinese_font()

    # 提取目标列
    target_df = df[col_names].copy()

    # 检查数据是否为空
    if target_df.empty:
        raise ValueError("目标数据为空，无法计算相关性")

    # 计算相关系数矩阵
    if method in ['pearson', 'kendall', 'spearman']:
        correlation_matrix = target_df.corr(method=method)

        # 设置归一化范围确保负相关也能显示
        vmin = 0
        vmax = 1
    else:
        raise ValueError(f"不支持的相关性计算方法: {method}。请选择 'pearson', 'kendall' 或 'spearman'")

    # 创建图形和布局
    fig = plt.figure(figsize=figsize, facecolor='white')
    n = len(col_names)

    # 使用GridSpec创建自定义布局
    gs = gridspec.GridSpec(n, n, wspace=0.1, hspace=0.1, figure=fig)

    # 创建自定义颜色映射
    cmap_main = LinearSegmentedColormap.from_list('correlation_cmap', [
        '#313695', '#4575b4', '#74add1', '#abd9e9',
        '#e0f3f8', '#ffffbf', '#fee090', '#fdae61',
        '#f46d43', '#d73027', '#a50026'
    ], N=256)

    # 用于密度图的自定义颜色映射
    density_cmap = ListedColormap(plt.cm.Blues(np.linspace(0.2, 1.0, 128)))

    # 1. 对角线区域：绘制单变量核密度估计图
    for i in range(n):
        # 对角线区域：创建直方图
        ax = plt.subplot(gs[i, i])
        # hist_axes.append(ax)

        # 获取当前列的数据
        col_name = col_names[i]
        data = target_df[col_name].dropna()

        # 绘制分布直方图
        sns.histplot(data, kde=False, ax=ax, color='#3498db', alpha=0.7)

        # 添加核密度估计曲线
        kde = gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 10)
        y = kde(x) * (len(data) * (data.max() - data.min()) / 15)  # 调整比例以匹配直方图
        ax.plot(x, y, color='#e74c3c', linewidth=2)

        # 设置标签
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])

        # 设置边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#7f8c8d')
            spine.set_linewidth(0.9)

        # 显示左上、右下角的标签数据
        if i == 0:
            ax.set_ylabel(col_name, rotation=90, fontsize=16)
        if i == n - 1:
            ax.set_xlabel(col_name, rotation=0, fontsize=16)



    # 2. 上三角区域：相关系数热力图
    for i in range(n):
        for j in range(i + 1, n):
            ax = fig.add_subplot(gs[i, j])
            corr_value = correlation_matrix.iloc[i, j]

            # 移除背景和边框
            ax.set_facecolor('none')
            ax.set_axis_off()
            # 计算颜色值（0-1范围内的归一化值）
            normalized_value = (abs(corr_value) - vmin) / (vmax - vmin)
            color = cmap_main(normalized_value)
            # 创建缩小的圆角矩形（box_size%大小）
            box_size = 0.8  # 盒子大小（原始尺寸的百分比）
            padding = (1 - box_size) / 2
            box = FancyBboxPatch(
                (padding, padding),  # 左下角坐标（在轴坐标中）
                box_size, box_size,  # 宽度和高度
                boxstyle=f"round,pad=0,rounding_size=0.1",  # 圆角设置
                edgecolor='none',  # 无边框
                facecolor=color,
                transform=ax.transAxes  # 使用轴坐标
            )
            ax.add_patch(box)
            # 添加相关系数值
            text_color = 'white' if (normalized_value >= 0.85 or normalized_value <= 0.15) else 'black'
            ax.text(0.5, 0.5, f"{corr_value:.2f}",
                    ha='center', va='center', fontsize=16,
                    color=text_color, transform=ax.transAxes)

            # # 计算颜色值
            # normalized_value = (abs(corr_value) - vmin) / (vmax - vmin)
            # color = cmap_main(normalized_value)
            #
            # # 设置背景颜色
            # ax.set_facecolor(color)
            #
            # # 添加相关系数值
            # text_color = 'white' if (normalized_value > 0.8 or normalized_value < 0.1) else 'black'
            # ax.text(0.5, 0.5, f"{corr_value:.2f}",
            #         ha='center', va='center', fontsize=16,
            #         color=text_color)
            #
            # # 移除坐标轴
            # ax.set_xticks([])
            # ax.set_yticks([])


    # 3. 下三角区域：绘制二维核密度估计图（等高线+散点）
    for i in range(1, n):
        for j in range(i):
            ax = fig.add_subplot(gs[i, j])

            # 获取当前两个变量
            col_x = col_names[j]
            col_y = col_names[i]
            x_data = target_df[col_x].dropna().values
            y_data = target_df[col_y].dropna().values

            # 计算密度并绘制等高线
            xx, yy = np.mgrid[x_data.min():x_data.max():100j,
                     y_data.min():y_data.max():100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])

            # 计算二维核密度
            try:
                kernel = gaussian_kde(np.vstack([x_data, y_data]))
                z = np.reshape(kernel(positions).T, xx.shape)

                # 绘制密度等高线
                contour = ax.contourf(xx, yy, z, 10, cmap=density_cmap, alpha=0.7)
                ax.contour(xx, yy, z, 5, colors='#333333', linewidths=0.5, alpha=0.7)

                # 添加散点图（仅显示部分点防止过度拥挤）
                if len(x_data) > 100:
                    idx = np.random.choice(len(x_data), 100, replace=False)
                    ax.scatter(x_data[idx], y_data[idx], s=8, color='#e74c3c',
                               alpha=0.3, edgecolors='none')
                else:
                    ax.scatter(x_data, y_data, s=8, color='#e74c3c',
                               alpha=0.3, edgecolors='none')
            except Exception:
                # 如果密度计算失败，使用散点图
                ax.scatter(x_data, y_data, s=8, color='#e74c3c', alpha=0.3)

            # 设置坐标轴
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(col_y, rotation=90, fontsize=16)
            if i == (n-1):
                ax.set_xlabel(col_x, rotation=0, fontsize=16)


    # 4. 添加颜色条
    cax = fig.add_axes([0.92, 0.55, 0.02, 0.3])  # [left, bottom, width, height]
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap_main, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('相关系数', fontproperties=chinese_font_prop)

    # 添加密度图颜色条
    cax_density = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    cbar_density = fig.colorbar(plt.cm.ScalarMappable(cmap=density_cmap), cax=cax_density)
    cbar_density.set_label('密度值', fontproperties=chinese_font_prop)

    # 添加主标题
    title = f"属性相关性分析 ({method.upper()}系数)"
    fig.suptitle(title, fontsize=16, fontproperties=chinese_font_prop, y=0.95)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # 为颜色条留出空间
    plt.show()

    # 返回相关系数矩阵
    if return_matrix:
        return correlation_matrix


if __name__ == '__main__':
    # 模拟数据（包含一定相关性）
    np.random.seed(42)
    n = 100

    # 创建具有相关性的数据
    base = np.random.normal(0, 1, n)

    data = {
        'STAT_ENT': base + np.random.normal(0, 0.3, n),
        'STAT_DIS': 2 * base + np.random.normal(0, 0.5, n),
        'STAT_CON': np.random.uniform(0, 10, n),
        'STAT_XY_HOM': base * 1.5 + np.random.normal(0, 0.4, n),
        'STAT_HOM': np.random.poisson(20, n),
        'STAT_XY_CON': base * 0.5 + np.random.normal(0, 0.6, n),
        'DYNA_DIS': np.random.lognormal(1, 0.3, n),
        'STAT_ENG': base + np.random.normal(0, 0.7, n),
    }

    df = pd.DataFrame(data)

    # 定义要分析的列
    COL_NAMES = ['STAT_ENT', 'STAT_DIS', 'STAT_CON', 'STAT_XY_HOM',
                 'STAT_HOM', 'STAT_XY_CON', 'DYNA_DIS', 'STAT_ENG']

    # 调用接口进行分析
    corr_matrix = plot_correlation_analyze(
        df=df,
        col_names=COL_NAMES,
        method='pearson',
        figsize=(14, 14)
    )

    # 查看相关系数矩阵
    print("\n相关系数矩阵:")
    print(corr_matrix)

