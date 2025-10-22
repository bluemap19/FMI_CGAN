from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from seaborn import color_palette
from typing import Optional
from src_random_data.create_random_data import get_random_logging_dataframe
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_boxes( df: pd.DataFrame = pd.DataFrame(),
                input_names: list = [],
                target_col='',
                plot_string='',
                target_col_dict: dict = {},
                figsize: Optional[tuple] = (10, 5),):
    """
    进行数据散步图绘制，制作不同类别数据分布
    :param df: 数据体，只能是dataframe
    :param input_names: 输入属性列名称，对应输入属性列
    :param target_col: 分类列名称，对应类别分类列信息
    :param plot_string: 图形名称title
    :param target_col_dict: 类别是int分类类别型，那就需要替换成 字符型，这个是替换字典信息
    :param figsize: 长宽比
    :return:
    """
    # 边界条件1: 检查输入数据框是否为空
    if df.empty:
        print("Warning: 输入数据框为空，无法绘制图形。")
        return

    # 边界条件2: 检查input_names是否为空列表
    if not input_names:
        print("Warning: input_names为空列表，请提供需要绘制的列名。")
        return

    # 边界条件3: 检查target_col是否为空或不在数据框中
    if not target_col or target_col not in df.columns:
        print(f"Warning: target_col '{target_col}' 无效或不在数据框列中。")
        return

    # 边界条件4: 检查input_names中的列是否都在数据框中
    missing_cols = [col for col in input_names if col not in df.columns]
    if missing_cols:
        print(f"Warning: 以下列不在数据框中: {missing_cols}")
        return

    # 数据预处理增强
    data = df[input_names].copy()
    # 判断target_col列是否是数值类型的数据
    if pd.api.types.is_numeric_dtype(df[target_col]):
        # 如果是空字典，初始化一个
        if not target_col_dict:
            target_col_dict = {0: 'type1', 1: 'type2', 2: 'type3', 3: 'type4',
                               4: 'type5', 5: 'type6', 6: 'type7', 7: 'type8',
                               8: 'type9', 9: 'type10', 10: 'type11'}
            data[target_col] = df[target_col].astype(int)
            data[target_col] = data[target_col].map(target_col_dict)
            target_col_dict_temp = target_col_dict.copy()
        else:  # 不是空字典的话，需要判断键值对是不是反的
            target_col_dict_temp = target_col_dict.copy()
            # 判断 元素 0  是否在字典target_col_dict的键keys中，在的话说明键值对没有反，不在的话说明反了，要把反转的添加进去
            if 0 not in list(target_col_dict.keys()):
                dict_temp = {v: k for k, v in target_col_dict.items()}
                target_col_dict_temp = dict_temp
                print(target_col_dict_temp)
            # 先把类别数据转换成int格式
            data[target_col] = df[target_col].astype(int)
            # 再通过字典进行映射
            data[target_col] = data[target_col].map(target_col_dict_temp)
    # target_col列不是数值类型，也要把目标列加上
    else:
        data[target_col] = df[target_col]
        target_col_dict_temp = {}
        # 如果是空字典，初始化一个
        if not target_col_dict:
            all_type = list(np.unique(data[target_col]))
            for i in range(len(all_type)):
                target_col_dict_temp[i] = all_type[i]

    # # 方案一：现代专业地质色系（推荐）
    # geology_colors = [
    #     '#2E86AB',  # 深海蓝 - 中GR长英黏土质
    #     '#A23B72',  # 紫红色 - 中低GR长英质
    #     '#F18F01',  # 琥珀色 - 富有机质长英质
    #     '#C73E1D',  # 赭红色 - 富有机质黏土质
    #     '#4CB944',  # 矿物绿 - 高GR富凝灰长英质
    #     '#6D6875'  # 高级灰 - 其他类型
    # ]
    geology_colors = [
        '#4B0082',  # 中GR长英黏土质（靛青）
        '#32CD32',  # 中低GR长英质（亮绿）
        '#FF4500',  # 富有机质长英质（橙红）
        '#8B4513',  # 富有机质黏土质（赭石）
        '#00BFFF',  # 高GR富凝灰长英质（深天蓝）
        '#A9A9A9'  # 其他类型（灰色）
    ]
    # # 方案二：柔和地质色调
    # geology_colors = [
    #     '#5B8E7D',  # 青绿色
    #     '#BC4B51',  # 陶土红
    #     '#F4A259',  # 沙黄色
    #     '#5D7592',  # 灰蓝色
    #     '#8FBC94',  # 苔藓绿
    #     '#A67F8E'  # 淡紫色
    # ]
    # # 方案三：科学出版物风格
    # geology_colors = [
    #     '#4056A1',  # 深蓝色
    #     '#D79922',  # 金黄色
    #     '#F13C20',  # 珊瑚红
    #     '#61825F',  # 橄榄绿
    #     '#815B8F',  # 紫罗兰
    #     '#A8A8A8'  # 中性灰
    # ]
    # 获取唯一类别数量
    n_categories = len(data[target_col].unique())
    palette = color_palette(geology_colors, n_colors=len(target_col_dict_temp))

    # 边界条件5: 检查是否有足够的数据绘制箱型图
    if len(data) < 5:  # 箱型图通常需要至少5个数据点才能有意义
        print("Warning: 数据量不足，无法绘制有意义的箱型图。")
        return

    # 计算需要的子图行数
    n_plots = len(input_names)

    # 创建图形和子图
    fig, axes = plt.subplots(nrows=1, ncols=n_plots, figsize=figsize, sharex=True)

    # 如果只有一个子图，将axes转换为列表形式以便统一处理
    if n_plots == 1:
        axes = [axes]

    # 设置总标题
    if plot_string:
        fig.suptitle(plot_string, fontsize=16)

    # 循环绘制每个input_name的箱型图
    for i, input_name in enumerate(input_names):
        ax = axes[i]
        # 边界条件6: 检查当前列是否有足够非空值
        if data[input_name].count() < 5:
            print(f"Warning: 列 '{input_name}' 数据量不足，跳过绘制。")
            ax.text(0.5, 0.5, f"数据不足\n无法绘制{input_name}",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            continue

        # # 绘制美化版箱型图
        box_plot = sns.boxplot(
            x=target_col,  # 控制X轴：使用哪个列作为分类依据
            y=input_name,  # 控制Y轴：使用哪个列作为数值变量
            data=data,  # 指定数据源：使用哪个DataFrame
            ax=ax,  # 指定绘制到哪个子图上
            palette=palette,  # 颜色方案：控制不同类别的颜色
            width=0.7,  # 箱体宽度：0-1之间，控制箱型图的宽度
            linewidth=0.8,  # 线条粗细：控制箱型图边框线的粗细
            fliersize=3,  # 异常点大小：控制异常值标记的尺寸
            showmeans=True,  # 是否显示均值：True表示显示平均值点
            meanprops={  # 均值点样式设置：
                "marker": "o",  # 点形状：圆形
                "markerfacecolor": "white",  # 点填充色：白色
                "markeredgecolor": "green",  # 点边缘色：黑色
                "markersize": "4"  # 点大小：6磅
            }
        )

        # 添加轻微抖动的小提琴图背景，增强可视化效果
        sns.violinplot(
            x=target_col,  # X轴分类列
            y=input_name,  # Y轴数值列
            data=data,  # 数据源
            ax=ax,  # 子图位置
            palette=palette,  # 颜色方案（与箱型图一致）
            alpha=0.8,  # 透明度：0-1之间，0.2表示20%不透明
            inner=None  # 内部显示：None表示不显示内部元素
        )

        # 设置标题和标签
        ax.set_title(input_name, fontsize=12, fontweight='bold', pad=10)
        # ax.set_ylabel(input_name, fontsize=12, fontweight='bold')

        # 旋转x轴标签以避免重叠
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

        # 添加网格线以便更好地读取数值
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')

        # 设置背景色为更柔和的颜色
        ax.set_facecolor('#f8f9fa')

        # 美化边框
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(1)

    # 调整子图间距
    plt.tight_layout()

    # 显示图形
    plt.show()





if __name__ == '__main__':
    logging_data = get_random_logging_dataframe(curve_name=['#DEPTH', 'SP', 'AC', 'CNL', 'DEN', 'Type'], logging_resolution=0.1, dep_start=100, dep_end=200)
    print(logging_data.shape)           # ----->(4000, 6)
    plot_boxes(logging_data, ['AC', 'CNL', 'DEN', 'SP'], 'Type', target_col_dict={'中GR长英黏土质': 0, '中低GR长英质': 1, '富有机质长英质': 2, '富有机质黏土质': 3, '高GR富凝灰长英质': 4, 'asdadsadas':5}, figsize=(13, 4))