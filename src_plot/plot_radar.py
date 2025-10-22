import numpy as np
import matplotlib.pyplot as plt

# 全局设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 支持中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 现在使用Python做一个雷达图绘制接口，要求如下：
# 接口输入要求：
# data_list：存放雷达图数据，list类型，每一个元素都包含M长度的矩阵
# radar_str: list类型，雷达图显示每一个子元素对应的字符串表示，即属性字符串
# pic_order='12' 雷达图的放置格式MN为M行N列，严格限制其与data_list长度一致
# figure=(16, 9)：整个图的图像长宽比，用户自己设置，也要与pic_order相对应
# pic_str=[]：每一个子雷达图对应的字符串设置，默认为['Radar1', 'Radar2', 'Radar3', 'Radar4']
# title='title'：整个图的名称


def draw_radar_chart(data_list, radar_str, pic_order='12',
                     figure=(16, 9), pic_str=None, title='Radar Chart',
                     norm=False):
    """
    绘制多个雷达图的接口

    参数:
    data_list - 雷达图数据列表，每个元素是包含M个数值的列表或数组
    radar_str - 雷达图属性标签列表，长度为M
    pic_order - 子图排列方式，格式为"MN" (M行N列)
    figure - 图像尺寸 (宽, 高)
    pic_str - 每个子图的标题列表
    title - 整个图表的总标题
    norm - 是否进行归一化处理 (True/False)
    """

    # 验证输入参数
    M = len(data_list)
    if M == 0:
        raise ValueError("data_list不能为空")

    # 验证pic_order格式
    if len(pic_order) != 2:
        raise ValueError("pic_order格式应为'MN'，如'22'表示2x2")

    # 计算行列数
    rows = int(pic_order[0])
    cols = int(pic_order[1])

    # 验证子图数量与数据长度匹配
    total_subplots = rows * cols
    if M > total_subplots:
        raise ValueError(f"pic_order={pic_order}仅支持{total_subplots}个子图，但提供了{M}组数据")

    # 设置默认子图标题
    if pic_str is None:
        pic_str = [f"Radar {i + 1}" for i in range(M)]
    elif len(pic_str) < M:
        # 补充不足的标题
        default_titles = [f"Radar {i + 1}" for i in range(len(pic_str), M)]
        pic_str.extend(default_titles)

    # 属性数量
    num_attributes = len(radar_str)
    if num_attributes < 3:
        raise ValueError("至少需要3个属性才能绘制雷达图")

    # 角度计算（等分圆周）
    angles = np.linspace(0, 2 * np.pi, num_attributes, endpoint=False).tolist()
    angles += angles[:1]  # 闭合多边形

    # 创建画布
    fig, axes = plt.subplots(rows, cols, figsize=figure,
                             subplot_kw=dict(polar=True),
                             constrained_layout=True)

    # 设置总标题
    if title:
        fig.suptitle(title, fontsize=20, fontweight='bold')

    # 处理子图数组
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 颜色列表
    colors = plt.cm.tab10(np.linspace(0, 1, M))

    for idx in range(total_subplots):
        # 只在前M个子图中绘制数据
        if idx < M:
            data = data_list[idx]
            sub_title = pic_str[idx]

            # 处理归一化选项
            if norm:
                # 归一化处理
                min_val = np.min(data)
                max_val = np.max(data)
                range_val = max_val - min_val

                if range_val > 0:
                    # 避免除零错误
                    scaled_data = [(val - min_val) / range_val for val in data]
                else:
                    # 所有值相同的情况
                    scaled_data = [0.5 for val in data]

                # 归一化数据闭合处理
                plot_data = scaled_data + [scaled_data[0]]
            else:
                # 不进行归一化
                # 直接使用原始数据，并添加第一个点来闭合多边形
                if isinstance(data, (list)):
                    plot_data = data + [data[0]]  # 关键修改：添加闭合点
                elif isinstance(data, (np.ndarray)):
                    plot_data = list(data) + [data[0]]


            ax = axes[idx]

            # 绘制雷达图边界
            ax.set_thetagrids(np.degrees(angles[:-1]), radar_str, fontsize=28)
            ax.set_rlabel_position(30)  # 径向标签位置

            # 设置雷达图背景
            ax.spines['polar'].set_visible(True)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            # 绘制数据线 - 现在angles和plot_data长度一致
            ax.plot(angles, plot_data, color=colors[idx], linewidth=2, linestyle='solid')

            # 填充颜色
            ax.fill(angles, plot_data, color=colors[idx], alpha=0.25)

            # 设置每个子图标题
            ax.set_title(sub_title, fontsize=12, pad=5)

            # 设置径向刻度范围（固定为0-1.1）
            ax.set_ylim(0, 1.1)

            # 添加数据值标签（使用原始数据）
            y_offset = 0.05
            for i, value in enumerate(data):
                angle = angles[i]  # 使用原始角度位置

                # 归一化后需要调整标签位置
                if norm:
                    if min_val != max_val:
                        label_y = (value - min_val) / (max_val - min_val) + y_offset
                    else:
                        label_y = 0.5 + y_offset
                else:
                    # 不归一化时使用原始值
                    label_y = value + y_offset

                # ax.text(angle, label_y, f"{value:.2f}",
                #         color=colors[idx], ha='center', va='bottom', fontsize=10)
        else:
            # 隐藏未使用的子图
            axes[idx].set_visible(False)

    return fig, axes


# 测试数据
if __name__ == "__main__":
    # 属性名称列表
    attributes = ['属性A', '属性B', '属性C', '属性D', '属性E']

    # 4组雷达图数据
    data1 = [4.2, 3.9, 4.5, 3.8, 4.1]
    data2 = [3.5, 4.2, 3.7, 4.0, 3.9]
    data3 = [2.8, 3.2, 3.0, 3.5, 2.9]
    data4 = [4.0, 4.5, 4.2, 4.3, 4.4]

    # 子图标题
    sub_titles = ['数据分析师', '机器学习工程师', '前端开发', '数据工程师']

    # 测试不进行归一化的情况
    print("测试归一化:")
    fig1, axes1 = draw_radar_chart(
        data_list=[data1, data2, data3, data4],
        radar_str=attributes,
        pic_order='14',
        figure=(22, 5),
        pic_str=sub_titles,
        title='技术团队技能评估雷达图 (无归一化)',
        norm=True
    )
    plt.savefig('radar_charts.png', dpi=300, bbox_inches='tight')
    plt.show()