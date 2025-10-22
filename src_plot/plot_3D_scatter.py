import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns


def plot_3d_pca(df, title="3D PCA Visualization", figsize=(12, 10),
                elev=25, azim=45, point_size=20, alpha=0.7,
                legend_loc='best', save_path=None):
    """
    3D PCA数据可视化接口

    参数:
    df: DataFrame, 必须包含['PCA1', 'PCA2', 'PCA3', 'Type']列
    title: 图表标题
    figsize: 图表尺寸
    elev: 仰角 (垂直视角)
    azim: 方位角 (水平视角)
    point_size: 点大小
    alpha: 点透明度
    legend_loc: 图例位置
    save_path: 保存路径 (None则不保存)

    返回:
    matplotlib Figure对象
    """
    # 验证输入
    required_cols = ['PCA1', 'PCA2', 'PCA3', 'Type']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame必须包含列: {required_cols}")

    # 获取唯一类型和颜色映射
    unique_types = df['Type'].unique()
    n_types = len(unique_types)

    # 创建自定义颜色映射
    if n_types <= 8:
        # 使用Tableau调色板
        colors = sns.color_palette("tab10", n_colors=n_types)
    else:
        # 使用HSV色环生成更多颜色
        colors = sns.color_palette("hsv", n_colors=n_types)

    color_map = dict(zip(unique_types, colors))

    # 创建3D图形
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # 设置视角
    ax.view_init(elev=elev, azim=azim)

    # 绘制每个类型的数据点
    for type_val in unique_types:
        type_data = df[df['Type'] == type_val]
        ax.scatter(
            type_data['PCA1'],
            type_data['PCA2'],
            type_data['PCA3'],
            s=point_size,
            alpha=alpha,
            c=[color_map[type_val]],
            label=str(type_val)
        )

    # 自适应数据范围
    pad = 0.05  # 边距比例
    min_val = df[['PCA1', 'PCA2', 'PCA3']].min().min()
    max_val = df[['PCA1', 'PCA2', 'PCA3']].max().max()
    range_val = max_val - min_val
    padding = range_val * pad

    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)
    ax.set_zlim(min_val - padding, max_val + padding)

    # 设置标签
    ax.set_xlabel('PCA1', fontsize=12, labelpad=10)
    ax.set_ylabel('PCA2', fontsize=12, labelpad=10)
    ax.set_zlabel('PCA3', fontsize=12, labelpad=10)

    # 添加标题和图例
    plt.title(title, fontsize=16, pad=20)
    ax.legend(loc=legend_loc, fontsize=10)

    # 优化布局
    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")

    return fig


# 高级接口：支持交互式旋转
# def interactive_3d_pca(df, title="Interactive 3D PCA Visualization", XYZ=['PCA1', 'PCA2', 'PCA3'], Type='Type',
#                        figsize=(14, 12), point_size=15, alpha=0.7):
#     """
#     交互式3D PCA可视化 (使用Plotly)
#
#     参数:
#     df: DataFrame, 必须包含['PCA1', 'PCA2', 'PCA3', 'Type']列
#     title: 图表标题
#     figsize: 图表尺寸 (仅影响静态导出)
#     point_size: 点大小
#     alpha: 点透明度
#
#     返回:
#     plotly Figure对象
#     """
#     try:
#         import plotly.express as px
#     except ImportError:
#         raise ImportError("使用交互式可视化需要安装plotly: pip install plotly")
#
#     # 验证输入
#     required_cols = XYZ + [Type]
#     if not all(col in df.columns for col in required_cols):
#         raise ValueError(f"DataFrame必须包含列: {required_cols}")
#
#     # 创建交互式图表
#     fig = px.scatter_3d(
#         df,
#         x=XYZ[0],
#         y=XYZ[1],
#         z=XYZ[2],
#         color=Type,
#         title=title,
#         size_max=point_size,
#         opacity=alpha,
#         width=figsize[0] * 100,
#         height=figsize[1] * 100
#     )
#
#     # 更新布局
#     fig.update_layout(
#         scene=dict(
#             xaxis_title=XYZ[0],
#             yaxis_title=XYZ[1],
#             zaxis_title=XYZ[2]
#         ),
#         legend_title_text=Type,
#         margin=dict(l=0, r=0, b=0, t=40)
#     )
#
#     return fig

# 我对您的交互式3D PCA可视化函数进行了全面的美观优化，提升了视觉效果和用户体验：
def interactive_3d_pca(df, title="Interactive 3D PCA Visualization",
                                  XYZ=['PCA1', 'PCA2', 'PCA3'], Type='Type',
                                  figsize=(14, 14), point_size=10, alpha=0.8,
                                  color_theme='plotly', axis_grid=True,
                                  axis_labels=True, hover_info=True,
                                  legend_title="Type", legend_position='right',
                                  camera_view=None, save_html=None,
                                  bg_color='rgba(0,0,0,0)',
                                  grid_color='rgba(0, 0, 0, 0.5)',
                                  grid_width=1):
    """
    优化版交互式3D PCA可视化 (使用Plotly)

    参数:
    df: DataFrame, 必须包含指定的XYZ和Type列
    title: 图表标题
    XYZ: PCA维度列名列表，默认为['PCA1', 'PCA2', 'PCA3']
    Type: 分类列名，默认为'Type'
    figsize: 图表尺寸 (宽度, 高度)
    point_size: 点大小 (默认10)
    alpha: 点透明度 (默认0.8)
    color_theme: 颜色主题，可选'plotly', 'd3', 'ggplot2', 'seaborn'或自定义颜色映射
    bg_color: 背景颜色 (默认白色'#FFFFFF'或透明色'rgba(0,0,0,0)')
    axis_grid: 是否显示坐标轴网格 (默认True)
    axis_labels: 是否显示坐标轴标签 (默认True)
    hover_info: 是否显示悬停信息 (默认True)
    legend_title: 图例标题 (默认"Type")
    legend_position: 图例位置 ('right', 'top', 'bottom', 'left')
    camera_view: 初始相机视角，格式为{'up': {'x':0, 'y':0, 'z':1}, 'center': {...}, 'eye': {...}}
    save_html: HTML保存路径 (默认None不保存)
    grid_color: 网格线颜色 (默认'rgba(0, 0, 0, 0.5)')
    grid_width: 网格线宽度 (默认1)
    grid_dash: 网格线样式 ('solid', 'dot', 'dash', 'dashdot')
    返回:
    plotly Figure对象
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("使用交互式可视化需要安装plotly: pip install plotly")

    # 验证输入
    required_cols = XYZ + [Type]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame必须包含列: {required_cols}")

    # 创建自定义颜色映射
    unique_types = df[Type].unique()
    n_types = len(unique_types)

    # 处理颜色主题
    if isinstance(color_theme, str):
        if color_theme == 'plotly':
            colors = px.colors.qualitative.Plotly
        elif color_theme == 'd3':
            colors = px.colors.qualitative.D3
        elif color_theme == 'ggplot2':
            colors = px.colors.qualitative.G10
        elif color_theme == 'seaborn':
            colors = px.colors.qualitative.Pastel
        else:
            colors = px.colors.qualitative.Plotly  # 默认主题
    else:
        colors = color_theme  # 自定义颜色列表

    # 确保有足够颜色
    if n_types > len(colors):
        # 生成更多颜色
        colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
        colors = colors[:n_types]

    color_discrete_map = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_types)}

    # 创建交互式图表
    fig = px.scatter_3d(
        df,
        x=XYZ[0],
        y=XYZ[1],
        z=XYZ[2],
        color=Type,
        title=title,
        size_max=point_size,
        opacity=alpha,
        width=figsize[0] * 100,
        height=figsize[1] * 100,
        color_discrete_map=color_discrete_map,
        hover_data={col: True for col in df.columns} if hover_info else None
    )

    # 更新布局 - 更美观的设置
    fig.update_layout(
        scene=dict(
            xaxis_title=XYZ[0] if axis_labels else '',
            yaxis_title=XYZ[1] if axis_labels else '',
            zaxis_title=XYZ[2] if axis_labels else '',
            xaxis=dict(
                gridcolor=grid_color if axis_grid else 'rgba(0,0,0,0)',
                gridwidth=grid_width,
                backgroundcolor=bg_color,
                showspikes=False  # False可移除坐标轴上的尖刺
            ),
            yaxis=dict(
                gridcolor=grid_color if axis_grid else 'rgba(0,0,0,0)',
                gridwidth=grid_width,
                backgroundcolor=bg_color,
                showspikes=False
            ),
            zaxis=dict(
                gridcolor=grid_color if axis_grid else 'rgba(0,0,0,0)',
                gridwidth=grid_width,
                backgroundcolor=bg_color,
                showspikes=False
            ),
            bgcolor=bg_color  # 场景背景色
        ),
        legend_title_text=legend_title,
        legend=dict(
            title_font=dict(size=14, family='Arial'),
            font=dict(size=12, family='Arial'),
            orientation='v' if legend_position in ['left', 'right'] else 'h',
            x=1.05 if legend_position == 'right' else 0,
            y=1 if legend_position == 'top' else 0.5
        ),
        title=dict(
            text=title,
            x=0.5,  # 居中
            font=dict(size=20, family='Arial, sans-serif', color='#333333')
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor=bg_color,  # 整个图表背景色
        plot_bgcolor=bg_color
    )

    # 更新标记样式
    fig.update_traces(
        marker=dict(
            size=point_size,
            opacity=alpha,
            # line=dict(width=0.1, color='DarkSlateGrey')  # 添加边框增强区分度
        ),
        selector=dict(mode='markers')
    )

    # 设置相机视角
    if camera_view:
        fig.update_layout(scene_camera=camera_view)
    else:
        # 默认优化视角
        fig.update_layout(
            scene_camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.25, y=1.25, z=1.25)
            )
        )

    # 保存为HTML
    if save_html:
        fig.write_html(save_html)
        print(f"交互式图表已保存至: {save_html}")

    return fig




if __name__ == '__main__':
    # 生成示例数据
    np.random.seed(42)
    n_samples = 40000
    types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    data = {
        'PCA1': np.random.randn(n_samples),
        'PCA2': np.random.randn(n_samples) * 0.8,
        'PCA3': np.random.randn(n_samples) * 0.5,
        'Type': np.random.choice(types, n_samples)
    }
    df = pd.DataFrame(data)

    # # 使用接口可视化
    # fig = plot_3d_pca(
    #     df,
    #     title="PCA Components Distribution",
    #     elev=30,
    #     azim=120,
    #     point_size=15,
    #     alpha=0.6,
    #     save_path="pca_3d_visualization.png"
    # )
    # plt.show()


    # 使用交互式接口
    interactive_fig = interactive_3d_pca(
        df,
        title="Interactive PCA Visualization",
        XYZ=['PCA1', 'PCA2', 'PCA3'],
        Type='Type',
        point_size=10,
        alpha=0.7,
    )
    interactive_fig.show()

    # # 保存为HTML
    # interactive_fig.write_html("interactive_pca.html")