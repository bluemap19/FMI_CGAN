import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from src_logging import get_resolution_by_depth

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_well_logs(data: pd.DataFrame,
                        depth_col: str = 'Depth',
                        curve_cols: list = ['L1', 'L2', 'L3', 'L4'],
                        type_cols: list = [],
                        figsize: tuple = (12, 10),
                        colors: list = ['#FF0000', '#00FF00', '#0000FF', '#00FFFF',
                                        '#FF00FF', '#8000FF', '#00FF80', '#FF0080',
                                        '#FFA500', '#FFFF00'],
                        legend_dict: dict = {},
                        figure=None):
    """
    测井数据可视化接口 - 交互功能优化版

    功能变更：
    - 鼠标滚轮：调整深度位置（上下滚动）
    - 滑动条：调整窗口大小（视野范围）
    """
    # 设置图像绘制底部位置
    logging_bottle_with_legend = 0.065
    logging_bottle_without_legend = 0.03

    # ===================== 1. 初始化阶段 =====================
    # 数据验证
    required_cols = [depth_col] + curve_cols + type_cols
    if not set(required_cols).issubset(data.columns):
        missing = set(required_cols) - set(data.columns)
        raise ValueError(f"数据缺少必要列: {missing}")

    # 创建图形和子图
    n_plots = len(curve_cols) + len(type_cols)
    if figure is None:
        fig, axs = plt.subplots(
            1, n_plots,
            figsize=figsize,
            sharey=True,
            gridspec_kw={'wspace': 0.0}
        )
    else:
        fig = figure
        axs = fig.subplots(1, n_plots, sharey=True, gridspec_kw={'wspace': 0.0})

    # 调整布局
    legend_adjustment = logging_bottle_with_legend if legend_dict else logging_bottle_without_legend
    plt.subplots_adjust(
        left=0.04, right=0.96,
        bottom=legend_adjustment,
        top=0.95,
        wspace=0.0
    )

    # 数据预处理
    data = data.sort_values(depth_col).reset_index(drop=True)
    depth_min = data[depth_col].min()
    depth_max = data[depth_col].max()
    # 初始显示窗口大小为总深度的10%
    window_size = (depth_max - depth_min) * 0.1
    resolution = get_resolution_by_depth(data[depth_col].dropna().values)

    # 初始深度位置（显示窗口顶部）
    depth_position = depth_min

    # 分类宽度配置
    if type_cols:
        type_data = data[type_cols].values
        unique_types = np.unique(type_data[~np.isnan(type_data)])
        type_count = len(unique_types)

        litho_width_config = {}
        for i, t in enumerate(sorted(unique_types)):
            litho_width_config[int(t)] = (i + 1) / type_count

        if type_count == 1:
            litho_width_config[unique_types[0]] = 1.0
    else:
        litho_width_config = {}

    # ===================== 2. 创建窗口大小滑动条 =====================
    # 关键修改：创建窗口大小滑动条（取代深度位置滑动条）
    slider_bottle_adjustment = legend_adjustment
    slider_length_adjustment = 0.95 - slider_bottle_adjustment
    ax_slider = plt.axes([0.965, slider_bottle_adjustment, 0.035, slider_length_adjustment])

    # 计算窗口大小范围
    min_window_size = (depth_max - depth_min) * 0.01  # 最小窗口大小（1%）
    max_window_size = depth_max - depth_min  # 最大窗口大小（100%）

    # 创建窗口大小滑动条
    window_size_slider = Slider(
        ax=ax_slider,
        label='',
        valmin=min_window_size,
        valmax=max_window_size,
        valinit=window_size,
        valstep=(max_window_size - min_window_size) / 100,
        orientation='vertical'
    )

    # 添加垂直文本标签
    ax_slider.text(
        x=0.5, y=0.5,
        s='Window Size(m)',
        rotation=270,
        ha='center', va='center',
        transform=ax_slider.transAxes,
        fontsize=10
    )

    # ===================== 3. 绘制测井曲线 =====================
    def init_curves():
        plots = []
        title_margin = 0.001
        title_height = 0.04

        for i, col in enumerate(curve_cols):
            ax = axs[i]
            orig_pos = ax.get_position()

            title_bbox = [
                orig_pos.x0 + title_margin,
                orig_pos.y0 + orig_pos.height + title_margin,
                orig_pos.width - 2 * title_margin,
                title_height
            ]

            title_rect = plt.Rectangle(
                (title_bbox[0], title_bbox[1]),
                title_bbox[2], title_bbox[3],
                transform=fig.transFigure,
                facecolor='#f5f5f5',
                edgecolor='#aaaaaa',
                linewidth=1,
                clip_on=False,
                zorder=10
            )
            fig.add_artist(title_rect)

            title_text = plt.Text(
                title_bbox[0] + title_bbox[2] / 2,
                title_bbox[1] + title_bbox[3] / 2,
                col + '-',
                fontsize=12,
                fontweight='bold',
                color=colors[i % len(colors)],
                ha='center', va='center',
                transform=fig.transFigure,
                clip_on=False,
                zorder=11
            )
            fig.add_artist(title_text)

            line, = ax.plot(
                data[col],
                data[depth_col],
                color=colors[i % len(colors)],
                lw=1.0,
                linestyle='-'
            )

            ax.set_title(col, fontsize=10, pad=5)
            ax.grid(True, alpha=0.3)
            min_temp = data[col].min() * 0.95 if data[col].min() >= 0 else data[col].min() * 1.05
            max_temp = data[col].max() * 1.05 if data[col].max() >= 0 else data[col].max() * 0.95
            ax.set_xlim(min_temp, max_temp)
            ax.invert_yaxis()

            if i == 0:
                ax.set_ylim([depth_min, depth_max])
                ax.yaxis.set_major_locator(plt.MaxNLocator(10))
                ax.tick_params(axis='y', labelsize=8, rotation=90, pad=1)
            else:
                ax.set_ylim(None)
                ax.tick_params(left=False, labelleft=False)

        return plots

    plots = init_curves()

    # ===================== 4. 绘制分类面板 =====================
    def plot_classification(ax, depth, litho):
        if math.isnan(litho) or litho < 0:
            return

        litho = int(litho)
        xmax = litho_width_config.get(litho, 0.1)

        ax.axhspan(
            depth - resolution / 2,
            depth + resolution / 2,
            xmin=0, xmax=xmax,
            facecolor=colors[litho % len(colors)],
            edgecolor='none',
            linewidth=0,
            clip_on=False
        )

    def init_class_panels():
        class_axes = []

        for i, col in enumerate(type_cols):
            ax_idx = len(curve_cols) + i
            ax = axs[ax_idx]

            orig_pos = ax.get_position()
            title_bbox = [
                orig_pos.x0 + 0.001,
                orig_pos.y0 + orig_pos.height + 0.001,
                orig_pos.width - 0.002,
                0.04
            ]

            title_rect = plt.Rectangle(
                (title_bbox[0], title_bbox[1]),
                title_bbox[2], title_bbox[3],
                transform=fig.transFigure,
                facecolor='#f5f5f5',
                edgecolor='#aaaaaa',
                linewidth=1,
                clip_on=False,
                zorder=10
            )
            fig.add_artist(title_rect)

            title_text = plt.Text(
                title_bbox[0] + title_bbox[2] / 2,
                title_bbox[1] + title_bbox[3] / 2,
                col,
                fontsize=12,
                fontweight='bold',
                color='#222222',
                ha='center', va='center',
                transform=fig.transFigure,
                clip_on=False,
                zorder=11
            )
            fig.add_artist(title_text)

            for depth, litho in zip(data[depth_col], data[col]):
                plot_classification(ax, depth, litho)

            ax.set_xticks([])
            ax.set_title(f"{col}", fontsize=10, pad=5)
            ax.set_ylim(None)
            ax.tick_params(left=False, labelleft=False)
            ax.invert_yaxis()

            class_axes.append(ax)

        return class_axes

    class_axes = init_class_panels() if type_cols else []

    # ===================== 5. 动态更新函数 =====================
    def update_display():
        """更新显示的函数"""
        # 计算显示范围
        top = depth_position
        bottom = depth_position + window_size

        # 更新所有子图深度范围
        for ax in axs:
            ax.set_ylim(bottom, top)

        # 更新分类显示
        if type_cols:
            visible_data = data[(data[depth_col] >= top) & (data[depth_col] <= bottom)]
            for i, col in enumerate(type_cols):
                ax = class_axes[i]
                ax.clear()
                ax.set_xticks([])
                ax.set_title(f"{col}", fontsize=10, pad=5)

                for depth, litho in zip(visible_data[depth_col], visible_data[col]):
                    plot_classification(ax, depth, litho)

        # 添加深度位置指示器
        if hasattr(fig, 'depth_indicator'):
            fig.depth_indicator.set_text(f"当前深度: {depth_position:.1f} - {bottom:.1f} m")
        else:
            fig.depth_indicator = fig.text(
                0.5, 0.97,
                f"当前深度: {depth_position:.1f} - {bottom:.1f} m",
                ha='center', va='top',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2')
            )

        fig.canvas.draw_idle()

    # ===================== 6. 事件绑定 =====================
    # 6.1 窗口大小滑动条事件
    def on_window_size_change(val):
        """窗口大小改变事件"""
        nonlocal window_size, depth_position  # 在函数开头声明所有需要修改的非局部变量
        window_size = val
        # 确保深度位置在有效范围内
        if depth_position > depth_max - window_size:
            depth_position = depth_max - window_size
        update_display()

    window_size_slider.on_changed(on_window_size_change)

    # 6.2 鼠标滚轮事件 - 关键修改：滚动调整深度位置
    def on_scroll(event):
        """鼠标滚轮事件 - 调整深度位置"""
        nonlocal depth_position
        # 计算滚动步长（窗口大小的10%）
        step = window_size * 0.1

        # 确定滚动方向
        if event.button == 'up':  # 向上滚动
            depth_position -= step
        elif event.button == 'down':  # 向下滚动
            depth_position += step

        # 确保深度位置在有效范围内
        depth_position = np.clip(depth_position, depth_min, depth_max - window_size)

        # 更新显示
        update_display()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # ===================== 7. 创建统一图例 =====================
    if legend_dict:
        n_items = len(legend_dict)
        legend_height = 0.02
        legend_width = 0.8

        legend_ax = plt.axes([0.1, 0.01, legend_width, legend_height], frameon=False)
        legend_ax.set_axis_off()

        handles = []
        labels = []
        sorted_keys = sorted(legend_dict.keys())

        for key in sorted_keys:
            patch = plt.Rectangle((0, 0), 1, 1, facecolor=colors[key % len(colors)], edgecolor='none')
            handles.append(patch)
            labels.append(legend_dict[key])

        legend = legend_ax.legend(
            handles, labels,
            loc='center',
            ncol=n_items,
            frameon=True,
            framealpha=0.8,
            fancybox=True,
            fontsize=10,
            handlelength=1.5,
            handleheight=1.5,
            handletextpad=0.5,
            borderpad=0.5,
            columnspacing=1.0
        )

        frame = legend.get_frame()
        frame.set_facecolor('#f8f8f8')
        frame.set_edgecolor('#aaaaaa')

    # ===================== 8. 初始显示 =====================
    update_display()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    sample_data = pd.DataFrame(
        np.hstack([
            np.linspace(300, 400, 1000).reshape(-1, 1),
            np.random.randn(1000, 4),
            np.random.choice([0, 1, 2, 3], size=(1000, 4))
        ]),
        columns=['Depth', 'L1', 'L2', 'L3', 'L4', 'D2', 'D4', 'D3', 'D5']
    )

    # 调用可视化接口
    visualize_well_logs(
        data=sample_data,
        depth_col='Depth',
        curve_cols=['L1', 'L2', 'L3', 'L4'],
        type_cols=['D2', 'D5'],
        legend_dict={0: 'Type0', 1: 'Type1', 2: 'Type2', 3: 'Type3'}
    )