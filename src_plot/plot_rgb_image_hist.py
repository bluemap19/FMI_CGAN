import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec


def plot_hist(ax_hist, channel_data, color='red', channel_name=''):
    """
    在给定的轴上绘制通道的直方图

    参数:
    ax_hist: matplotlib轴对象
    channel_data: 通道数据 (2D数组)
    color: 直方图颜色
    channel_name: 通道名称 (用于标题)
    """
    # 绘制直方图
    ax_hist.hist(channel_data.ravel(), bins=256, color=color, alpha=0.7, label=color)

    # 添加完整坐标轴和边框
    ax_hist.set_title(f'{channel_name} Channel Histogram', fontsize=10)
    ax_hist.set_xlim(0, 255)
    ax_hist.set_ylim(bottom=0)
    ax_hist.grid(True, linestyle='--', alpha=0.3)

    # 显示所有边框
    ax_hist.spines['top'].set_visible(True)
    ax_hist.spines['right'].set_visible(True)
    ax_hist.spines['bottom'].set_visible(True)
    ax_hist.spines['left'].set_visible(True)

    # 设置刻度
    ax_hist.xaxis.set_major_locator(ticker.MultipleLocator(64))
    ax_hist.xaxis.set_minor_locator(ticker.MultipleLocator(16))
    ax_hist.yaxis.set_major_locator(ticker.AutoLocator())

    # 添加轴标签
    ax_hist.set_xlabel('Pixel Value')
    ax_hist.set_ylabel('Frequency')


def visualize_rgb_channels(img_rgb):
    """
    可视化RGB图像的原始图像和三个通道的灰度分布

    参数:
    img_rgb: RGB图像 (numpy数组, uint8格式, 形状为 H×W×3)

    返回:
    无返回值，直接显示图像
    """
    # 验证输入
    if not isinstance(img_rgb, np.ndarray):
        raise ValueError("输入必须是numpy数组")
    if img_rgb.dtype != np.uint8:
        raise ValueError("图像必须是uint8格式")
    if len(img_rgb.shape) != 3 or img_rgb.shape[2] != 3:
        raise ValueError("输入必须是H×W×3的RGB图像")

    # 分离通道
    r_channel = img_rgb[:, :, 0]
    g_channel = img_rgb[:, :, 1]
    b_channel = img_rgb[:, :, 2]

    # 创建自定义布局
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 0.01])

    # 原始RGB图像
    ax_original = fig.add_subplot(gs[0, 0])
    ax_original.imshow(img_rgb)
    ax_original.axis('off')
    ax_original.set_title('Original RGB Image', fontsize=10)
    ax_r = fig.add_subplot(gs[0, 1])
    gray_image_1 = np.mean(img_rgb, axis=2).astype(np.uint8)
    ax_r.imshow(gray_image_1, cmap='gray')
    ax_r.axis('off')
    ax_r.set_title('Red Channel', fontsize=10)
    ax_hist_r = fig.add_subplot(gs[0, 2])
    plot_hist(ax_hist_r, gray_image_1, color='red', channel_name='Red')

    ax_g = fig.add_subplot(gs[1, 1])
    gray_image_2 = 0.2 * img_rgb[:, :, 0] + 0.1 * img_rgb[:, :, 1] + 0.7 * img_rgb[:, :, 2]
    ax_g.imshow(gray_image_2, cmap='gray')
    ax_g.axis('off')
    ax_g.set_title('Green Channel', fontsize=10)
    ax_hist_g = fig.add_subplot(gs[1, 2])
    plot_hist(ax_hist_g, gray_image_2, color='green', channel_name='Green')

    ax_b = fig.add_subplot(gs[2, 1])
    gray_image_3 = np.max(img_rgb, axis=2)
    # gray_image_3 = np.min(img_rgb, axis=2)
    ax_b.imshow(gray_image_3, cmap='gray')
    ax_b.axis('off')
    ax_b.set_title('Blue Channel', fontsize=10)
    ax_hist_b = fig.add_subplot(gs[2, 2])
    plot_hist(ax_hist_b, gray_image_3, color='blue', channel_name='Blue')


    # 添加共享标签
    fig.text(0.5, 0.02, 'Pixel Value', ha='center', va='center', fontsize=12)
    fig.text(0.02, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', fontsize=12)

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.3, bottom=0.1, left=0.1)
    plt.show()


if __name__ == '__main__':
    # 示例1: 使用OpenCV读取图像
    import cv2

    # 读取RGB图像 (注意OpenCV默认BGR顺序)
    # img_bgr = cv2.imread(r'C:\Users\ZFH\Desktop\1-15\BaseExample\dark.png', cv2.IMREAD_COLOR)
    # img_bgr = cv2.imread(r'C:\Users\ZFH\Desktop\1-15\BaseExample\shiny.png', cv2.IMREAD_COLOR)
    img_bgr = cv2.imread(r'C:\Users\ZFH\Desktop\1-15\BaseExample\block.png', cv2.IMREAD_COLOR)
    print(img_bgr.shape)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 调用接口
    visualize_rgb_channels(img_rgb)