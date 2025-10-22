import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
import os
import warnings

from src_plot.plot_heatmap import plot_clustering_heatmap

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)

def set_ultimate_chinese_font():
    """
    终极中文解决方案：确保所有中文正常显示
    优先使用系统字体，失败则回退到默认字体
    """
    try:
        # Windows 系统中的常见中文字体文件路径
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
            'C:/Windows/Fonts/simkai.ttf',  # 楷体
        ]

        # Linux/MacOS 中的常见中文字体路径
        if os.name != 'nt':
            font_paths = [
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                '/System/Library/Fonts/PingFang.ttc',
                '/Library/Fonts/Arial Unicode.ttf'
            ]

        # 查找第一个可用的字体文件
        font_file = None
        for path in font_paths:
            if os.path.exists(path):
                font_file = path
                print(f"✅ 找到字体文件: {path}")
                break

        if font_file:
            # 添加字体属性
            font_prop = fm.FontProperties(fname=font_file)
            font_name = font_prop.get_name()

            # 核心设置
            plt.rcParams['font.family'] = [font_name, 'sans-serif']
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial Unicode MS']

            # 用于检查的示例文本
            test_text = "测试中文字体: 窗长 均 类"
            fig, ax = plt.subplots(figsize=(2, 1))
            ax.text(0.5, 0.5, test_text, ha='center', va='center', fontproperties=font_prop)
            # plt.savefig('chinese_test.png', dpi=100)
            # print(f"✅ 中文测试图已保存，请检查 'chinese_test.png'")
            plt.close(fig)

        # 设置通用的中文相关参数
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 返回用于特定元素的字体属性
        return font_prop if font_file else None

    except Exception as e:
        print(f"⚠ 设置字体时出错: {str(e)}")
        # 安全后备方案
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
        return None



