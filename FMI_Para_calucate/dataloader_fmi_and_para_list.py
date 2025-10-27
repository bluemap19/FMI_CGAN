import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from src_ele.dir_operation import search_files_by_criteria
from src_ele.pic_opeeration import show_Pic
np.set_printoptions(precision=6, suppress=True)


class FMIDataset():
    """
        电成像数据集类
        用于加载和处理单个FMI电成像数据集
    """
    def __init__(self, path_dyna=r'F:\DeepLData\FMI_SIMULATION\simu_FMI_2\4_fmi_dyna.png', windows_length=200, windows_step=100):
        """
            初始化电成像数据集
            参数:
            path_dyna: 动态图像路径
            windows_length: 窗口长度
            windows_step: 窗口步长
        """
        assert path_dyna.__contains__('dyna'), 'path_dyna must contain \'dyna\' element'
        self.path_dyna = path_dyna
        self.path_stat = path_dyna.replace('_fmi_dyna.png', '_fmi_stat.png')
        self.path_mask_cracks = path_dyna.replace('_fmi_dyna.png', '_cracks_mask.png')
        self.path_mask_holes = path_dyna.replace('_fmi_dyna.png', '_holes_mask.png')
        self.path_paras = path_dyna.replace('_fmi_dyna.png', '_background_para_processed.csv')

        # 加载图像和数据
        self.fmi_dyna = self.load_image(self.path_dyna)
        self.fmi_stat = self.load_image(self.path_stat)
        self.fmi_mask_cracks = self.load_image(self.path_mask_cracks)
        self.fmi_mask_holes = self.load_image(self.path_mask_holes)
        self.fmi_paras = pd.read_csv(self.path_paras)

        # 验证所有图像具有相同的高度
        heights = [img.shape[0] for img in [self.fmi_dyna, self.fmi_stat, self.fmi_mask_cracks, self.fmi_mask_holes]]
        if len(set(heights)) > 1:
            raise ValueError("所有图像必须具有相同的高度")

        self.cols_paras = self.fmi_paras.columns.tolist()
        self.windows_length = windows_length
        self.windows_step = windows_step
        self.total_rows = self.fmi_dyna.shape[0]

        # 计算窗口数量
        self.windows_num = (self.total_rows - windows_length) // windows_step + 1

        # print(f"数据集初始化: {os.path.basename(path_dyna)}")
        # print(f"  总行数: {self.total_rows}, 窗口数: {self.windows_num}")
        # print(f"  窗口长度: {windows_length}, 窗口步长: {windows_step}")

    def __len__(self):
        return self.windows_num

    def load_image(self, path):
        """加载图像并确保为灰度图"""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法加载图像: {path}")
        return img

    def get(self, index, ouu_shape):
        if index < 0 or index > self.windows_num:
            raise IndexError(f"索引超出范围: {index} (有效范围: 0-{self.windows_num - 1})")

        start_index = index * self.windows_step
        end_index = min(start_index + self.windows_length, self.fmi_dyna.shape[0])
        start_index = end_index - self.windows_length

        pic_dict = {
            'fmi_dyna': cv2.resize(self.fmi_dyna[start_index:end_index], ouu_shape),
            'fmi_stat': cv2.resize(self.fmi_stat[start_index:end_index], ouu_shape),
            'fmi_mask_cracks': cv2.resize(self.fmi_mask_cracks[start_index:end_index], ouu_shape),
            'fmi_mask_holes': cv2.resize(self.fmi_mask_holes[start_index:end_index], ouu_shape),
            'fmi_paras': self.fmi_paras.loc[(start_index + end_index) // 2, ['crack_length', 'crack_width', 'crack_area', 'crack_density', 'hole_area', 'hole_density', 'hole_area_ratio']].values,
            # 'fmi_paras': self.fmi_paras.loc[(start_index + end_index) // 2, ['crack_length', 'crack_width', 'crack_area', 'crack_density', 'hole_area', 'hole_density', 'hole_area_ratio']],
        }
        return pic_dict


# 用来进行图像修复的dataloader，使用模型进行修复时，使用此dataloader
class dataloader_FMI_logging(Dataset):
    """
        电成像数据加载器
        用于加载多个电成像数据集
    """
    def __init__(self, path_folder=r'F:\DeepLData\FMI_SIMULATION\simu_FMI', len_windows=200, step_windows=10, out_shape=256):
        """
        初始化数据加载器
        参数:
        path_folder: 数据集文件夹路径
        len_windows: 窗口长度
        step_windows: 窗口步长
        """
        super().__init__()
        self.path_folder = path_folder
        self.len_windows = len_windows
        self.step_windows = step_windows
        self.out_shape = (out_shape, out_shape)

        self.list_path_dyna = search_files_by_criteria(search_root=path_folder, name_keywords=['fmi_dyna'], file_extensions=['.png'])
        print(f"找到 {len(self.list_path_dyna)} 个电成像数据集")

        # 初始化数据集列表
        self.datasets = []
        self.dataset_lengths = []
        # 加载所有数据集
        for path in self.list_path_dyna:
            try:
                dataset = FMIDataset(
                    path_dyna=path,
                    windows_length=len_windows,
                    windows_step=step_windows
                )
                self.datasets.append(dataset)
                self.dataset_lengths.append(len(dataset))
            except Exception as e:
                print(f"加载数据集 {path} 失败: {str(e)}")

        # 计算累积长度
        self.cumulative_lengths = np.cumsum(self.dataset_lengths)
        self.total_length = sum(self.dataset_lengths)

        print(f"数据集加载完成，总窗口数: {self.total_length}")
        print(f"各数据集长度: {self.dataset_lengths}")
        print(f"累积长度: {self.cumulative_lengths}")

    def __getitem__(self, index):
        """
            获取指定索引的数据
            参数:
            index: 全局索引
            返回:
            数据字典
        """
        if index < 0 or index >= self.total_length:
            raise IndexError(f"索引超出范围: {index} (有效范围: 0-{self.total_length - 1})")

        # 使用 searchsorted 查找数据集索引
        dataset_idx = np.searchsorted(self.cumulative_lengths, index, side='right')

        # 计算在数据集内的局部索引
        if dataset_idx == 0:
            local_idx = index
        else:
            local_idx = index - self.cumulative_lengths[dataset_idx - 1]

        # 获取数据集
        dataset = self.datasets[dataset_idx]

        # 获取数据
        return dataset.get(local_idx, self.out_shape)

    def __len__(self):
        """返回总窗口数"""
        return self.total_length

    def get_dataset_info(self):
        """获取数据集信息"""
        info = {
            'total_datasets': len(self.datasets),
            'total_windows': self.total_length,
            'window_length': self.len_windows,
            'window_step': self.step_windows,
            'dataset_lengths': self.dataset_lengths,
            'cumulative_lengths': self.cumulative_lengths.tolist(),
            'dataset_paths': [os.path.basename(path) for path in self.list_path_dyna],
            'output_windows_length': self.out_shape,
        }
        return info


if __name__ == '__main__':

    # 测试数据加载器
    try:
        # 初始化数据加载器
        data_loader = dataloader_FMI_logging(
            path_folder=r'F:\DeepLData\FMI_SIMULATION\simu_FMI_2',
            len_windows=200,
            step_windows=10
        )

        print(f"总窗口数: {len(data_loader)}")

        # 获取数据集信息
        info = data_loader.get_dataset_info()
        print("\n数据集信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # 测试获取不同索引的数据
        test_indices = [0, 101, 500, 1000, len(data_loader) - 1]

        for idx in test_indices:
            try:
                print(f"\n获取索引 {idx} 的数据...")
                data = data_loader[idx]

                # 打印数据摘要
                print(f"  动态图像形状: {data['fmi_dyna'].shape}")
                print(f"  静态图像形状: {data['fmi_stat'].shape}")
                print(f"  裂缝掩模形状: {data['fmi_mask_cracks'].shape}")
                print(f"  孔洞掩模形状: {data['fmi_mask_holes'].shape}")
                print(f"  参数数据: {type(data['fmi_paras'])} -》 \n{data['fmi_paras']}")
                show_Pic([data['fmi_dyna'], data['fmi_stat'], data['fmi_mask_cracks'], data['fmi_mask_holes']])
            except Exception as e:
                print(f"获取索引 {idx} 失败: {str(e)}")

    except Exception as e:
        print(f"测试失败: {str(e)}")


