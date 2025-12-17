import math
import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from src_ele.dir_operation import search_files_by_criteria
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic
from src_plot.plot_logging import visualize_well_logs

np.set_printoptions(precision=6, suppress=True)


# 用来进行图像修复的dataloader，使用模型进行修复时，使用此dataloader
class dataloader_fmi_seg(Dataset):
    """
        电成像数据加载器
        用于加载多个电成像数据集
    """
    def __init__(self, path_dyna=r'F:\DeepLData\FMI_SIMULATION\simu_FMI', windows_length=400, windows_step=10, out_shape=256, normalize_params=True, normalization_method='minmax'):
        """
        初始化数据加载器
        参数:
        path_folder: 数据集文件夹路径
        len_windows: 窗口长度
        step_windows: 窗口步长
        out_shape: 输出图像尺寸
        """
        super().__init__()
        self.path_dyna = path_dyna
        self.path_stat = path_dyna.replace('_fmi_dyna.png', '_fmi_stat.png')            # 静态图像路径
        self.path_mask_cracks = path_dyna.replace('_fmi_dyna.png', '_cracks_mask.png')  # 裂缝掩码路径
        self.path_mask_holes = path_dyna.replace('_fmi_dyna.png', '_holes_mask.png')    # 孔洞掩码路径
        self.path_paras = path_dyna.replace('_fmi_dyna.png', '_background_para_processed.csv')  # 参数文件路径

        # 加载图像和数据
        self.fmi_dyna = self.load_image(self.path_dyna)     # 加载动态电成像图像
        self.fmi_stat = self.load_image(self.path_stat)     # 加载静态电成像图像
        self.fmi_mask_cracks = self.load_image(self.path_mask_cracks)       # 加载裂缝掩码图像
        self.fmi_mask_holes = self.load_image(self.path_mask_holes)         # 加载孔洞掩码图像
        self.fmi_paras = pd.read_csv(self.path_paras)               # 加载孔洞缝参数CSV文件
        self.data_depth = self.fmi_paras['depth']

        self.read_windows_shape = (self.fmi_dyna.shape[1], windows_length)
        self.out_shape = (out_shape, out_shape)     # 输出图像形状（正方形）

        # 验证所有图像具有相同的高度 （确保数据一致性）
        heights = [img.shape[0] for img in [self.fmi_dyna, self.fmi_stat, self.fmi_mask_cracks, self.fmi_mask_holes]]
        if len(set(heights)) > 1:
            raise ValueError("所有图像必须具有相同的高度")

        # 获取参数列名并设置窗口参数
        self.cols_paras = self.fmi_paras.columns.tolist()
        self.windows_length = windows_length    # 滑动窗口长度
        self.windows_step = windows_step        # 滑动窗口步长
        self.total_rows = self.fmi_dyna.shape[0]    # 图像总行数（高度）
        # 计算窗口数量：总行数减去窗口长度，除以步长，再加1
        self.windows_num = math.ceil((self.total_rows-windows_length)/windows_step) + 1
        self.target_paras = ['crack_length', 'crack_width', 'crack_area', 'crack_density', 'hole_area', 'hole_density', 'hole_area_ratio']
        self.normalize_params = normalize_params
        self.normalization_method = normalization_method

        # 计算全局参数统计信息
        self.global_param_means = None
        self.global_param_stds = None
        self.global_param_mins = None
        self.global_param_maxs = None
        self.intialize_global_param_stats()

        print(f"数据集加载完成，总窗口数: {self.windows_num}")
        print(f"全局参数均值: {self.global_param_means}")
        print(f"全局参数标准差: {self.global_param_stds}")

    def __getitem__(self, index):
        """
            获取指定索引的数据
            参数:
            index: 全局索引
            返回:
            数据字典
        """
        if index < 0 or index > self.windows_num:
            raise IndexError(f"索引超出范围: {index} (有效范围: 0-{self.windows_num - 1})")

        start_index = index * self.windows_step
        end_index = min(start_index + self.windows_length, self.fmi_dyna.shape[0])
        start_index = end_index - self.windows_length

        # 分别调整每个图像到目标尺寸
        dyna_resized = cv2.resize(self.fmi_dyna[start_index:end_index], self.out_shape)/256
        stat_resized = cv2.resize(self.fmi_stat[start_index:end_index], self.out_shape)/256
        mask_cracks_resized = cv2.resize(self.fmi_mask_cracks[start_index:end_index], self.out_shape)/256
        mask_holes_resized = cv2.resize(self.fmi_mask_holes[start_index:end_index], self.out_shape)/256

        # 修改点1：将动态图像和静态图像合并为2通道图像
        # 形状从 (H, W) 变为 (2, H, W)，符合PyTorch的通道优先格式
        fmi_image = np.stack([dyna_resized, stat_resized], axis=0)

        # 修改点2：将裂缝掩码和孔洞掩码合并为2通道掩码
        # 形状从 (H, W) 变为 (2, H, W)
        fmi_mask = np.stack([mask_cracks_resized, mask_holes_resized], axis=0)

        # 修改点3：提取参数数据
        params = self.fmi_paras.loc[(start_index + end_index) // 2, self.target_paras].values

        # 对参数进行归一化（如果需要）
        if self.normalize_params:
            params = self.normalize_parameters(params)

        # 修改点4：重新组织返回的字典结构
        data_dict = {
            'fmi_image': fmi_image.astype(np.float32),  # 2通道FMI图像 [2, H, W]
            'fmi_mask': fmi_mask.astype(np.float32),  # 2通道掩码图像 [2, H, W]
            'fmi_params': params.astype(np.float32),  # 孔洞缝参数 [7,]
        }
        return data_dict

    def load_image(self, path):
        """加载图像并确保为灰度图"""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法加载图像: {path}")
        return img

    def __len__(self):
        """返回总窗口数"""
        return self.windows_num

    def get_dataset_info(self):
        """获取数据集信息"""
        info = {
            'total_windows': self.windows_num,                          # 总窗口数
            'window_length': self.windows_length,                       # 窗口长度
            'window_step': self.windows_step,                           # 窗口步长
            'output_windows_length': self.out_shape,                    # 输出图像尺寸
            'normalize_params': self.normalize_params,                  # 是否归一化参数
            'normalization_method': self.normalization_method,          # 归一化方法
            'global_param_means': self.global_param_means.tolist(),
            'global_param_stds': self.global_param_stds.tolist(),
            'global_param_min': self.global_param_mins.tolist(),
            'global_param_max': self.global_param_maxs.tolist()
        }
        return info

    def intialize_global_param_stats(self):
        """
        根据数据集特征，自己设置孔洞缝参数，这个主要是用来进行后续的孔洞缝参数预测结果恢复的
        """

        # 计算全局统计量
        # self.global_param_means = np.array([0.6221, 0.0904, 0.0815, 0.5685, 0.0546, 11.1021, 0.000754])
        # self.global_param_stds = np.array([0.5950, 0.0752, 0.1270, 0.6751, 0.0253, 4.2401, 0.03280])
        # self.global_param_mins = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # self.global_param_maxs = np.array([3.6356, 0.5822, 2.0849, 3.0, 0.1931, 32.0, 0.2637509])
        self.global_param_means = np.array([0.647133, 0.09096, 0.102039, 0.848195, 0.039036, 7.733515, 0.002664])
        self.global_param_stds = np.array([0.627959, 0.094842, 0.160714, 0.781702, 0.027156, 4.829096, 0.001792])
        self.global_param_mins = np.array([0, 0, 0, 0, 0, 0, 0])
        self.global_param_maxs = np.array([3.893162, 0.632009, 2.202867, 3, 0.186791, 32, 0.012173])

    def normalize_parameters(self, params):
        """ 使用全局统计信息归一化参数 """
        if not self.normalize_params or self.global_param_means is None:
            return params

        if self.normalization_method == 'zscore':
            # Z-score归一化: (x - mean) / std
            return (params - self.global_param_means) / self.global_param_stds
        elif self.normalization_method.lower() == 'minmax':
            # Min-Max归一化: (x - min) / (max - min)
            ranges = self.global_param_maxs - self.global_param_mins
            ranges = np.where(ranges < 1e-6, 1.0, ranges)  # 防止除零
            return (params - self.global_param_mins) / ranges
        else:
            return params

    def denormalize_parameters(self, normalized_params):
        """反归一化参数"""
        if not self.normalize_params or self.global_param_means is None:
            return normalized_params

        if self.normalization_method == 'zscore':
            # Z-score反归一化: x * std + mean
            return normalized_params * self.global_param_stds + self.global_param_means
        elif self.normalization_method == 'minmax':
            # Min-Max反归一化: x * (max - min) + min
            ranges = self.global_param_maxs - self.global_param_mins
            ranges = np.where(ranges < 1e-8, 1.0, ranges)  # 防止除零
            return normalized_params * ranges + self.global_param_mins
        else:
            return normalized_params

    def combine_model_data(self, model_seg_result, model_paras_pred):
        depth_list_paras = []
        mask_cracks = np.zeros_like(self.fmi_dyna, dtype=np.float32)
        mask_holes = np.zeros_like(self.fmi_dyna, dtype=np.float32)
        mask_weight = np.zeros_like(self.fmi_dyna, dtype=np.float32)

        for i in range(self.windows_num):
            start_index = i * self.windows_step
            end_index = min(start_index + self.windows_length, self.fmi_dyna.shape[0])
            start_index = end_index - self.windows_length

            middle_index = (start_index + end_index) // 2
            depth_list_paras.append(self.data_depth.iloc[middle_index])

            mask_crack = cv2.resize(model_seg_result[i, 0, :, :], self.read_windows_shape)
            mask_hole = cv2.resize(model_seg_result[i, 1, :, :], self.read_windows_shape)

            drop_length = 100
            if (i == 0) or (i==self.windows_num-1):
                mask_cracks[start_index:end_index] += mask_crack
                mask_holes[start_index:end_index] += mask_hole
                mask_weight[start_index:end_index] += 1.0
            else:
                mask_cracks[start_index+drop_length:end_index-drop_length] += mask_crack[drop_length:-drop_length, :]
                mask_holes[start_index+drop_length:end_index-drop_length] += mask_hole[drop_length:-drop_length, :]
                mask_weight[start_index+drop_length:end_index-drop_length] += 1.0

        mask_cracks = mask_cracks/mask_weight
        mask_holes = mask_holes/mask_weight
        _, mask_cracks = cv2.threshold(mask_cracks*255, 64, 255, cv2.THRESH_BINARY)
        _, mask_holes = cv2.threshold(mask_holes*255, 64, 255, cv2.THRESH_BINARY)

        model_paras_pred = np.hstack((np.array(depth_list_paras).reshape((-1, 1)), model_paras_pred))
        model_paras_pred = pd.DataFrame(model_paras_pred, columns=['depth']+self.target_paras)

        # visualize_well_logs(
        #     data=self.fmi_paras,
        #     depth_col='depth',
        #     curve_cols=self.target_paras,
        #     figsize=(8, 10),
        # )
        #
        # visualize_well_logs(
        #     data=model_paras_pred,
        #     depth_col='depth',
        #     curve_cols=self.target_paras,
        #     figsize=(8, 10),
        # )

        # 反归一化
        model_paras_pred[self.target_paras] = self.denormalize_parameters(model_paras_pred[self.target_paras].values)
        # 将 裂缝密度、孔洞密度 进行整数化
        columns_to_convert = ['crack_density', 'hole_density']
        # 对 指定列 进行 四舍五入 并转换为整数
        for col in columns_to_convert:
            model_paras_pred[col] = model_paras_pred[col].round().astype(int)

        # show_Pic(pic_list=[255-self.fmi_dyna, 255-self.fmi_stat, self.fmi_mask_cracks, self.fmi_mask_holes, mask_cracks, mask_holes], pic_order=[1, 6], figure=(6, 15))

        # print(model_paras_pred.describe())
        # visualize_well_logs(
        #     data=model_paras_pred,
        #     depth_col='depth',
        #     curve_cols=self.target_paras,
        #     figsize=(8, 10),
        # )
        return mask_cracks, mask_holes, model_paras_pred, self.fmi_dyna, self.fmi_stat, self.fmi_mask_cracks, self.fmi_mask_holes





# 用来进行图像修复的dataloader，使用模型进行修复时，使用此dataloader
class dataloader_fmi_real_seg(Dataset):
    """
        电成像数据加载器
        用于加载多个电成像数据集
    """
    def __init__(self, path_dyna=r'F:\logging_workspace\樊页3HF\樊页3HF_DYNA_ORIGIN_target_result.txt', windows_length=400, windows_step=10, out_shape=256, normalize_params=True, normalization_method='minmax'):
        """
        初始化数据加载器
        参数:
        path_folder: 数据集文件夹路径
        len_windows: 窗口长度
        step_windows: 窗口步长
        out_shape: 输出图像尺寸
        """
        super().__init__()
        self.path_dyna = path_dyna
        self.path_stat = path_dyna.replace('DYNA', 'STAT')                                  # 静态图像路径

        # 加载图像和数据
        self.fmi_dyna, depth_dyna = get_ele_data_from_path(self.path_dyna)     # 加载动态电成像图像
        self.fmi_stat, depth_stat = get_ele_data_from_path(self.path_stat)     # 加载静态电成像图像
        self.data_depth = pd.DataFrame({
            'depth': depth_dyna.ravel(),
        })

        self.read_windows_shape = (self.fmi_dyna.shape[1], windows_length)
        self.out_shape = (out_shape, out_shape)     # 输出图像形状（正方形）

        # 验证所有图像具有相同的高度 （确保数据一致性）
        heights = [img.shape[0] for img in [self.fmi_dyna, self.fmi_stat]]
        if len(set(heights)) > 1:
            raise ValueError("所有图像必须具有相同的高度")

        # 获取参数列名并设置窗口参数
        self.windows_length = windows_length    # 滑动窗口长度
        self.windows_step = windows_step        # 滑动窗口步长
        self.total_rows = self.fmi_dyna.shape[0]    # 图像总行数（高度）
        # 计算窗口数量：总行数减去窗口长度，除以步长，再加1
        self.windows_num = math.ceil((self.total_rows-windows_length)/windows_step) + 1
        self.target_paras = ['crack_length', 'crack_width', 'crack_area', 'crack_density', 'hole_area', 'hole_density', 'hole_area_ratio']
        self.normalize_params = normalize_params
        self.normalization_method = normalization_method

        # 计算全局参数统计信息
        self.global_param_means = None
        self.global_param_stds = None
        self.global_param_mins = None
        self.global_param_maxs = None
        self.intialize_global_param_stats()

        print(f"数据集加载完成，总窗口数: {self.windows_num}")
        print(f"全局参数均值: {self.global_param_means}")
        print(f"全局参数标准差: {self.global_param_stds}")

    def __getitem__(self, index):
        """
            获取指定索引的数据
            参数:
            index: 全局索引
            返回:
            数据字典
        """
        if index < 0 or index > self.windows_num:
            raise IndexError(f"索引超出范围: {index} (有效范围: 0-{self.windows_num - 1})")

        start_index = index * self.windows_step
        end_index = min(start_index + self.windows_length, self.fmi_dyna.shape[0])
        start_index = end_index - self.windows_length

        # 分别调整每个图像到目标尺寸
        dyna_resized = cv2.resize(self.fmi_dyna[start_index:end_index], self.out_shape)/256
        stat_resized = cv2.resize(self.fmi_stat[start_index:end_index], self.out_shape)/256

        # 修改点1：将动态图像和静态图像合并为2通道图像
        # 形状从 (H, W) 变为 (2, H, W)，符合PyTorch的通道优先格式
        fmi_image = np.stack([dyna_resized, stat_resized], axis=0)

        # 修改点4：重新组织返回的字典结构
        data_dict = {
            'fmi_image': fmi_image.astype(np.float32),  # 2通道FMI图像 [2, H, W]
        }
        return data_dict

    def __len__(self):
        """返回总窗口数"""
        return self.windows_num

    def get_dataset_info(self):
        """获取数据集信息"""
        info = {
            'total_windows': self.windows_num,                          # 总窗口数
            'window_length': self.windows_length,                       # 窗口长度
            'window_step': self.windows_step,                           # 窗口步长
            'output_windows_length': self.out_shape,                    # 输出图像尺寸
            'normalize_params': self.normalize_params,                  # 是否归一化参数
            'normalization_method': self.normalization_method,          # 归一化方法
            'global_param_means': self.global_param_means.tolist(),
            'global_param_stds': self.global_param_stds.tolist(),
            'global_param_min': self.global_param_mins.tolist(),
            'global_param_max': self.global_param_maxs.tolist()
        }
        return info

    def intialize_global_param_stats(self):
        """
        根据数据集特征，自己设置孔洞缝参数，这个主要是用来进行后续的孔洞缝参数预测结果恢复的
        """

        # 计算全局统计量
        self.global_param_means = np.array([0.647133, 0.09096, 0.102039, 0.848195, 0.039036, 7.733515, 0.002664])
        self.global_param_stds = np.array([0.627959, 0.094842, 0.160714, 0.781702, 0.027156, 4.829096, 0.001792])
        self.global_param_mins = np.array([0, 0, 0, 0, 0, 0, 0])
        self.global_param_maxs = np.array([3.893162, 0.632009, 2.202867, 3, 0.186791, 32, 0.012173])

    def denormalize_parameters(self, normalized_params):
        """反归一化参数"""
        if not self.normalize_params or self.global_param_means is None:
            return normalized_params

        if self.normalization_method == 'zscore':
            # Z-score反归一化: x * std + mean
            return normalized_params * self.global_param_stds + self.global_param_means
        elif self.normalization_method == 'minmax':
            # Min-Max反归一化: x * (max - min) + min
            ranges = self.global_param_maxs - self.global_param_mins
            ranges = np.where(ranges < 1e-8, 1.0, ranges)  # 防止除零
            return normalized_params * ranges + self.global_param_mins
        else:
            return normalized_params

    def combine_model_data(self, model_seg_result, model_paras_pred):
        depth_list_paras = []
        mask_cracks = np.zeros_like(self.fmi_dyna, dtype=np.float32)
        mask_holes = np.zeros_like(self.fmi_dyna, dtype=np.float32)
        mask_weight = np.zeros_like(self.fmi_dyna, dtype=np.float32)

        for i in range(self.windows_num):
            start_index = i * self.windows_step
            end_index = min(start_index + self.windows_length, self.fmi_dyna.shape[0])
            start_index = end_index - self.windows_length

            middle_index = (start_index + end_index) // 2
            depth_list_paras.append(self.data_depth.iloc[middle_index])

            mask_crack = cv2.resize(model_seg_result[i, 0, :, :], self.read_windows_shape)
            mask_hole = cv2.resize(model_seg_result[i, 1, :, :], self.read_windows_shape)

            drop_length = 100
            if (i == 0) or (i==self.windows_num-1):
                mask_cracks[start_index:end_index] += mask_crack
                mask_holes[start_index:end_index] += mask_hole
                mask_weight[start_index:end_index] += 1.0
            else:
                mask_cracks[start_index+drop_length:end_index-drop_length] += mask_crack[drop_length:-drop_length, :]
                mask_holes[start_index+drop_length:end_index-drop_length] += mask_hole[drop_length:-drop_length, :]
                mask_weight[start_index+drop_length:end_index-drop_length] += 1.0

        mask_cracks = mask_cracks/mask_weight
        mask_holes = mask_holes/mask_weight
        _, mask_cracks = cv2.threshold(mask_cracks*255, 64, 255, cv2.THRESH_BINARY)
        _, mask_holes = cv2.threshold(mask_holes*255, 64, 255, cv2.THRESH_BINARY)

        model_paras_pred = np.hstack((np.array(depth_list_paras).reshape((-1, 1)), model_paras_pred))
        model_paras_pred = pd.DataFrame(model_paras_pred, columns=['depth']+self.target_paras)

        # visualize_well_logs(
        #     data=self.fmi_paras,
        #     depth_col='depth',
        #     curve_cols=self.target_paras,
        #     figsize=(8, 10),
        # )
        #
        # visualize_well_logs(
        #     data=model_paras_pred,
        #     depth_col='depth',
        #     curve_cols=self.target_paras,
        #     figsize=(8, 10),
        # )

        # 反归一化
        model_paras_pred[self.target_paras] = self.denormalize_parameters(model_paras_pred[self.target_paras].values)
        # 将 裂缝密度、孔洞密度 进行整数化
        columns_to_convert = ['crack_density', 'hole_density']
        # 对 指定列 进行 四舍五入 并转换为整数
        for col in columns_to_convert:
            model_paras_pred[col] = model_paras_pred[col].round().astype(int)

        # show_Pic(pic_list=[255-self.fmi_dyna, 255-self.fmi_stat, self.fmi_mask_cracks, self.fmi_mask_holes, mask_cracks, mask_holes], pic_order=[1, 6], figure=(6, 15))

        # print(model_paras_pred.describe())
        # visualize_well_logs(
        #     data=model_paras_pred,
        #     depth_col='depth',
        #     curve_cols=self.target_paras,
        #     figsize=(8, 10),
        # )
        return mask_cracks, mask_holes, model_paras_pred, self.fmi_dyna, self.fmi_stat

if __name__ == '__main__':

    # 初始化数据加载器
    data_loader = dataloader_fmi_seg(
        path_dyna=r'F:\DeepLData\FMI_SIMULATION\simu_FMI\3_fmi_dyna.png',
        windows_length=400,
        windows_step=100
    )
    print(f"总窗口数: {len(data_loader)}")

    # 获取数据集信息
    info = data_loader.get_dataset_info()
    print("\n数据集信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 测试获取不同索引的数据
    test_indices = [0, len(data_loader)//2, len(data_loader) - 1]

    for idx in test_indices:
        print(f"\n获取索引 {idx} 的数据...")
        data = data_loader[idx]

        # 修改点5：更新测试代码以适应新的数据结构
        print(f"  FMI图像形状: {data['fmi_image'].shape}")  # 应该是 (2, 256, 256)
        print(f"  FMI掩码形状: {data['fmi_mask'].shape}")  # 应该是 (2, 256, 256)
        print(f"  参数数据: {type(data['fmi_params'])} - 值: {data['fmi_params']}")

        # 修改点6：分别显示两个通道的图像
        # 提取动态图像和静态图像
        dyna_image = data['fmi_image'][0]  # 第一个通道：动态图像
        stat_image = data['fmi_image'][1]  # 第二个通道：静态图像
        cracks_mask = data['fmi_mask'][0]  # 第一个通道：裂缝掩码
        holes_mask = data['fmi_mask'][1]  # 第二个通道：孔洞掩码

        print(f"  动态图像范围: [{dyna_image.min()}, {dyna_image.max()}]")
        print(f"  静态图像范围: [{stat_image.min()}, {stat_image.max()}]")
        print(f"  裂缝掩码范围: [{cracks_mask.min()}, {cracks_mask.max()}]")
        print(f"  孔洞掩码范围: [{holes_mask.min()}, {holes_mask.max()}]")

        # 显示图像 （需要确保show_Pic函数能处理单通道图像）
        show_Pic([dyna_image, stat_image, cracks_mask, holes_mask], pic_str=['动态图像', '静态图像', '裂缝掩码', '孔洞掩码'])



