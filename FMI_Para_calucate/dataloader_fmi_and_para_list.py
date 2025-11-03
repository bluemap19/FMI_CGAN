import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from src_ele.dir_operation import search_files_by_criteria
from src_ele.pic_opeeration import show_Pic, pic_list_add_random_stripe

np.set_printoptions(precision=6, suppress=True)
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
#设置科学计数法
pd.set_option('display.float_format', lambda x: '%.6f' % x)


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

    def get(self, index, out_shape, target_paras_cols):
        if index < 0 or index > self.windows_num:
            raise IndexError(f"索引超出范围: {index} (有效范围: 0-{self.windows_num - 1})")

        start_index = index * self.windows_step
        end_index = min(start_index + self.windows_length, self.fmi_dyna.shape[0])
        start_index = end_index - self.windows_length

        # 分别调整每个图像到目标尺寸
        dyna_resized = cv2.resize(self.fmi_dyna[start_index:end_index], out_shape)/256
        stat_resized = cv2.resize(self.fmi_stat[start_index:end_index], out_shape)/256
        mask_cracks_resized = cv2.resize(self.fmi_mask_cracks[start_index:end_index], out_shape)/256
        mask_holes_resized = cv2.resize(self.fmi_mask_holes[start_index:end_index], out_shape)/256

        # 噪声添加，对动静态图像添加随机的噪声数据
        ratio_t = np.random.random()*0.5
        type_t = np.random.choice(['gaussian', 'salt_pepper', 'poisson', 'speckle'])
        dyna_resized = self.pic_add_noise(dyna_resized, ratio=ratio_t, noise_type=type_t)
        stat_resized = self.pic_add_noise(dyna_resized, ratio=ratio_t, noise_type=type_t)

        # 随机的空白带添加
        ratio_striped = np.random.random()
        if ratio_striped < 0.4:
            masked_list, mask_list = pic_list_add_random_stripe([dyna_resized, stat_resized])
            dyna_resized, stat_resized = masked_list[0], masked_list[1]

        # 修改点1：将动态图像和静态图像合并为2通道图像
        # 形状从 (H, W) 变为 (2, H, W)，符合PyTorch的通道优先格式
        fmi_image = np.stack([dyna_resized, stat_resized], axis=0)

        # 修改点2：将裂缝掩码和孔洞掩码合并为2通道掩码
        # 形状从 (H, W) 变为 (2, H, W)
        fmi_mask = np.stack([mask_cracks_resized, mask_holes_resized], axis=0)

        # 修改点3：提取参数数据
        params = self.fmi_paras.loc[(start_index + end_index) // 2, target_paras_cols].values

        # 修改点4：重新组织返回的字典结构
        data_dict = {
            'fmi_image': fmi_image.astype(np.float32),  # 2通道FMI图像 [2, H, W]
            'fmi_mask': fmi_mask.astype(np.float32),  # 2通道掩码图像 [2, H, W]
            'fmi_params': params.astype(np.float32),  # 孔洞缝参数 [7,]
        }
        return data_dict

    def pic_add_noise(self, pic, ratio=0.2, noise_type='gaussian'):
        """
        为图像添加随机噪声
        参数:
        pic: 输入图像 (可以是单通道或多通道)
        ratio: 噪声幅度比例 (0-1之间)
        noise_type: 噪声类型 ('gaussian', 'salt_pepper', 'poisson', 'speckle', 'mixed')
        返回:
        添加噪声后的图像
        """
        # 确保图像是浮点类型 (0-1范围)
        if np.max(pic) > 200:
            pic_float = pic.astype(np.float32) / 255.0
        else:
            pic_float = pic.copy()

        # 根据噪声类型添加噪声
        if noise_type == 'gaussian':
            noisy_pic = self._add_gaussian_noise(pic_float, ratio)
        elif noise_type == 'salt_pepper':
            noisy_pic = self._add_salt_pepper_noise(pic_float, ratio, salt_vs_pepper=0.4+0.2*np.random.random())
        elif noise_type == 'poisson':
            noisy_pic = self._add_poisson_noise(pic_float, ratio)
        elif noise_type == 'speckle':
            noisy_pic = self._add_speckle_noise(pic_float, ratio)
        else:
            raise ValueError(f"不支持的噪声类型: {noise_type}")

        # 确保值在有效范围内
        noisy_pic = np.clip(noisy_pic, 0, 1)

        # 转换为原始数据类型
        if np.max(pic) > 200:
            noisy_pic = (noisy_pic*255).astype(pic.dtype)
        else:
            noisy_pic = noisy_pic.astype(pic.dtype)

        return noisy_pic

    def _add_gaussian_noise(self, image, ratio):
        """
        添加高斯噪声

        参数:
        image: 输入图像 (0-1范围)
        ratio: 噪声幅度比例
        mean: 噪声均值 (默认0)

        返回:
        添加高斯噪声后的图像
        """
        # 计算标准差 (基于图像的标准差和ratio)
        std = ratio * np.std(image)

        # 生成高斯噪声
        noise = np.random.normal(0, std, image.shape)

        # 添加噪声
        noisy_image = image + noise

        return noisy_image

    def _add_salt_pepper_noise(self, image, ratio, salt_vs_pepper=0.5):
        """
        添加椒盐噪声
        参数:
        image: 输入图像 (0-1范围)
        ratio: 噪声比例 (被噪声污染的像素比例)
        返回:
        添加椒盐噪声后的图像
        """
        noisy_image = image.copy()

        # 计算盐噪声和椒噪声的数量
        amount = ratio
        salt_amount = amount * salt_vs_pepper
        pepper_amount = amount * (1 - salt_vs_pepper)

        # 添加盐噪声 (白色点)
        salt_mask = np.random.random(image.shape) < salt_amount
        noisy_image[salt_mask] = 1

        # 添加椒噪声 (黑色点)
        pepper_mask = np.random.random(image.shape) < pepper_amount
        noisy_image[pepper_mask] = 0

        return noisy_image

    def _add_poisson_noise(self, image, ratio):
        """
        添加泊松噪声 (适用于低光照条件下的噪声模拟)

        参数:
        image: 输入图像 (0-1范围)
        ratio: 噪声强度比例

        返回:
        添加泊松噪声后的图像
        """
        # 将图像缩放到适合泊松噪声的强度范围
        scaled_image = image * ratio * 255

        # 生成泊松噪声
        noisy_image = np.random.poisson(scaled_image) / 255.0

        # 归一化到0-1范围
        noisy_image = np.clip(noisy_image, 0, 1)

        return noisy_image

    def _add_speckle_noise(self, image, ratio):
        """
        添加乘性噪声 (散斑噪声)，适用于雷达和超声图像
        参数:
        image: 输入图像 (0-1范围)
        ratio: 噪声强度比例
        返回:
        添加散斑噪声后的图像
        """
        # 生成均匀分布的噪声
        noise = np.random.randn(*image.shape) * ratio

        # 添加乘性噪声
        noisy_image = image + image * noise

        return noisy_image



# 用来进行图像修复的dataloader，使用模型进行修复时，使用此dataloader
class dataloader_FMI_logging(Dataset):
    """
        电成像数据加载器
        用于加载多个电成像数据集
    """
    def __init__(self, path_folder=r'F:\DeepLData\FMI_SIMULATION\simu_FMI', len_windows=200, step_windows=10, out_shape=256, normalize_params=True, normalization_method='zscore'):
        """
        初始化数据加载器
        参数:
        path_folder: 数据集文件夹路径
        len_windows: 窗口长度
        step_windows: 窗口步长
        out_shape: 输出图像尺寸
        """
        super().__init__()
        self.path_folder = path_folder
        self.len_windows = len_windows
        self.step_windows = step_windows
        self.out_shape = (out_shape, out_shape)     # 输出图像形状（正方形）
        self.normalize_params = normalize_params
        self.normalization_method = normalization_method
        self.paras_target_cols = ['crack_length', 'crack_width', 'crack_area', 'crack_density', 'hole_area', 'hole_density', 'hole_area_ratio']

        # 搜索包含'dyna'关键字的PNG文件
        self.list_path_dyna = search_files_by_criteria(search_root=path_folder, name_keywords=['fmi_dyna'], file_extensions=['.png'])
        print(f"找到 {len(self.list_path_dyna)} 个电成像数据集")

        # 初始化数据集列表
        self.datasets = []              # 存储所有FMIDataset实例
        self.dataset_lengths = []       # 存储每个数据集的窗口数量
        # 加载所有数据集
        for path in self.list_path_dyna:
            try:
                # 为每个动态图像路径创建 FMIDataset 实例
                dataset = FMIDataset(
                    path_dyna=path,
                    windows_length=len_windows,
                    windows_step=step_windows
                )
                self.datasets.append(dataset)
                self.dataset_lengths.append(len(dataset))       # 记录每个数据集的窗口数
            except Exception as e:
                print(f"加载数据集 {path} 失败: {str(e)}")

        # 计算累积长度：用于快速定位数据集 ， 例如：[10, 20, 15] -> [10, 30, 45]
        self.cumulative_lengths = np.cumsum(self.dataset_lengths)
        self.total_length = sum(self.dataset_lengths)

        # 计算全局参数统计信息
        self.global_param_means = None
        self.global_param_stds = None
        self.global_param_mins = None
        self.global_param_maxs = None
        self._calculate_global_param_stats()

        print(f"数据集加载完成，总窗口数: {self.total_length}")
        print(f"各数据集长度: {self.dataset_lengths}")
        print(f"累积长度: {self.cumulative_lengths}")
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
        # 索引范围检查
        if index < 0 or index >= self.total_length:
            raise IndexError(f"索引超出范围: {index} (有效范围: 0-{self.total_length - 1})")

        # 使用二分查找确定索引属于哪个数据集
        # np.searchsorted返回插入位置，side='right'确保正确匹配
        dataset_idx = np.searchsorted(self.cumulative_lengths, index, side='right')

        # 计算在数据集内的局部索引
        if dataset_idx == 0:
            local_idx = index                                                   # 第一个数据集，索引直接使用
        else:
            local_idx = index - self.cumulative_lengths[dataset_idx - 1]        # 减去前面数据集的累积长度

        # 获取数据集
        dataset = self.datasets[dataset_idx]
        data_model_train = dataset.get(local_idx, self.out_shape, self.paras_target_cols)

        # 对参数进行归一化（如果需要）
        if self.normalize_params:
            data_model_train['original_params'] = data_model_train['fmi_params'].copy()  # 保存原始参数（未归一化）
            normalized_params = self.normalize_parameters(data_model_train['fmi_params'])
            data_model_train['fmi_params'] = normalized_params.astype(np.float32)

        # 调用数据集的get方法获取具体数据
        return data_model_train

    def __len__(self):
        """返回总窗口数"""
        return self.total_length

    def get_dataset_info(self):
        """获取数据集信息"""
        info = {
            'total_datasets': len(self.datasets),  # 数据集总数
            'total_windows': self.total_length,  # 总窗口数
            'window_length': self.len_windows,  # 窗口长度
            'window_step': self.step_windows,  # 窗口步长
            'dataset_lengths': self.dataset_lengths,  # 每个数据集的窗口数
            'cumulative_lengths': self.cumulative_lengths.tolist(),  # 累积长度列表
            'dataset_paths': [os.path.basename(path) for path in self.list_path_dyna],  # 数据集文件名
            'output_windows_length': self.out_shape,  # 输出图像尺寸
            'normalize_params': self.normalize_params,  # 是否归一化参数
            'normalization_method': self.normalization_method,  # 归一化方法
            'global_param_means': self.global_param_means,
            'global_param_stds': self.global_param_stds,
            'global_param_min': self.global_param_mins,
            'global_param_max': self.global_param_maxs,
        }
        return info

    def _calculate_global_param_stats(self):
        """计算所有数据集的全局参数统计信息"""
        all_params = []
        total_samples = 0

        # 收集所有数据集的参数
        for dataset in self.datasets:
            # 获取当前数据集的参数数据
            params_data = dataset.fmi_paras[self.paras_target_cols].values
            all_params.append(params_data)
            total_samples += len(params_data)

        # 合并所有参数
        all_params = np.vstack(all_params)
        print('all paras describe:{}'.format(pd.DataFrame(all_params, columns=self.paras_target_cols).describe()))

        # 计算全局统计量
        self.global_param_means = all_params.mean(axis=0)
        self.global_param_stds = all_params.std(axis=0)
        self.global_param_mins = all_params.min(axis=0)
        self.global_param_maxs = all_params.max(axis=0)

        # 防止标准差为0
        self.global_param_stds = np.where(self.global_param_stds < 1e-6, 1.0, self.global_param_stds)

        """
describe:	crack_length	crack_width	crack_area	crack_density	hole_area	hole_density	hole_area_ratio
count		2000000
mean	    0.622114	0.090478	0.081583	0.568526	0.054652	11.102179	0.000754
std	        0.595028	0.075209	0.127097	0.675148	0.025313	4.240160	0.000328
min	        0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0
0.250000	0.053829	0.031885	0.003321	0.000000	0.035980	8.000000	0.000513
0.500000	0.522625	0.085071	0.040329	0.000000	0.052383	11.000000	0.000728
0.750000	0.962085	0.131655	0.109317	1.000000	0.070878	14.000000	0.000968
max	        3.635651	0.582167	2.08494	    3	        0.193132	32	        0.002638
        """

    def normalize_parameters(self, params):
        """使用全局统计信息归一化参数"""
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
            ranges = np.where(ranges < 1e-6, 1.0, ranges)  # 防止除零
            return normalized_params * ranges + self.global_param_mins
        else:
            return normalized_params


if __name__ == '__main__':

    # 初始化数据加载器
    data_loader = dataloader_FMI_logging(
        path_folder=r'F:\DeepLData\FMI_SIMULATION\simu_FMI_2',
        # path_folder=r'F:\DeepLData\FMI_SIMULATION\simu_FMI',
        len_windows=400,
        step_windows=50,
        normalize_params=True,
        normalization_method='minmax',
    )

    print(f"总窗口数: {len(data_loader)}")

    # 获取数据集信息
    info = data_loader.get_dataset_info()
    print("\n数据集信息:")
    for key, value in info.items():
        print(f"  {key}\t:\t{value}")

    # 测试获取不同索引的数据
    test_indices = [0, 10, 20, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

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

        # 显示图像（需要确保show_Pic函数能处理单通道图像）
        show_Pic([dyna_image, stat_image, cracks_mask, holes_mask],
                 pic_str=['动态图像', '静态图像', '裂缝掩码', '孔洞掩码'])



