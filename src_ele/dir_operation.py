import os
from pathlib import Path
from typing import Union, Iterable, List


# def check_and_make_dir(dir_path):
#     if os.path.exists(dir_path):
#         return
#         # print("{} already exists!!!".format(dir_path))
#         # exit()
#     else:
#         os.makedirs(dir_path)
#         print('successfully create dir:{}'.format(dir_path))
#         assert os.path.exists(dir_path), dir_path
def check_and_make_dir(dir_path):
    """
    :param dir_path: string -- folder path needed to check
    :return: NULL
    """
    if os.path.exists(dir_path):
        return
    else:
        os.makedirs(dir_path)
        print('successfully create dir:{}'.format(dir_path))
        assert os.path.exists(dir_path), dir_path




# def traverseFolder_folder(path):
#     path_folder = []
#     for path in os.walk(path):
#         for file_name in path[1]:
#             path_folder.append(path[0].replace('\\', '/')+'/' + file_name)
#
#     return path_folder
def get_all_subfolder_paths(root_dir: Union[str, Path]) -> List[str]:
    """获取目录及其所有子目录路径
    Args:
        root_dir: 需要遍历的根目录路径
    Returns:
        包含所有子目录绝对路径的列表，路径使用正斜杠格式
    Examples:
        >>> folders = get_all_subfolder_paths('D:/projects')
        >>> print(folders[:2])
        ['D:/projects', 'D:/projects/src']
    """
    root_path = Path(root_dir)
    folder_paths = []
    for current_dir, dirs, _ in os.walk(root_path):
        # 添加子目录
        for dir_name in dirs:
            folder_paths.append(str(Path(current_dir) / dir_name))

    return sorted(set(folder_paths))  # 去重并排序



# def traverseFolder(path):
#     FilePath = []
#     for path in os.walk(path):
#         for file_name in path[2]:
#             FilePath.append(path[0].replace('\\', '/')+'/' + file_name)
#
#     return FilePath
def get_all_file_paths(root_dir: Union[str, Path]) -> List[str]:
    """获取目录及其子目录下所有文件路径
    Args:
        root_dir: 需要遍历的根目录路径
    Returns:
        包含所有文件绝对路径的列表，路径使用正斜杠格式
    Examples:
        >>> paths = get_all_file_paths('D:/projects')
        >>> print(paths[:2])
        ['D:/projects/README.md', 'D:/projects/utils/__init__.py']
    """
    root_path = Path(root_dir)
    file_paths = []

    for current_dir, _, files in os.walk(root_path):
        for filename in files:
            # 统一转换为正斜杠路径
            full_path = Path(current_dir) / filename
            file_paths.append(str(full_path))

    return file_paths


def search_files_by_criteria(
        search_root: Union[str, Path],
        name_keywords: Iterable[str] = (),
        file_extensions: Iterable[str] = (),
        all_keywords: bool = True,
) -> List[str]:
    """根据名称关键字和文件扩展名搜索文件
    Args:
        search_root: 需要搜索的根目录
        name_keywords: 文件名需要包含的关键字序列
        file_extensions: 需要匹配的文件扩展名序列（如 .txt）
    Returns:
        匹配文件的绝对路径列表，按字母顺序排列
    Examples:
        >>> find_files = search_files_by_criteria(
        ...     'D:/data',
        ...     name_keywords=['log', '2023'],
        ...     file_extensions=['.csv', '.xlsx']
        ... )
    """
    all_files = get_all_file_paths(search_root)
    matched_files = []

    # 转换大小写不敏感的扩展名集合
    if len(file_extensions) > 0:
        ext_set = {ext.lower().strip('.') for ext in file_extensions}
    else:
        ext_set = []

    for file_path in all_files:
        path_obj = Path(file_path)

        # 排除临时文件（两种常见临时文件格式）
        if path_obj.name.startswith('~') or '~' in path_obj.name:
            continue

        if len(ext_set)>0:
            # 检查扩展名
            if path_obj.suffix.lstrip('.').lower() not in ext_set:
                continue

        # 检查名称关键字,首先判断是严格对照所有关键词，还是只对照其中一项关键词
        filename = path_obj.stem.lower()
        if all_keywords:
            contains_all_keywords = all(
                keyword.lower() in filename
                for keyword in name_keywords
            )
        else:
            contains_all_keywords = any(
                keyword.lower() in filename
                for keyword in name_keywords
            )

        if contains_all_keywords:
            matched_files.append(file_path)

    return sorted(matched_files)


if __name__ == '__main__':

    path_dir = r'F:\FMI_SIMULATION'
    path_list = search_files_by_criteria(search_root=path_dir, name_keywords=['0', 'dyna'], file_extensions=['png'])
    print(path_list)