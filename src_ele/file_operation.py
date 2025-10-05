import os

import numpy as np
import xlwt
import cv2
# from PIL import Image

def add_data_to_sheet(booksheet, data):
    # DATA = (('学号', '姓名', '年龄', '性别', '成绩'),
    #         ('1001', 'A', '11', '男', '12'),
    #         ('1002', 'B', '12', '女8888', '88822'),
    #         ('1003', 'C', '13', '女', '32'),
    #         ('1004', 'D', '14', '男', '52'),)
    # for i, row in enumerate(DATA):
    #     for j, col in enumerate(row):
    #         booksheet.write(i, j, col)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            booksheet.write(i, j, data[i][j])


def save_file_as_xlsx(data, path='model/aaa.xlsx', sheet_name=['sheet1', 'sheet2']):
    # # 1.导入openpyxl模块
    # import openpyxl
    # # 2.调用Workbook()方法
    # wb = openpyxl.Workbook()
    # # 3. 新建一个excel文件，并且在单元表为"sheet1"的表中写入数据
    # ws = wb.create_sheet("sheet1")
    # # 4.在单元格中写入数据
    # # ws.cell(row=m, column=n).value = *** 在第m行n列写入***数据
    # ws.cell(row=1, column=1).value = "时间"
    # ws.cell(row=1, column=2).value = "零食"
    # ws.cell(row=1, column=3).value = "是否好吃"
    # # 5.保存表格
    # wb.save('嘿嘿.xlsx')
    # print('保存成功！')
    workbook = xlwt.Workbook(encoding='utf-8')
    for i in range(len(data)):
        sheet_name_t = 'a'
        if i < len(sheet_name):
            sheet_name_t = sheet_name[i]
        else:
            sheet_name_t = 'sheet{}'.format(i+1)
        booksheet = workbook.add_sheet(sheet_name_t, cell_overwrite_ok=True)
        add_data_to_sheet(booksheet, data[i])
    workbook.save(path)


def get_ele_data_from_path(strname = r'D:\Data\target\107S\YS107_FMI_BorEID_FA.txt', depth = [-1.0, -1.0]):
    # print(strname.__contains__('.png')|strname.__contains__('.jpg'))

    img_data = np.array([])
    depth_data = np.array([])
    if (strname.endswith('.png') | strname.endswith('.jpg')) :
        # print(strname)
        # print('Get Data from png')
        # img_data = cv2.imread(strname)             # 三通道？？？  (186810, 240 ,3)
        # img_data = np.array(Image.open(strname))     # 222222
        img_data = np.array(cv2.imread(strname, cv2.IMREAD_GRAYSCALE))     # 222222
        # print(img_data)
        if img_data.shape[0] == 0:
            print('image in data is empty:{}'.format(strname))
            exit(0)
        depth_data = np.zeros((img_data.shape[0], 1))
        if len(strname.split('_')) < 4:
            print(strname)
            print('file name error, donnot contain depth information')
            exit(0)
            return img_data, depth_data
        # print(strname.split('\\')[-1].split('_')[1], strname.split('_')[1], strname.split('\\')[-1])
        startdep = float(strname.split('\\')[-1].split('/')[-1].split('_')[2])
        enddep = float(strname.split('\\')[-1].split('/')[-1].split('_')[3])
        # print(startdep, enddep)
        Step = (enddep-startdep)/img_data.shape[0]
        for i in range(img_data.shape[0]):
            depth_data[i] = i*Step + startdep
    elif (strname.endswith('.txt')):
        AllData = np.loadtxt(strname, delimiter='\t', skiprows=8, encoding='GBK')
        if AllData.shape[0] ==0 :
            print('text in data is empty:{}'.format(strname))
            exit(0)
        img_data = AllData[:, 1:]
        depth_data = AllData[:, 0].reshape((AllData.shape[0], 1))
        # startdep = depth_data[0, 0]
        # enddep = depth_data[0, 0]
        # print(AllData.shape)
        # print('from Txt get ELE Data')
    elif (strname.endswith('.csv')):
        print('CsvData')

    # 从深度方向截取数据
    if (depth[0] < 0) & (depth[1] < 0):
        pass
    else:
        Step = (depth_data[-1] - depth_data[0])/depth_data.shape[0]
        start_index = 0
        end_index = 0

        if depth[0] <= 0:
            start_index = 0
        if depth[1] <= 0:
            end_index = img_data.shape[0]-1
        for i in range(depth_data.shape[0]):
            if abs((depth_data[i]-depth[0])) <= Step/2:
                start_index = i
            if abs((depth_data[i]-depth[1])) <= Step/2:
                end_index = i

        # print(step, depth_data[0], depth_data[-1], depth, start_index, end_index)
        return img_data[start_index:end_index, :], depth_data[start_index:end_index, :]

    return img_data, depth_data

# get_ele_data_from_path()


def get_test_ele_data():
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建数据目录路径
    data_dir = os.path.join(script_dir, "get_random_data")

    # 定义文件名
    # dyna_file = "lg7-4_163_5183.3027_5183.9277_dyna.png"
    # stat_file = "lg7-4_163_5183.3027_5183.9277_stat.png"
    dyna_file = 'guan17-11_344_3752.0025_3753.2525_dyna.png'
    stat_file = 'guan17-11_344_3752.0025_3753.2525_stat.png'

    # 构建完整路径
    path_test1 = os.path.join(data_dir, dyna_file)
    path_test2 = os.path.join(data_dir, stat_file)

    # 检查文件是否存在
    if not os.path.exists(path_test1):
        raise FileNotFoundError(f"文件不存在: {path_test1}")
    if not os.path.exists(path_test2):
        raise FileNotFoundError(f"文件不存在: {path_test2}")

    # 获取数据
    data_img_dyna, data_depth = get_ele_data_from_path(path_test1)
    data_img_stat, data_depth = get_ele_data_from_path(path_test2)

    return data_img_dyna, data_img_stat, data_depth

