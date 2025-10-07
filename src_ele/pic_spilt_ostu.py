import math
import numpy as np
import cv2
from src_ele.file_operation import get_ele_data_from_path

# 单OTSU阈值分割
class OTSU_Segmentation():
    def __init__(self, pic):
        self.source_img = pic
        self.u1 = 0.0
        self.u2 = 0.0
        self.th = 0.0

    def CalTh(self, GrayScale):
        img_gray = np.zeros([])
        if len(self.source_img.shape) == 3:
            img_gray = cv2.cvtColor(self.source_img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = self.source_img
        img_gray = np.array(img_gray).ravel().astype(np.uint8)
        PixSum = img_gray.size
        PixCount = np.zeros(GrayScale)
        PixRate = np.zeros(GrayScale)
        # 统计各个灰度值的像素个数
        for i in range(PixSum):
            Pixvalue = img_gray[i]
            PixCount[Pixvalue] = PixCount[Pixvalue] + 1
        # 确定各个灰度值对应的像素点的个数在所有的像素点中的比例。
        for j in range(GrayScale):
            PixRate[j] = PixCount[j] * 1.0 / PixSum
        Max_var = 0
        # 确定最大类间方差对应的阈值
        for i in range(1, GrayScale):  # 从1开始是为了避免w1为0.
            u1_tem = 0.0
            u2_tem = 0.0
            # 背景像素的比列
            w1 = np.sum(PixRate[:i])
            # 前景像素的比例
            w2 = 1.0 - w1
            if w1 == 0 or w2 == 0:
                pass
            else:  # 背景像素的平均灰度值
                for m in range(i):
                    u1_tem = u1_tem + PixRate[m] * m
                self.u1 = u1_tem * 1.0 / w1
                # 前景像素的平均灰度值
                for n in range(i, GrayScale):
                    u2_tem = u2_tem + PixRate[n] * n
                self.u2 = u2_tem / w2
                # print(self.u1)
                # 类间方差公式：G=w1*w2*(u1-u2)**2
                tem_var = w1 * w2 * np.power((self.u1 - self.u2), 2)
                # print(tem_var)
                # 判断当前类间方差是否为最大值。
                if Max_var < tem_var:
                    Max_var = tem_var
                    self.th = i
        # print(self.th)

    def Otsu_translation(self):
        img_g = np.zeros([])
        if len(self.source_img.shape) == 3:
            img_g = cv2.cvtColor(self.source_img, cv2.COLOR_BGR2GRAY)
        else:
            img_g = self.source_img

        result, img_seg = cv2.threshold(img_g, self.th, 255, cv2.THRESH_BINARY)
        # print(result)
        # cv2.imwrite('test_file/1_seg.jpg', img_seg)
        # cv2.imshow('The result of OTSU segtaton', img_seg)
        # cv2.waitKey(0)






# 基于直方图金字塔的otsu多阈值分割
class _OtsuPyramid(object):

    def load_image(self, im, bins=256):
        '''
        读取图片并该图片的生成金字塔灰度直方图、和对应的omega、mu、和计算比例
        '''
        self.im = im
        hist, ranges = np.histogram(im, bins)
        # convert the numpy array to list of ints
        hist = [int(h) for h in hist]
        histPyr, omegaPyr, muPyr, ratioPyr = \
            self._create_histogram_and_stats_pyramids(hist)
        # reversed是为了反向排序，后续是从最小的金字塔开始向上放大搜索阈值的
        self.omegaPyramid = [omegas for omegas in reversed(omegaPyr)]
        self.muPyramid = [mus for mus in reversed(muPyr)]
        self.ratioPyramid = ratioPyr

    def _create_histogram_and_stats_pyramids(self, hist):
        """
        输入原始直方图（0-255），生成金字塔灰度直方图、和对应的omega、mu、和计算比例
        """
        bins = len(hist)
        ratio = 2
        reductions = int(math.log(bins, ratio))
        compressionFactor = []
        histPyramid = []
        omegaPyramid = []
        muPyramid = []
        for _ in range(reductions):
            histPyramid.append(hist)
            reducedHist = [sum(hist[i:i + ratio]) for i in range(0, bins, ratio)]
            # 通过合并两个像素生成一个尺寸为原直方图一般的灰度直方图，数值为两数相加
            hist = reducedHist
            # 更新直方图尺寸
            bins = bins // ratio
            compressionFactor.append(ratio)
        # "compression"：[1, 2, 2, 2, 2, 2, 2, 2]
        compressionFactor[0] = 1
        # print('compressionFactor:', compressionFactor)
        for hist in histPyramid:
            omegas, mus, muT = \
                self._calculate_omegas_and_mus_from_histogram(hist)
            omegaPyramid.append(omegas)
            muPyramid.append(mus)
        return histPyramid, omegaPyramid, muPyramid, compressionFactor

    def _calculate_omegas_and_mus_from_histogram(self, hist):
        """ 计算omega和mu
        """
        probabilityLevels, meanLevels = \
            self._calculate_histogram_pixel_stats(hist)
        bins = len(probabilityLevels)
        # these numbers are critical towards calculations, so we make sure
        # they are float
        ptotal = float(0)
        # sum of probability levels up to k
        omegas = []
        for i in range(bins):
            ptotal += probabilityLevels[i]
            omegas.append(ptotal)
        mtotal = float(0)
        mus = []
        for i in range(bins):
            mtotal += meanLevels[i]
            mus.append(mtotal)
        # muT is the total mean levels.
        muT = float(mtotal)
        return omegas, mus, muT

    def _calculate_histogram_pixel_stats(self, hist):
        """
        计算像素比例
        """
        # bins = number of intensity levels
        bins = len(hist)
        # N = number of pixels in image. Make it float so that division by
        # N will be a float
        N = float(sum(hist))
        # percentage of pixels at each intensity level: i => P_i
        hist_probability = [hist[i] / N for i in range(bins)]
        # mean level of pixels at intensity level i   => i * P_i
        pixel_mean = [i * hist_probability[i] for i in range(bins)]
        return hist_probability, pixel_mean


class OtsuFastMultithreshold(_OtsuPyramid):
    """总体方法是通过从最小的金字塔直方图开始不断按比例系数向尺寸最大的直方图逼近的方法，通过检验其左右（5个像素的类间方差去修正该直方图的最佳阈值）
    """

    def calculate_k_thresholds(self, k):
        self.threshPyramid = []
        start = self._get_smallest_fitting_pyramid(k)
        self.bins = len(self.omegaPyramid[start])
        thresholds = self._get_first_guess_thresholds(k)
        deviate = self.bins // 2  # 间隔
        for i in range(start, len(self.omegaPyramid)):
            omegas = self.omegaPyramid[i]
            mus = self.muPyramid[i]
            hunter = _ThresholdHunter(omegas, mus, deviate)
            thresholds = hunter.find_best_thresholds_around_estimates(thresholds)
            self.threshPyramid.append(thresholds)

            scaling = self.ratioPyramid[i]  # 压缩率
            deviate = scaling

            thresholds = [t * scaling for t in thresholds]

        # 最后一个循环会多放大scaling倍阈值，所以return的时候要返回对应//scaling阈值。
        return [t // scaling for t in thresholds]

    def _get_smallest_fitting_pyramid(self, k):
        """
        在金字塔直方图中获取满足阈值数量的最小一层直方图
        """
        for i, pyramid in enumerate(self.omegaPyramid):
            if len(pyramid) >= k:
                return i

    def _get_first_guess_thresholds(self, k):
        """
        获取粗略的阈值
        """
        kHalf = k // 2
        midway = self.bins // 2
        firstGuesses = [midway - i for i in range(kHalf, 0, -1)] + [midway] + \
                       [midway + i for i in range(1, kHalf)]
        # additional threshold in case k is odd
        firstGuesses.append(self.bins - 1)
        return firstGuesses[:k]

    def apply_thresholds_to_image(self, thresholds, im=None):
        '''
        通过划分好的阈值对图片进行多阈值分割
        '''
        if im is None:
            im = self.im
        k = len(thresholds)
        bookendedThresholds = [None] + thresholds + [None]
        greyValues = [0] + [int(256 / k * (i + 1)) for i in range(0, k - 1)] \
                     + [255]
        greyValues = np.array(greyValues, dtype=np.uint8)
        finalImage = np.zeros(im.shape, dtype=np.uint8)
        for i in range(k + 1):
            kSmall = bookendedThresholds[i]
            bw = np.ones(im.shape, dtype=np.bool)
            if kSmall:
                bw = (im >= kSmall)
            kLarge = bookendedThresholds[i + 1]
            if kLarge:
                bw &= (im < kLarge)
            greyLevel = greyValues[i]
            greyImage = bw * greyLevel
            finalImage += greyImage
        return finalImage


class _ThresholdHunter(object):
    """
    对_get_first_guess_thresholds函数中获取的粗略阈值进行微调，使其结果更加精确，通过比较左右5个像素范围（如果不满足5个就按阈值间像素个数
    来算）最大类间方差来精确比较最佳阈值
    """

    def __init__(self, omegas, mus, deviate=2):
        self.sigmaB = _BetweenClassVariance(omegas, mus)
        self.bins = self.sigmaB.bins
        self.deviate = deviate

    def find_best_thresholds_around_estimates(self, estimatedThresholds):
        """
        求取精确阈值
        """
        bestResults = (
            0, estimatedThresholds, [0 for t in estimatedThresholds]
        )
        bestThresholds = estimatedThresholds
        bestVariance = 0
        for thresholds in self._jitter_thresholds_generator(
                estimatedThresholds, 0, self.bins):

            variance = self.sigmaB.get_total_variance(thresholds)
            if variance == bestVariance:
                if sum(thresholds) < sum(bestThresholds):
                    bestThresholds = thresholds
            elif variance > bestVariance:
                bestVariance = variance
                bestThresholds = thresholds
        return bestThresholds

    def _jitter_thresholds_generator(self, thresholds, min_, max_):
        '''
        生成器
        '''
        pastThresh = thresholds[0]
        if len(thresholds) == 1:
            # -2 through +2
            for offset in range(-self.deviate, self.deviate + 1):
                thresh = pastThresh + offset
                if thresh < min_ or thresh >= max_:
                    continue
                yield [thresh]
        else:
            thresholds = thresholds[1:]
            m = len(thresholds)
            for offset in range(-self.deviate, self.deviate + 1):
                thresh = pastThresh + offset
                if thresh < min_ or thresh + m >= max_:
                    continue
                recursiveGenerator = self._jitter_thresholds_generator(
                    thresholds, thresh + 1, max_
                )
                for otherThresholds in recursiveGenerator:
                    yield [thresh] + otherThresholds


class _BetweenClassVariance(object):
    '''
    计算类间方差
    '''

    def __init__(self, omegas, mus):
        self.omegas = omegas
        self.mus = mus
        self.bins = len(mus)
        self.muTotal = sum(mus)

    def get_total_variance(self, thresholds):
        thresholds = [0] + thresholds + [self.bins - 1]
        numClasses = len(thresholds) - 1
        sigma = 0
        for i in range(numClasses):
            k1 = thresholds[i]
            k2 = thresholds[i + 1]
            sigma += self._between_thresholds_variance(k1, k2)
        return sigma

    def _between_thresholds_variance(self, k1, k2):
        omega = self.omegas[k2] - self.omegas[k1]
        mu = self.mus[k2] - self.mus[k1]
        muT = self.muTotal
        return omega * ((mu - muT) ** 2)


# # if __name__ == '__main__':
#     # 基于直方图金字塔的otsu多阈值分割 接口的测试
#
#     filename = r'D:\1111\Input\test-pic.txt'
#     pic, depth = get_ele_data_from_path(filename, depth=[6498, 6499.5])
#     otsu = OTSU_Segmentation(pic)
#     # otsu.CalTh(256)
#     # otsu.Otsu_translation()
#
#
#     # filename = 'kouzhao.jpg'
#     # dot = filename.index('.')
#     # prefix, extension = filename[:dot], filename[dot:]
#     # im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#
#     cv2.imwrite('test_file/test1.png', pic)
#
#     otsu = OtsuFastMultithreshold()
#     otsu.load_image(pic)
#     for k in [1, 2, 3, 4, 5, 6]:
#         savename = 'test_save_{}.png'.format(k)
#         kThresholds = otsu.calculate_k_thresholds(k)
#         print(kThresholds)
#         crushed = otsu.apply_thresholds_to_image(kThresholds)
#         cv2.imwrite(savename, crushed)



