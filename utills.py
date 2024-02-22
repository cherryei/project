import numpy as np
import pandas as pd
import math
import os
import tkinter.filedialog
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pywt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
my_font = FontProperties(fname=r"c:\windows\fonts\SimHei.ttf", size=12)

def bar_bottom(data,labels,tit,path):
    '''
    柱状堆积图
    :param data: 数据（二维）data[0]一层堆积数据，data[1]二层堆积数据
    :param labels: 堆积图横坐标（长度与数据长度一致）
    :param tit:堆积图名称
    :param path:堆积图保存地址
    :return:
    '''
    plt.style.use('seaborn')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    width = 0.35  # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots(figsize=(5, 3), dpi=200)
    ax.bar(labels, data[0], width, label='T',color="g")
    ax.bar(labels, data[1], width, color="r",bottom=data[0], label='F')
    for i in range(len(data[0])):
        ax.text(i, data[0][i]*0.5, str(data[0][i]), ha='center',fontsize=8)
        ax.text(i, data[0][i] + data[1][i] * 0.5, str(data[1][i]), ha='center',fontsize=8)
    ax.set_ylabel('Scores')
    ax.set_title(tit+'堆积柱状图')
    ax.legend()
    plt.savefig(path+"/"+tit+"堆积.jpg")
    plt.close()

def desample_data(data, out_nums):
    """
    降采样原始数据 - 所有数据在desample中调用 - out_nums需要输出的点数
    """
    result = np.zeros(out_nums)
    step = np.floor(len(data) // out_nums)
    for i in range(out_nums):
        result[i] = data[int(step * (i - 1))]
    return result

def pinghua(data,num_test):
    '''
    平滑曲线
    :param data: 需平滑的数据
    :param num_test: 平滑窗大小
    :return:
    '''
    n = int(len(data)/num_test)
    mean_data_all = np.zeros_like(data)
    mean_data_all[:num_test] = data[0:num_test]
    for i in range(len(data) - num_test):
        a = i
        b = i + num_test
        mean_data = np.mean(data[a:b])
        mean_data_all[i+num_test]=mean_data
    return mean_data_all

def Wavelet_transform(signal,fs):
    '''
    连续小波变换
    :param signal: 需要进行小波变换的原始信号
    :param fs: 采集频率
    :return: 小波变换后系数绝对值矩阵
    '''
    # 采样频率
    sampling_rate = 3000
    # 尺度长度
    totalscal = fs
    # 小波基函数
    wavename = 'morl'
    # 小波函数中心频率
    fc = pywt.central_frequency(wavename)
    # 常数c
    cparam = 2 * fc * totalscal
    # 尺度序列
    scales = cparam / np.arange(totalscal, 0, -1)
    # print(scales)
    # 进行CWT连续小波变换
    coefficients, frequencies = pywt.cwt(signal, scales, wavename, 1.0 / 1000)
    # 小波系数矩阵绝对值
    amp = abs(coefficients)
    return amp

def envelope_features(data,m):
    '''
    包络线特征值
    :param data: 振动数据
    :param m: 间隔点数
    :return:
    '''
    n = int(len(data) / m)
    max_all = []
    whe = [0]
    max_all.append(data[0])
    for i in range(n):
        a = max(data[m * i:m * (i + 1)])
        data_ = [0] * len(data)
        data_[m * i:m * (i + 1)] = data[m * i:m * (i + 1)]
        wh = data_.index(a)
        whe.append(wh)
        max_all.append(a)
    max_all.append(data[-1])
    whe.append(len(data) - 1)

def box_pictuer(data):
    '''
    箱型图合理范围内正常值
    :param data: 原始数据
    :return: 筛除异常值后的数据
    '''
    data_ = []
    percentile = np.percentile(data, (25, 50, 75))
    Q1 = percentile[0]  # 上四分位数
    Q2 = percentile[1]
    Q3 = percentile[2]  # 下四分位数
    IQR = Q3 - Q1  # 四分位距
    ulim = Q3 + 1.5 * IQR  # 上限 非异常范围内的最大值
    llim = Q1 - 1.5 * IQR  # 下限 非异常范围内的最小值
    for i in range(len(data)):
        if llim<data[i]<ulim:
            data_.append(data[i])
    return data_

def plt_confusion_matrix(data,classes,tit,path):
    plt.style.use('seaborn-white')
    confusion_matrix = np.array(data)
    len_ = len(data)
    proportion = []
    for i in confusion_matrix:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)
    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(len_,len_)  # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(len_, len_)
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
    }
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    plt.title(tit+'confusion_matrix',fontproperties=my_font)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    iters = np.reshape([[[i, j] for j in range(len_)] for i in range(len_)], (confusion_matrix.size, 2))
    for i, j in iters:
        if (i == j):
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=12, color='r',
                     weight=5)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12, color='r')
        else:
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=12)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    # plt.show()
    plt.savefig(path + "/" + tit + "混淆矩阵.jpg")
    plt.close()

def gaussian_kernel_vectorization(x1, x2, l=1.0, sigma_f=1.0):
    '''
    高斯分布的核函数（协方差矩阵）
    :param x1:数组1
    :param x2:数组2
    :param l:超参
    :param sigma_f:超参
    :return:数组1和数组2的协方差矩阵
    '''
    # """More efficient approach."""
    dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)

def file_neam(path):
    '''
    提取文件夹中所有的文件名
    :param path: 文件夹地址
    :return: 文件名列表
    '''
    file_all = []
    for iroot, idirs, ifiles in os.walk(path):
        if not idirs:
            file_all.extend(ifiles)
    return file_all

def get_derivative(data, num_test,delta1):
    '''
    求导函数
    :param f: 求导数组
    :param num_test: 前面去掉点数
    :param delta: 间隔
    :return: 求导后数组
    '''
    derivative_ = [0]*num_test

    for i in range(delta1,len(data)-delta1):
        derivative1= (data[i+delta1]-data[i])/delta1
        derivative_.append(derivative1)
    return derivative_