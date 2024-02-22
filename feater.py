# -*- coding: utf-8 -*-
"""
@Time    : 2021/11/18 0:33
@Author  : ONER
@FileName: plt_cm.py
@SoftWare: PyCharm
"""

# confusion_matrix
import random
import os
import tkinter.filedialog
from matplotlib import rcParams
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from matplotlib.font_manager import FontProperties
from Fast_Fourier_Transform import *
import pywt
import seaborn as sns
# sns.set(style="darkgrid") #这是seaborn默认的风格

import shutil
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
my_font = FontProperties(fname=r"c:\windows\fonts\SimHei.ttf", size=12)





def Max_peak_valley_value(data):  # 最大峰谷值计算5
    changdu = math.ceil(len(data) / 11)
    changdu_1 = int(len(data) / 11)
    num_0 = len(data)-changdu_1*11
    num_1 = 0
    data = [abs(i) for i in data]
    data_max = []
    for i in range(11):
        if num_1 <= num_0:
            data_1 = max(data[(i * changdu_1+num_1):(i + 1) * changdu_1+num_1+1])
            num_1 += 1
            data_max.append(data_1)
        else:
            data_1 = max(data[i * changdu_1:(i + 1) * changdu_1+1])
            data_max.append(data_1)
    percentile = np.percentile(data_max, (25, 50, 75))
    Q1 = percentile[0]  # 上四分位数
    Q2 = percentile[1]
    Q3 = percentile[2]  # 下四分位数
    IQR = Q3 - Q1  # 四分位距
    ulim = Q3 + 1.5 * IQR  # 上限 非异常范围内的最大值
    llim = Q1 - 1.5 * IQR  # 下限 非异常范围内的最小值
    right_list = []
    Error_Point_num = 0
    value_total = 0
    average_num = 0
    for item in data_max:
        if item < llim or item > ulim:
            Error_Point_num += 1
        else:
            right_list.append(item)
            value_total += item
            average_num += 1
    max_value = max(right_list)
    return max_value

def root_mean_square(data):  # 有效值计算6
    data_rms = math.sqrt(sum([x ** 2 for x in data]) / len(data))

    return data_rms

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

def Crest_factor(data):
    '''
    波峰因数
    :param data:
    :return:
    '''
    max_p=Max_peak_valley_value(data)
    rms_ = root_mean_square(data)
    C = max_p/rms_
    return C

def Skew_features(data):
    '''
    偏斜度
    :param data:
    :return:
    '''
    S = 0
    len_ = len(data)
    mean_ = np.mean(data)  # 1.均值
    for i in range(len_):
        S += (data[i] - mean_) ** 3
    std_ = np.std(data)
    S = S / ((len_ - 1) * std_ ** 3)
    return abs(S)

def qiaodu(data):   # 峭度8''
    '''
    峭度
    :param data:
    :return:
    '''
    a = np.mean(data)
    root_mean = math.sqrt(sum([(x-a) ** 2 for x in data]) / len(data))
    K = (sum([((x-a)/root_mean)**4 for x in data]))/len(data)
    return K

def yuduyinzi(data):       # 裕度因子9
    '''
    裕度因子
    :param data:
    :return:
    '''
    a = Max_peak_valley_value(data)
    yudu = a / pow(abs((sum(np.sqrt([abs(x) for x in data])) / len(data))), 2)
    return yudu

def sa(data):
    '''
    与峰度非常相似，但总体性能比峰度更好
    :param data:
    :return:
    '''
    a = np.mean(data)
    gm_1 = sum([abs(x-a) ** 3 for x in data]) / len(data)
    gm_2 = (sum([abs(x-a) for x in data])/len(data))**3
    GM3 = gm_1/gm_2
    return GM3

def Ikur(data):
    '''
    标准化第六中心力矩
    :param data:
    :return:
    '''
    a = np.mean(data)
    I0 = np.sqrt(sum((x-a)**2 for x in data)/len(data))
    Ikur6 = (sum((x-a)**6 for x in data)/len(data))/I0**6
    return Ikur6

def Isf(data):
    '''
    波形因数
    :param data:
    :return:
    '''
    a = root_mean_square(data)
    b = np.mean(data)
    isf = a/b
    return abs(isf)

def Iif(data):
    '''
    脉冲因子
    :param data:
    :return:
    '''
    d_max = Max_peak_valley_value(data)
    iif = d_max/np.mean(data)
    return abs(iif)

def Iclf(data):
    '''
    余隙系数
    :param data:
    :return:
    '''
    d_max = Max_peak_valley_value(data)
    iclf = d_max/(np.mean(data)**2)
    return abs(iclf)

def box_pictuer(data):
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

def plt_box(data,classes,tit,path):
    max_len = max(len(sublist) for sublist in data)
    padded_lst = [sublist + [np.nan] * (max_len - len(sublist)) for sublist in data]
    arr = np.array(padded_lst)
    transposed_arr = np.transpose(arr)
    # print(transposed_arr)
    tips = pd.DataFrame(list(transposed_arr), columns=classes)
    # tips = pd.DataFrame(data).T
    # print(tips)
    ax = sns.boxplot(data=tips, showfliers=False)
    plt.title(tit)
    plt.savefig(path + "/" + tit + "箱型图.jpg")
    # plt.show()

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

def bar_bottom(data,labels,tit,path):
    plt.style.use('seaborn')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # my_font = FontProperties(fname=r"c:\windows\fonts\SimHei.ttf", size=12)

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
    # ax.text(.87, -.08, '\nVisualization by DataCharm', transform=ax.transAxes,
    #         ha='center', va='center', fontsize=5, color='black', fontweight='bold', family='Roboto Mono')
    # plt.savefig(r'F:\DataCharm\SCI paper plots\sci_bar_guanwang', width=5, height=3,
    #             dpi=900, bbox_inches='tight')
    # ax.bar(labels, mu_number, width, label=labels[0], ec='k', lw=.6)
    # ax.bar(labels, ma_number, width, bottom=mu_number, label=labels[1], ec='k', lw=.6)
    # ax.bar(labels, en_number, width, bottom=[sum(x) for x in zip(mu_number, ma_number)], label=labels[2], ec='k', lw=.6)
    # ax.bar(labels, ch_number, width, bottom=[sum(x) for x in zip(mu_number, ma_number, en_number)], label=labels[3],
    #        ec='k', lw=.6)
    plt.savefig(path+"/"+tit+"堆积.jpg")
    plt.close()

def gaussian_kernel_vectorization(x1, x2, l=1.0, sigma_f=1.0):
    """More efficient approach."""
    dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    # return sigma_f ** 2 * np.exp(-1.2 / l ** 2 * dist_matrix)
    return sigma_f ** 2 * np.exp(-dist_matrix / 1.2*l ** 2)

def Wavelet_transform(signal,fs):
    # # 采样频率
    # sampling_rate = 3000
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
    return amp,frequencies,scales

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

def desample_data(data, out_nums):
    """
    降采样原始数据 - 所有数据在desample中调用 - out_nums需要输出的点数
    """
    result = np.zeros(out_nums)
    step = np.floor(len(data) // out_nums)
    for i in range(out_nums):
        result[i] = data[int(step * (i - 1))]
    return result


def perk(data,fs,num_test,out_nums,ow,path):
    amp,frequencies,scales = Wavelet_transform(data,fs)

    mean_arr = amp.mean(axis=0)

    mean_arr_ = pinghua(mean_arr, num_test)

    out_nums = int(len(data) / out_nums)
    mean_arr_ = desample_data(mean_arr_, out_nums)

    k1 = gaussian_kernel_vectorization(mean_arr_.reshape(-1, 1), mean_arr_.reshape(-1, 1), l=1.0, sigma_f=1.0)
    k2 = gaussian_kernel_vectorization(mean_arr.reshape(-1, 1), mean_arr_.reshape(-1, 1), l=1.0, sigma_f=1.0)
    identity_matrix = np.eye(len(mean_arr_))
    inverse_k = np.linalg.inv(k1+ow*identity_matrix)
    mds = np.dot(np.dot(k2,inverse_k),mean_arr_)

    er = abs(mean_arr-mds)

    return er


# time = np.linspace(0, 1, 3000, endpoint=False)  # 时间
# data = np.sin(2 * np.pi * 30 * time)
# def prodict_perk(path):
#     file_name = []
#     n = 0
#     normal = np.load('normal.npy')
#     fault = np.load('fault.npy')
#     ratio = fault/normal
#     for iroot,idirs,ifiles in os.walk(path):
#         if not idirs:
#             file_name.extend(ifiles)
#     for f_nam in file_name[:-2]:
#         print(f_nam)

def prodict_perk(path):
    # print(path)
    data = pd.read_csv(path,encoding='gbk')
    data = data.iloc[:,1]
    k = 8000
    data = data[5 * k:int(5.1 * k + 1)].values.tolist()
    er = perk(data,int(k/80),10,1,0.5,path)
    kuet_per = qiaodu(er)
    return kuet_per

# def train_perk(path,neamtit):
#     file_name = []
#     kuet_per_all = []
#     for iroot, idirs, ifiles in os.walk(path):
#         if not idirs:
#             file_name.extend(ifiles)
#     for f_nam in file_name[:-2]:
#         print(f_nam)
#         data = pd.read_csv(path + '/' + f_nam, encoding='gbk')
#         data = data.iloc[:, 1]
#         k = 8000
#         data = data[5 * k:int(6.5 * k + 1)].values.tolist()
#         # data = data[:6000]
#         er = perk(data, 3000, 10, 10, 2)
#         kuet_per = qiaodu(er)
#         kuet_per_all.append(kuet_per)
#     np.save(neamtit+'.npy', np.mean(kuet_per_all))


def xingchil():
    path_normal = r"C:\luojin\预维新特征\残余能量特征交叉验证\正常"
    patn_fault = r"C:\luojin\预维新特征\残余能量特征交叉验证\报警"
    file_name_normal = []
    file_name_fault = []
    kuet_per_normal = []
    kuet_per_fault = []
    for iroot, idirs, ifiles in os.walk(path_normal):
        if not idirs:
            file_name_normal.extend(ifiles)
    for iroot, idirs, ifiles in os.walk(patn_fault):
        if not idirs:
            file_name_fault.extend(ifiles)

    if "desktop.ini" in file_name_normal:
        file_name_normal.remove("desktop.ini")
    if "desktop.ini" in file_name_fault:
        file_name_fault.remove("desktop.ini")
    for i in file_name_normal:
        print(i)
        kuet_per_n = prodict_perk(path_normal+"/"+i)
        kuet_per_normal.append(kuet_per_n)
    for j in file_name_fault:
        kuet_per_f = prodict_perk(patn_fault + "/" + j)
        kuet_per_fault.append(kuet_per_f)
    classes = ['normal', 'fault']

    path = r"C:\luojin\预维新特征\残余能量特征交叉验证\pickter\箱型图"
    plt_box([kuet_per_normal,kuet_per_fault], classes, "残余能量峭度", path)
    max_len = max(len(sublist) for sublist in [kuet_per_normal,kuet_per_fault])
    padded_lst = [sublist + [np.nan] * (max_len - len(sublist)) for sublist in [kuet_per_normal,kuet_per_fault]]
    # data_box = {}
    # data_box['normal'] = kuet_per_normal
    # data_box['fault'] = kuet_per_fault
    A_1 = pd.DataFrame(padded_lst[0])
    A_2 = pd.DataFrame(padded_lst[1])
    A = pd.concat([A_1,A_2],axis=1)
    A.to_csv(r"C:\luojin\预维新特征\残余能量特征交叉验证\pickter\箱型图\全部验证数据.csv",header=classes)


# xingchil()














if __name__ == "__main__":

    # path_0 = tkinter.filedialog.askopenfilename(title='选择异常数据')
    # path_0 = [r"C:\luojin\大连大众项目数据\故障csv数据",r"C:\luojin\大连大众项目数据\正常csv数据"]
    path_tit = r"C:\luojin\预维新特征\第二阶段批量验证数据\pectier"
    data_path = r"C:\luojin\预维新特征\残余能量特征交叉验证\VWED数据转换"
    path_0 = ["漏报", "漏预", "正常报警预警", "正常数据", "误报"]
    file_all = [[] * n for n in range(len(path_0))]
    for nae in range(len(path_0)):
        for iroot, idirs, ifiles in os.walk(data_path + '/' + path_0[nae]):
            if not idirs:
                file_all[nae].extend(ifiles)
    # print(file_all)

    # tz_all = []
    k = 8000
    tz_gs = ['Crest_factor(data)', 'Skew_features(data)', 'qiaodu(data)', 'yuduyinzi(data)', 'sa(data)', 'Ikur(data)',
             'Isf(data)', 'Iif(data)', 'Iclf(data)']
    title = ['波峰因数', '偏斜度', '峭度', '裕度因子', '峰度变形', '标准化第六中心力矩', '波形因数', '脉冲因子',
             '余隙系数']
    path_ = r"C:\luojin\预维新特征\数据\验证数据\结.csv"
    try:
        os.remove(path_)
    except:
        pass
    file_all[3] = file_all[3][0:-1]
    for tz in range(len(tz_gs)):
        features_all = [[] * n for n in range(len(path_0))]
        for i in range(len(file_all)):
            for j in file_all[i]:
                # print(path_0[i]+j)
                path = data_path + '/' + path_0[i] + '/' + j
                doker = pd.read_csv(path, encoding='gbk')
                C_ = []
                for td in range(3):
                    data = doker.iloc[:, td+1]
                    data = data[5 * k:int(6.5 * k + 1)].values.tolist()
                    # data = abs(np.array(data))
                    C = eval(tz_gs[tz])
                    C_.append(C)
                if "正常数据" in path:
                    C = np.mean(C_[1:])
                else:
                    C = max(C_)

                features_all[i].append(C)
                # tz_all.append(features_all)
        A = pd.DataFrame(features_all)
        A.to_csv(path_, header=False, index=False, mode='a')
        # print(features_all)
        # max_len = max(len(sublist) for sublist in features_all)
        # padded_lst = [sublist + [np.nan] * (max_len - len(sublist)) for sublist in features_all]
        # arr = np.array(padded_lst)
        # transposed_arr = np.transpose(arr)
        # print(transposed_arr)
        # tips = pd.DataFrame(list(transposed_arr), columns=path_0)
        # print(tips)
        # ax = sns.boxplot(data=tips,showfliers=False)
        # plt.title(title[tz])
        # plt.savefig(path_tit + "/" + title[tz] + "箱型图.jpg")
        # plt.show()

        fea_num = np.zeros(len(path_0))
        fea_num_ = []
        # fea_2 = box_pictuer(features_all[-2])
        fea_2 = list(features_all[-2])+list(features_all[-1])
        # fea_2 = box_pictuer(fea_2)
        fea_2.sort()
        n = int(len(fea_2) * 0.8)
        print(fea_2[n], '====nnn')
        # print(len(features_all[-2]))
        for f_n in range(len(fea_num)):
            for num in range(len(features_all[f_n])):
                if features_all[f_n][num] > fea_2[n]:
                    fea_num[f_n] += 1

        # print(max(features_all[-2]))
        classes = ['normal', 'fault']
        data = [
            [(len(features_all[-2]) + len(features_all[-1]) - fea_num[-1] - fea_num[-2]), fea_num[-1] + fea_num[-2]],
            [(len(features_all[0]) + len(features_all[1]) + len(features_all[2]) - sum(fea_num[:3])), sum(fea_num[:3])]]
        plt_confusion_matrix(data, classes, title[tz], path_tit)
        data_1 = fea_num.copy()
        data_1[-1] = len(features_all[-1])
        # data_2 = [(len(features_all[0])-fea_num[0]),(len(features_all[1])-fea_num[1]),(len(features_all[2])-fea_num[2]),fea_num[-2],fea_num[-1]]
        data_3 = fea_num.copy()

        data_3[-1] = len(features_all[-1]) - fea_num[-1]
        data_3[-2] = len(features_all[-2]) - fea_num[-2]
        data_2 = [(len(features_all[0]) - data_3[0]), (len(features_all[1]) - data_3[1]),
                  (len(features_all[2]) - data_3[2]), fea_num[-2], fea_num[-1]]
        # labels = ['漏报', '漏预', '正常报警预警', '正常数据', '误报']
        bar_bottom([data_3, data_2], path_0, title[tz], path_tit)
        plt.close()


