# -*- coding: utf-8 -*-
"""
@Time    : 2021/11/18 0:33
@Author  : ONER
@FileName: plt_cm.py
@SoftWare: PyCharm
"""

# confusion_matrix
import os
import tkinter.filedialog
from matplotlib import rcParams
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from matplotlib.font_manager import FontProperties
from Fast_Fourier_Transform import *
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

# path_0 = tkinter.filedialog.askopenfilename(title='选择异常数据')
# path_0 = [r"C:\luojin\大连大众项目数据\故障csv数据",r"C:\luojin\大连大众项目数据\正常csv数据"]
path_tit = r"C:\luojin\预维新特征\第二阶段批量验证数据\pectier"
data_path = r"C:\Users\IGT\Desktop\VWED数据转换"
path_0 = ["漏报","漏预","正常报警预警","正常数据","误报"]
file_all = [[],[],[],[],[]]
for nae in range(len(path_0)):
    for iroot,idirs,ifiles in os.walk(data_path+'/'+path_0[nae]):
        if not idirs:
            file_all[nae].extend(ifiles)
# print(file_all)

tz_all = []
k = 8000
tz_gs = ['Crest_factor(data)','Skew_features(data)','qiaodu(data)','yuduyinzi(data)','sa(data)','Ikur(data)','Isf(data)','Iif(data)','Iclf(data)']
title = ['波峰因数','偏斜度','峭度','裕度因子','峰度变形','标准化第六中心力矩','波形因数','脉冲因子','余隙系数']
path_ = r"C:\luojin\预维新特征\数据\验证数据\结.csv"
try:
    os.remove(path_)
except:
    pass
file_all[3] = file_all[3][0:-1]
for tz in range(len(tz_gs)):
    features_all = [[], [], [], [],[]]
    for i in range(len(file_all)):
        for j in file_all[i]:
            # print(path_0[i]+j)
            path = data_path+'/'+path_0[i]+ '/' + j
            data = pd.read_csv(path,encoding='gbk')
            data = data.iloc[:,1]
            data = data[3*k:8*k+1].values.tolist()
            C = eval(tz_gs[tz])
            features_all[i].append(C)
            tz_all.append(features_all)
    A = pd.DataFrame(features_all)
    A.to_csv(path_,header=False,index=False,mode='a')

    fea_num = [0,0,0,0,0]
    fea_num_ = []
    # fea_2 = box_pictuer(features_all[-2])
    fea_2 = (features_all[-2])
    fea_2 = box_pictuer(fea_2)
    fea_2.sort()
    n = int(len(fea_2)*0.8)
    print(fea_2[n], '====nnn')
    # print(len(features_all[-2]))
    for f_n in range(len(fea_num)):
        for num in range(len(features_all[f_n])):
            if features_all[f_n][num] > fea_2[n]:
                fea_num[f_n] += 1

    # print(max(features_all[-2]))
    classes = ['normal', 'fault']
    data = [[(len(features_all[-2])+len(features_all[-1])-fea_num[-1]-fea_num[-2]),fea_num[-1]+fea_num[-2]], [(len(features_all[0])+len(features_all[1])+len(features_all[2])-sum(fea_num[:4])),sum(fea_num[:4])]]
    plt_confusion_matrix(data,classes,title[tz],path_tit)
    data_1 = fea_num.copy()
    data_1[-1] = len(features_all[-1])
    # data_2 = [(len(features_all[0])-fea_num[0]),(len(features_all[1])-fea_num[1]),(len(features_all[2])-fea_num[2]),fea_num[-2],fea_num[-1]]
    data_3 = fea_num.copy()

    data_3[-1] = len(features_all[-1])-fea_num[-1]
    data_3[-2] = len(features_all[-2])-fea_num[-2]
    data_2 = [(len(features_all[0])-data_3[0]),(len(features_all[1])-data_3[1]),(len(features_all[2])-data_3[2]),fea_num[-2],fea_num[-1]]
    labels = ['漏报','漏预','正常报警预警','正常数据','误报']
    bar_bottom([data_3,data_2],labels,title[tz],path_tit)
    plt.close()

