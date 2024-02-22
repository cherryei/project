# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
import os
import tkinter.filedialog
import matplotlib.pyplot as plt


# 汉字图标
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 通过最大正常倍数，取异常振动数据的异常点
def read_shuju(path_0,x=None,y=None,z=None):
    data = pd.read_csv(path_0,encoding='gbk',header=None)
    # print(data)
    mean_all = 0.0001
    beishu_max_all = 160
    lie = data.columns
    data_pl = data[lie[0]].values.tolist()
    yichang = []
    yichang_pl = []
    yichang_index = []
    if x == 1:
        data_x = data[lie[1]].values.tolist()
        # data_pl = np.linspace(0,8000, len(data_x))
        # mean_0 = np.mean(data_x)
        beishu = [item / mean_all for item in data_x]
        for i in range(len(beishu)):
            if beishu[i] > beishu_max_all:
                yichang.append(data_x[i])
                yichang_pl.append(data_pl[i])
                yichang_index.append(i)
    return data_pl,yichang,yichang_pl,yichang_index

# 异常幅值与故障频率、边频大概相等的个数
def guzpl(path_0,path_2,path_3,BPFO_f,BPFI_f,BSF_f,FTF_f,rmp,fs):
    R = float(rmp/60)
    x = 1
    y = 0
    z = 0
    shuju_x = ['x']
    shuju_y = ['y']
    shuju_z = ['z']
    cj_pl = fs
    data_pl,yichang,yichang_pl,yichang_index= read_shuju(path_0, x=x, y=y, z=z)
    lists = {}
    Failure_factor = [R,BPFO_f,BPFI_f,BSF_f,FTF_f]
    Failure_str = ["R","BPFO", "BPFI", "BSF", "FTF"]
    cs_all = []
    for i in range(len(Failure_factor)):
        if (Failure_str[i]) != "R":
            cs = int(cj_pl / (Failure_factor[i] * R))
            cs_all.append(cs)
            lists[str(Failure_str[i])] = list((Failure_factor[i] * R)*(j+1) for j in range(cs))
        else:
            cs = int(cj_pl / R)
            lists[str(Failure_str[i])] = list(R * (j + 1) for j in range(cs))
            cs_all.append(cs)
        if i in [1, 2, 3]:
            for j in range(cs):
                for m in ['R', 'FTF', 'BSF']:
                    FTF_0 = FTF_f * R
                    BSF_0 = BSF_f * R
                    name = ((str(Failure_str[i]) + str(j + 1)) + "_" + m)
                    lists[name] = []
                    a = [-2, -1, 1, 2]
                    if m == "FTF":
                        for bpfo1 in a:
                            lists[name].append(lists[Failure_str[i]][0] * (j + 1) + bpfo1 * FTF_0)
                    elif m == "BSF":
                        if i != 3:
                            for bpfo1 in a:
                                lists[name].append(lists[Failure_str[i]][0] * (j + 1) + bpfo1 * BSF_0)
                    else:
                        for bpfo1 in a:
                            lists[name].append(lists[Failure_str[i]][0] * (j + 1) + bpfo1 * R)
    tz_x = [[],[],[]]
    lie_ame = []
    for i in lists.keys():
        lie_ame.append(i)
        if x == 1:
            count = 0
            for num in lists[i]:
                n = 0
                abnormal = [[],[]]
                for j in range(len(yichang_pl)):
                    if np.abs(yichang_pl[j] - num) <= 3:
                        tz_x[0].append(i)if n == 0 else None
                        abnormal[0].append(yichang[j])
                        abnormal[1].append(yichang_pl[j])
                        count += 1 if n == 0 else 0
                        n += 1
                try:
                    tz_x[1].append(max(abnormal[0]))
                    abm = abnormal[0].index(max(abnormal[0]))
                    tz_x[2].append(abnormal[1][abm])
                except:
                    pass
            shuju_x.append(count)
    D = ['异常数据']
    D = pd.DataFrame([D])
    D.to_csv(path_3, index=False, header=False, mode='w')
    lie_2 = ['坐标轴']+lie_ame
    df = pd.DataFrame([lie_2])
    df.to_csv(path_2, index=False, header=False, mode='w')
    df_x = pd.DataFrame([shuju_x])
    df_x.to_csv(path_2, index=False, header=False, mode='a')
    tz_x[1].insert(0,'x异常频率')
    tz_x[2].insert(0, 'x异常幅值')
    tz_x[0].insert(0, '故障类型')
    A_0 = pd.DataFrame([tz_x[1]])
    A_1 = pd.DataFrame([tz_x[2]])
    A_2 = pd.DataFrame([tz_x[0]])
    A = pd.concat([A_2,A_0,A_1],axis=0)
    A.to_csv(path_3, index=False, header=False, mode='a')

def main():
    fs = 20000
    BPFO_f = 7.8
    BPFI_f = 14.8467
    BSF_f = 7.8
    FTF_f = 4.5633
    rmp = 3000
    # path_0 = tkinter.filedialog.askopenfilename(title='选择异常数据')
    # path_1 = r"C:\Users\IGT\Desktop\验证\1111.csv"
    # path_2 = r"C:\Users\IGT\Desktop\验证\12222.csv"
    # guzpl(path_0, path_1,path_2,BPFO_f,BPFI_f,BSF_f,FTF_f,rmp,fs)
    # path_1 = tkinter.filedialog.askdirectory(title='选择正常数据文件夹')
    path_1 = r"C:\Users\IGT\Desktop\小五轴机预维振动数据\20240122-135101.772\0"
    file_list_all = []
    for iroot, idirs, ifiles in os.walk(path_1):
        if not idirs:
            file_list_all.extend(ifiles)
    path = r"C:\Users\IGT\Desktop\小五轴机预维振动数据\20240122-135101.772\10"
    for i in file_list_all:
        print(i)
        guzpl(path_1+"/"+i, path+'/'+"结果"+i,path+'/'+"特征"+i,BPFO_f,BPFI_f,BSF_f,FTF_f,rmp,fs)

main()