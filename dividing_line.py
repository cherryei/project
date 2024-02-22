import numpy as np
import pandas as pd
import math
from feater import *


def find_optimal_threshold(data1, data2):
    num = len(data1)-len(data2)
    # 数据预处理，确保两组数据有相同的长度
    try:
        data1 = np.pad(data1, [(0, data2.shape[0] - data1.shape[0])], 'constant')
    except:
        data2 = np.pad(data2, [(0, data1.shape[0] - data2.shape[0])], 'constant')
    print(data1,data2)

    # 计算平均值作为初始分界线
    threshold = (data1.mean() + data2.mean()) / 2

    # 初始化计数器
    count_above = 0
    count_below = 0

    # 计算每个数据点与分界线的比较结果
    above = (data1 > threshold)
    print(above)
    below = (data2 < threshold)
    count_above += above.sum()
    count_below += below.sum()

    # 迭代调整分界线，直到达到最优解或达到最大迭代次数
    for _ in range(10000):  # 例如，最大迭代10次
        old_count_above = count_above
        old_count_below = count_below

        # 尝试提高分界线，看看是否能减少大于分界线的数量
        for i in range(len(data1)):
            if above[i] == 0 and below[i] == 1:  # 数据点在分界线下，尝试提高分界线使其通过该点
                threshold += 0.01  # 每次迭代增加0.01单位，根据实际情况调整步长
                break
        else:  # 如果上面的循环没有找到合适的点，则尝试降低分界线
            threshold -= 0.01  # 每次迭代减少0.01单位，根据实际情况调整步长

        # 重新计算计数器
        above = (data1 > threshold)
        below = (data2 < threshold)
        count_above = above.sum()
        count_below = below.sum()

        # 检查是否达到最优解（例如，计数器没有变化）
        # if count_above == old_count_above and count_below == old_count_below:
        #     break  # 找到最优解，退出循环
        if count_above + count_below >= old_count_above + old_count_below:
            break  # 找到最优解，退出循环
    else:  # 如果最大迭代次数已到，但未找到最优解，则可以抛出错误或返回一个警告信息
        raise ValueError("Failed to find optimal threshold.")  # 或其他适当的错误处理机制

    return threshold, count_above, count_below-num  # 返回分界线值以及大于和小于分界线的数量


# a = [1,2,3,4,5,5.5,5.6,5.7,7,8,9,10]
# b = [3,4,5,5.1,5.3,5.7,7,8,13,14,16,17,18,19,19,30,27,36]
docker = pd.read_csv(r"C:\luojin\预维新特征\残余能量特征交叉验证\pickter\箱型图\全部验证数据.csv",encoding='gbk')
a = docker["normal"].values.tolist()
b = docker["fault"].values.tolist()

is_nan = np.isnan(b)
b = [x for i, x in enumerate(b) if not is_nan[i]]
a.sort()
n = int(len(a) * 0.8)
print(a[n], '====nnn')
# print(len(features_all[-2]))
features_all = [a,b]
fea_num = [0,0]
for f_n in range(len(fea_num)):
    for num in range(len(features_all[f_n])):
        if features_all[f_n][num] > a[n]:
            fea_num[f_n] += 1
# m = find_optimal_threshold(np.array(a), np.array(b))
classes = ['normal', 'fault']
data = [[(len(a)-fea_num[0]),fea_num[0]],[len(b)-fea_num[1],fea_num[1]]]
plt_confusion_matrix(data, classes, "正能量残差峰度", r"C:\luojin\预维新特征\第二阶段批量验证数据\pectier")
# bar_bottom()
# print(m)
