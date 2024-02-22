# # 假设字符串列表为 my_list
# my_list = ['hello', 'worlnnd', 'pythonn', 'python', 'hello',"heeelo"]
#
# # 使用列表推导式遍历列表，检查每个字符串中某个字符是否出现两次
# new_list = [s for s in my_list if s.count('n') <= 1]
#
# print(new_list)
#
# import numpy as np
#
# # 假设你有以下的二维数组
# arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#
# # 你可以使用mean函数沿着axis=0（即沿着列）来求均值
# mean_arr = arr.mean(axis=1)
#
# print(mean_arr)
import random

import matplotlib.pyplot as plt
import pandas as pd

# # 创建数据
# x = np.arange(5)
# y1 = np.array([2, 3, 1, 4, 2])
# y2 = np.array([1, 1, 2, 2, 1])
# y3 = np.array([1, 2, 3, 2, 1])
#
# # 创建堆积图
# fig, ax = plt.subplots()
# ax.bar(x, y1, label='y1')
# ax.bar(x, y2, bottom=y1, label='y2')
# ax.bar(x, y3, bottom=y1 + y2, label='y3')
#
# # 添加数值标签
# for i in range(len(x)):
#     ax.text(i, y1[i]*0.5, str(y1[i]), ha='center')
#     ax.text(i, y1[i] + y2[i] *0.5, str(y2[i]), ha='center')
#     ax.text(i, y1[i] + y2[i] + y3[i] *0.5, str(y3[i]), ha='center')
#
# # 设置图表属性
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_xticks(x)
# ax.set_xticklabels(('A', 'B', 'C', 'D', 'E'))
# ax.legend()
#
# # 显示图表
# plt.show()


# import pandas as pd
#
# # 假设有一个名为my_list的列表，包含数据
# # my_list = [['Tom', 10], ['Nick', 15], ['John', 20]]
# #
# # # 将列表转换为DataFrame
# # df = pd.DataFrame(my_list, columns=['Name', 'Age'])
# #
# # # 打印DataFrame
# # print(df)
import seaborn as sns
import numpy as np

# 假设你有一个列表如下：
lst = [[1, 2, 3], [4, 5, 6]]


# 你可以将其转换为numpy数组，然后进行转置：
arr = np.array(lst)
transposed_arr = np.transpose(arr)
print(transposed_arr)
df = pd.DataFrame(list(transposed_arr), columns=['Name', 'Age'])
print(df)
ax = sns.boxplot(data=df)
plt.show()
#
# n = 5  # 指定长度
# empty_2d_list = [[] * i for i in range(n)]
# print(empty_2d_list)

# import numpy as np
#
# # 定义数组的长度
# length = 10
#
# # 生成全零数组
# zero_array = np.zeros(length)
#
# print(zero_array)
# list1 = [1, 2, 3, 4, 5]
# list2 = [3, 4, 5, 6, 7]
#
# # 将两个列表转换为集合
# lis = [list1,list2]
# lis = np.array(lis)
# mean_arr = lis.mean(axis=0)
# print(mean_arr)
# # 执行减法操作
# # result = set1 - set2
#
# # 将结果转换回列表
# # result_list = list(result)
#
# # print(list1[:-1])  # 输出: [1, 2]