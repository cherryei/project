import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.DataFrame(data=np.random.randint(0,100,(100,4)),columns=list("ABCD"))
print(df)

#解决不能显示中文的问题
plt.rcParams['font.sans-serif']=['Simhei']
plt.rcParams['axes.unicode_minus']=False

#第一个散点图，颜色为红色，透明度50%，图例为散点图1
plt.scatter(df["A"],df["B"]+100,c="r",alpha=0.5,label="散点图1")
#第二个散点图，颜色为蓝色，透明度50%，图例为散点图2
plt.scatter(df["C"],df["D"],c="b",alpha=0.5,label="散点图2")

#更改X轴和Y轴的范围
plt.xlim([-10,130])
plt.ylim([-10,120])

#显示图例
plt.legend(loc="best")

#给标题
plt.title("散点图")
plt.show()

