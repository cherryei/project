import numpy as np
import matplotlib.pyplot as plt

# 创建混淆矩阵
confusion_matrix = np.array([[10, 5, 3],
                             [2, 15, 1],
                             [4, 2, 12]])

# 获取混淆矩阵的行数和列数
num_classes = confusion_matrix.shape[0]

# 创建画布和子图
fig, ax = plt.subplots()
ax.set_aspect('equal')

# 绘制网格线
for i in range(num_classes + 1):
    ax.axhline(i, color='black', lw=1)
    ax.axvline(i, color='black', lw=1)

# 绘制矩阵中的数字
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j + 0.5, i + 0.5, confusion_matrix[i, j], ha='center', va='center')

# 设置坐标轴标签
ax.set_xticks(np.arange(num_classes) + 0.5)
ax.set_yticks(np.arange(num_classes) + 0.5)
ax.set_xticklabels(['Class 1', 'Class 2', 'Class 3'])
ax.set_yticklabels(['Class 1', 'Class 2', 'Class 3'])

# 设置坐标轴标题
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')

# 显示网格和矩阵
plt.grid(True)
plt.show()