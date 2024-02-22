# #选择对称且连续的小波类型
# from matplotlib import pyplot as plt
# import pywt
# 使用python实现小波变换
#
# import numpy as np
# import pywt
# import matplotlib.pyplot as plt
# # from scipy import signal
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
#
# # 生成信号变量
# t = np.linspace(0, 1, num=5000)
# signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t) + np.sin(3 * np.pi * 30 * t)
#
# # 添加随机噪声
# noise = np.random.normal(0, 0.05, len(signal))
# signal = signal + noise

# 常见的几种小波基函数包括：

# 1. Daubechies小波基（db）：Daubechies小波基是最常用的小波基函数之一。它具有紧凑支持和良好的频率局部化特性。常见的Daubechies小波基包括db2、db4、db6等。

# 2. Symlets小波基（sym）：Symlets小波基是对称的Daubechies小波基。它们在频率局部化和相位对称性方面与Daubechies小波基类似。常见的Symlets小波基包括sym2、sym4、sym8等。

# 3. Coiflets小波基（coif）：Coiflets小波基是具有紧凑支持和较好频率局部化特性的小波基。它们在一些应用中比Daubechies小波基具有更好的性能。常见的Coiflets小波基包括coif1、coif2、coif3等。

# 4. Biorthogonal小波基（bior）：Biorthogonal小波基是一组成对的小波基函数。它们具有可变的支持长度和频率响应。常见的Biorthogonal小波基包括bior2.2、bior3.3、bior6.8等。

# wavelet_name = 'db4'  # 定义小波基名称为'db4'
# wavelet_name = 'sym4'  # 定义小波基名称为'sym4'
# wavelet_name = 'bior3.3'  # 定义小波基名称为'bior3.3'
# coeffs = pywt.wavedec(signal, wavelet_name,level=20) # 离散小波
# details = [c[1:] for c in coeffs]
# coeffs_filtered = coeffs[:3+1]
# # reconstructed_signal = pywt.waverec(coeffs_filtered, wavelet_name)
# averages = []
# for detail in coeffs[1:]:  # 忽略近似系数
#     u = np.array(detail)  # 将细节系数转换为numpy数组
#     average = np.mean(u)  # 计算平均值
#     averages.append(average)

#  # 定义连续小波变换参数
# cwt_params = {
#     'scales': np.arange(1, 31),  # 尺度范围
#     'wavelet': 'morl',  # 小波类型
#     'method': 'conv',  # 变换方法
#
# }
# # 小波变换
# coeffs = pywt.cwt(signal, **cwt_params)  # 使用指定小波基进行4级小波分解
# # print(len(coeffs[0][29]),len(coeffs[1]))
#
# wavename = 'cgau8'
# totalscal = 256
# fc = pywt.central_frequency(wavename)
# cparam = 2 * fc * totalscal
# scales = cparam / np.arange(totalscal, 1, -1)
# # print(scales)



# 绘制原始信号图像
# plt.figure(figsize=(8, 6))
# plt.subplot(5, 1, 1)
# plt.plot(t, signal)
# plt.title('Original Signal')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
#
# # 绘制小波分解信号图像
# for i in range(1, len(coeffs)):
#     plt.subplot(5, 1, i + 1)
#     plt.plot(t[:len(coeffs[i])], coeffs[i])
#     plt.title(f'Wavelet Coefficients - Level {i}')
#     plt.xlabel('Time')
#     plt.ylabel('Amplitude')
#
# # plt.subplot(5, 1, 5)
# # plt.plot(t[:len(reconstructed_signal)], reconstructed_signal)
# # plt.title('Reconstructed Signal')
# # plt.xlabel('Time')
# # plt.ylabel('Amplitude')
# plt.tight_layout()
# plt.show()



# M = 100
# s = 4.0
# w = 2.0
# wavelet = signal.morlet(M, s, w)
# plt.plot(wavelet.real, label="real")
# plt.plot(wavelet.imag, label="imag")
# plt.legend()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pywt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 生成三个不同频率成分的信号  3000个点
fs = 1000  # 采样率
time = np.linspace(0, 1, fs, endpoint=False)  # 时间
# 第一个频率成分
signal1 = np.sin(2 * np.pi * 30 * time)
# 第二个频率成分
signal2 = np.sin(2 * np.pi * 60 * time)
# 第三个频率成分
signal3 = np.sin(2 * np.pi * 120 * time)
# 合并三个信号
signal = np.concatenate((signal1, signal2, signal3))

# 连续小波变换参数
# 采样频率
sampling_rate = 3000
# 尺度长度
totalscal = 128
# 小波基函数
wavename = 'morl'
# 小波函数中心频率
fc = pywt.central_frequency(wavename)
# 常数c
cparam = 2 * fc * totalscal
# 尺度序列
scales = cparam / np.arange(totalscal, 0, -1)
print(scales)
 # 进行CWT连续小波变换
coefficients, frequencies = pywt.cwt(signal, scales, wavename, 1.0/1000)
plt.plot(coefficients)
# 小波系数矩阵绝对值
amp = abs(coefficients)
print(len(coefficients),len(coefficients[0]))

# 根据采样频率 sampling_period 生成时间轴 t
t = np.linspace(0, 1.0/sampling_rate, sampling_rate, endpoint=False)
# 绘制时频图谱
plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
plt.plot(signal)
plt.title('30Hz和60Hz和120Hz的分段波形')
plt.subplot(2,1,2)
amp_ = np.zeros_like(amp)
amp_[1]=amp[60]
plt.contourf(t, frequencies, amp, cmap='jet')
plt.title('对应时频图')

plt.show()
