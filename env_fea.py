import numpy as np
from scipy.signal import find_peaks
import pandas as pd
from scipy.fftpack import fft,fftshift
from matplotlib import pyplot as plt
# from Fault_algorithm import *
from scipy.signal import butter, filtfilt,lfilter

def butter_bandstop(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, data)
    return y

def butter_highpass(data, highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='highpass')

    y = lfilter(b, a, data)
    return y

def butter_lowpass(data, lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='lowpass')
    y = lfilter(b, a, data)
    return y

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
    return max_all,whe

def fastft(data,fs):
    signal = data
    N = len(data)                           # 采样点数
    sample_freq = fs                  # 采样频率 120 Hz, 大于两倍的最高频率
    # sample_interval = 1/sample_freq     # 采样间隔
    # signal_len = N*sample_interval      # 信号长度
    # t = np.arange(0,signal_len,sample_interval)

    fft_data = fft(signal)
    fft_amp0 = np.array(np.abs(fft_data)/N*2)   # 用于计算双边谱
    fft_amp0[0] = 0.5*fft_amp0[0]
    N_2 = int(N/2)
    fft_amp1 = fft_amp0[0:N_2]  # 单边谱

    # 计算频谱的频率轴
    list1 = np.array(range(0, int(N/2)))
    freq1 = sample_freq*list1/N        # 单边谱的频率轴

    return freq1,fft_amp1




def envelope(y, m, mode):
    """
    计算信号的包络线。

    参数：
        y：信号数组。
        m：包络线的阶数。
        mode：包络线的类型，'peak'表示峰值包络线。

    返回：
        up：上包络线。
        low：下包络线。
    """
    if mode == 'peak':
        # 找到峰值和谷值
        peaks, _ = find_peaks(y, distance=np.Inf)
        valleys, _ = find_peaks(-y, distance=np.Inf)
        # 计算上包络线和下包络线
        up = np.interp(np.arange(len(y)), peaks, y[peaks])
        low = np.interp(np.arange(len(y)), valleys, y[valleys])
    else:
        raise ValueError('Invalid mode')
    return up, low



# 生成测试信号
# t = np.linspace(0, 1, 10000)
# y = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t) + np.sin(2 * np.pi * 200 * t) + 0.2 * np.random.randn(
#     len(t))

path = r"C:\luojin\预维新特征\第二阶段批量验证数据\VWED数据转换\正常报警预警\报警\40-1-SP1\CBS-1-2023-01-09-06-23-05.csv"
doker = pd.read_csv(path, encoding='gbk')
k = 8000
data = doker.iloc[:, 1]
data = data[5 * k:int(6.5 * k + 1)].values.tolist()
data = butter_highpass(data, 3000, k, order=5)
freq3,fft_amp3 = fastft(data,8000)
plt.plot(data)
plt.show()
plt.plot(freq3,fft_amp3)
plt.show()
data = abs(np.array(data))
# data = [x for x in data if x >= 0]
data = butter_lowpass(data, 1700, k, order=5)
freq4,fft_amp4 = fastft(data,8000)
plt.plot(data)
plt.show()
plt.plot(freq4,fft_amp4)
plt.show()
y = np.array(data)
t = np.linspace(0, 1, len(y))

# 计算包络线
m = 12
up,whe = envelope_features(data,m)
# up, low = envelope(y, m, 'peak')
freq1,fft_amp1 = fastft(up,8000)
freq2,fft_amp2 = fastft(y,8000)
# 绘制原始信号和包络线
plt.figure()
plt.plot(t, y, 'b', label='Original signal')
t_ = np.linspace(0, 1, len(up))
plt.plot(t_,up, 'r', label='Upper envelope')
# plt.plot(t, low, 'g', label='Lower envelope')
plt.legend()
plt.show()

plt.plot(freq1,fft_amp1)
plt.title('Upper envelope fft')
# plt.axhline(np.mean(fft_amp1)*8)
plt.show()

# plt.plot(freq2,fft_amp2)
# plt.title('Original signal fft')
# # plt.axhline(np.mean(fft_amp1)*8)
# plt.show()

fft_max_all = []
for ft_m in range(len(fft_amp1)):
    fft_max_all.append(freq1[ft_m]) if fft_amp1[ft_m] > np.mean(fft_amp1)*5 else None

fft_diff = []
for ft_d in fft_max_all:
    fft_diff.extend(list(abs(ft_d-i) for i in fft_max_all))
bpfi = 1689
bpfu_num = 0
for num in fft_diff:
    if num==bpfi:
        bpfu_num += 1
print(bpfu_num)
# main()