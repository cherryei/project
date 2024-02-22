import numpy as np
from scipy.fftpack import fft,fftshift
import matplotlib.pyplot as plt

import pandas as pd


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def fastft(path_0, path_1):
    data_1 = pd.read_csv(path_0,encoding='gbk')
    y = data_1.iloc[:,1].values.tolist()
    signal = y[0:80000]
    signal_ = y[-80001:-1]
    N = 80000                           # 采样点数
    sample_freq = 8000                  # 采样频率 120 Hz, 大于两倍的最高频率
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
