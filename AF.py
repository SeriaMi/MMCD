import numpy as np
import Conv as conv
from scipy.fft import fft, ifft

import numpy as np
from scipy.fft import fft, ifft, fftshift

def compute_AF_fft(h_current, x_m1):
    N = len(h_current)
    M = 2 * N - 1  # 输出长度满足线性卷积要求
    
    # 频域加速计算(包含共轭和补零)
    h_padded = np.pad(h_current.conj(), (0, M - N))  # 共轭并后补零[1](@ref)
    x_padded = np.pad(x_m1, (0, M - N))
    
    # FFT变换与频域相乘
    H_fft = fft(h_padded)
    X_fft = fft(x_padded)
    A_fft = H_fft * X_fft
    
    # 逆变换与时序调整
    A_full = ifft(A_fft)
    # return fftshift(A_full)  # 零时延居中排列[2](@ref)
    return A_full[::-1]
def compute_AF(h_current,x_m1):
    N = len(h_current)
    # 1. 多普勒频移相位
    n = np.arange(N)
    # doppler_phase = np.exp(1j * 2 * np.pi * f_q * n / N)
    # x_m1q = x_m1 * doppler_phase  # 频移后的发射信号 
    x_m1q = x_m1
    # 2. 计算模糊函数 A_{x_{m1} h_m}(k, q)
    A_conj = np.zeros(2 * N - 1, dtype=np.complex128)
    for delay in range(-(N - 1), N):
        delay_idx = delay + (N - 1)
        # 生成时延后的信号
        if delay >= 0:
            x_shifted = np.concatenate([x_m1q[delay:], np.zeros(delay, dtype=np.complex128)])
        else:
            x_shifted = np.concatenate([np.zeros(-delay, dtype=np.complex128), x_m1q[:delay]])
        # 计算模糊函数共轭值
        A_conj[delay_idx] = np.dot(h_current.conj(), x_shifted)
    return A_conj

if __name__ == '__main__':
    # 定义模糊函数
    h_0 = np.array([1, 2, 3,4])
    
    # 定义发射信号  
    x_0 = np.array([1, 2, 3, 4])
    x_0_reverse = np.array([4, 3, 2, 1])
    fft_result = compute_AF_fft(h_0, x_0_reverse)
    AF = compute_AF(h_0, x_0)

    for i in range(1000):
        h1 = np.random.randn(4) + 1j * np.random.randn(4)  # 生成4个复高斯随机数
        x1 = np.random.randn(4) + 1j * np.random.randn(4)  # 生成4个复高斯随机数
        x1_reverse = x1[::-1]
        fft_result1 = compute_AF_fft(h1, x1_reverse)
        AF1 = compute_AF(h1, x1)
        result = [x - y for x, y in zip(AF1, fft_result1)]
        result_cleaned = [
            0 if abs(c) < 1e-10 else c
            for c in result
        ]
        print(result_cleaned)



 