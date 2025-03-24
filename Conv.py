from scipy.signal import convolve
import numpy as np
from scipy.fft import fft, ifft
def fft_conv(x, h):
    # 计算输出长度并填充到 N + M - 1
    N = len(x) + len(h) - 1
    x_padded = np.pad(x, (0, N - len(x)))  # 后端补零
    h_padded = np.pad(h, (0, N - len(h)))

    # FFT变换与频域相乘
    X = fft(x_padded)
    H = fft(h_padded)
    Y = X * H

    # 逆变换并取实部（虚部由计算误差导致）
    y = ifft(Y)
    return y[:len(x) + len(h) - 1]  # 截断到有效长度

if __name__ == '__main__':
    x = [1, 2, 3,4]
    h = [5,6,7,8]
    fft_result = fft_conv(x,h)
    result = convolve(x, h, method='auto')  # 自动选择快速算法（FFT或直接计算）
    print("Scipy卷积结果:", result)  # 输出: [0, 1, 4, 8, 8, 6]

