import numpy as np
from numpy.fft import fft,ifft
def compute_AF_Correlate(H,X,q):
    n = np.arange(N)
    Receive_H = H.conj()*X#H为滤波器系数,由H.conj()X得到接收信号
    delay_interval = np.arange(1-N,N)
    mainlobe = np.abs(np.dot(X, Receive_H.conj()))
    f_q = fq_values[q]
    doppler_phase = np.exp(1j * 2 * np.pi * f_q * n / N)
    Transmit_X = X * doppler_phase  # 频移后的发射信号
    sidelobes1 = [np.correlate(Transmit_X[np.abs(k):], (Receive_H[:N + k]))  for k in
                  delay_interval[:len(delay_interval) // 2]]
    sidelobes2 = [np.correlate(Transmit_X[:N - k], (Receive_H[k:])) for k in
                   delay_interval[len(delay_interval) // 2:]]
    AF1 = sidelobes1
    AF2 = sidelobes2
    AF = np.append(AF1, AF2)# 固定fq,所有延迟对应的AF值
    return AF
def compute_AF_fft(H, X,q):
    fq = fq_values[q]
    D_q = np.exp(1j * 2 * np.pi * fq / N * np.arange(N))
    H_m = H.conj()*X#发射信号
    x_m = X*D_q  # 应用多普勒频移
    x_m_reverse = x_m[::-1]
    T = 2 * N - 1  # 输出长度满足线性卷积要求
    # 频域加速计算(包含共轭和补零)
    h_padded = np.pad(H_m.conj(), (0, T - N))  # 共轭并后补零[1](@ref)
    x_padded = np.pad(x_m_reverse, (0, T - N))
    # FFT变换与频域相乘
    H_fft = fft(h_padded)
    X_fft = fft(x_padded)
    A_fft = H_fft * X_fft
    # 逆变换与时序调整
    A_full = ifft(A_fft)
    return A_full
if __name__ == '__main__':
    max_iter = 20000
    L = 2
    lambda_val  = 1
    epision = 1/(lambda_val+1)
    mu = 0.5
    Q = 11  # 多普勒单元数,Q必须为奇数
    fq_values = np.linspace(-0.5, 0.5, num=Q)  # 示例多普勒频点
    # 参数设置
    N = 8
    M = 2
    a_max = N * 10 ** (-mu / 20)
    K_range = (-4, 4)
    for k in range(1000):
        H_raw = np.random.randn(N, M) + 1j * np.random.randn(N, M)
        norms_raw = np.linalg.norm(H_raw, axis=0)  # shape=(M,)
        H = H_raw / (norms_raw / np.sqrt(N))[None, :]


        # phases = np.exp(1j * 2 * np.pi * np.arange(L) / L)  # [1, j, -1, -j]
        # phases = np.round(phases, decimals=15)
        phases = np.array([-1,1])
        X = np.array([[np.random.choice(phases) for _ in range(N)] for _ in range(M)])

        AF1 = compute_AF_fft(H[:,0],X[0],0)
        AF2 = compute_AF_Correlate(H[:,0],X[0],0)
        sum = 0 + 0j
        sum1 = 0 + 0j
        for i in range(len(AF1)):
            sum += AF1[i] - AF2[i]
            sum = np.round(sum, 14)
            if np.abs(sum) > 1e-16:
                print("angownoamwpdmawpo不符合")
                break
            print(np.abs(sum))



    # AF2.imag = np.where(AF2.imag < 1e-10, 0, AF2.imag)
    pass