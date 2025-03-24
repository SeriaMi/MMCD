import numpy as np
#import tensorflow.compat.v1 as tf
#import tensorflow as tf
import scipy.io as sio
# import matplotlib.pyplot as plt

from matplotlib import pyplot as plt

from math import pi
from datetime import datetime

from numpy.linalg import matrix_rank

# tf.keras.backend.set_floatx('float64')


#单脉冲波形
def OFDM_pulse(f0, B, noc, d,
               ts):  # generate OFDM signal, f0 carrier frequency, B bandwidth, noc num of carriers, d data bit
    T = noc / B
    t = np.arange(0, T, ts)  # 用于生成等间隔的数值数组
    g = np.zeros(len(t), complex)
    for i in range(noc):
        fs = i * B / noc
        f = d[i] * np.exp(1j * 2 * np.pi * (fs + f0) * t)
        g = g + f
    return g

#能量归一化
def BPSK_modul(seq):
    res = []
    for i in range(len(seq)):
        if seq[i] == 1:
            res.append(-1 / np.sqrt(2) - 1j / np.sqrt(2))  # append: add obj in the end of the list
        else:
            res.append(1 / np.sqrt(2) + 1j / np.sqrt(2))
    return res

#格雷互补序列
def construct_GCP(N):
    F1, F2 = np.array([1, 1]), np.array([1, -1])
    L = 2
    while L != N:
        tmp1 = np.concatenate((F1, F2))  # concatenate default is 0, [[f1],[f2]]锛屼袱涓煩闃佃繛璧锋潵
        tmp2 = np.concatenate((F1, -1 * F2))
        F1, F2 = tmp1, tmp2
        L *= 2  # L=L*2
    return F1, F2

#画模糊函数
def plot_3D_AF(AF, delay_interval, doppler_intereval, ts, T,runtime):
    fig = plt.figure(figsize=(16, 9), dpi=300)
    ax = plt.axes(projection='3d')
    xx = delay_interval / ts * T
    yy = doppler_intereval
    X, Y = np.meshgrid(xx, yy)
    Z = AF
    ax.plot_surface(X, Y, Z, cmap='rainbow')
    # ax.set_zlim(-90, 0)
    ax.ticklabel_format(style='sci', scilimits=(-1, 1), axis='x')
    ax.ticklabel_format(style='sci', scilimits=(-1, 1), axis='y')
    ax.set_xlabel('Delay')
    ax.set_ylabel('Doppler shift (Hz)')
    ax.set_zlabel('$\chi(f,\\tau)$ (dB)')
    # 将 timedelta 对象转换为字符串
    zticks = [-90,-80,-70,-60,-50,-40,-30,-20,-10,0]
    ax.set_zticks(zticks)
    run_time_str = str(run_time)
    # 去掉字符串中的空格
    run_time_str_without_space = run_time_str.replace(" ", "")
    new_str = run_time_str_without_space.replace(":", "").replace(",", "")
    plt.savefig(f'plot__time{new_str}.png')
    plt.show()



COMP_RATIO = 2
DEC_LAYERS = 5
EPOCH_SIZE = 2000
PRT_INTERV = 5
LEARN_RATE = 1e-5
BATCH_SIZE = 512


if __name__ == '__main__':
    start_time = datetime.now()

    f0, B, noc, d, ts = 60, 1.76, 1, [1 / np.sqrt(2) + 1j / np.sqrt(2)], 0.05
    # 生成OFDM单脉冲波形（使用正交频分复用技术）
    pluse = OFDM_pulse(f0, B, noc, d, ts)  # 参数：载频60MHz，带宽1.76MHz，1个子载波，QPSK调制符号

    # 构造格雷互补序列对（Golay Complementary Pair）
    L = 64  # 序列长度
    gcpx, gcpy = construct_GCP(L)  # 生成64阶格雷互补对
    print(gcpx, gcpy)  # 输出验证序列

    # 通过克罗内克积生成发射信号
    s_x = np.kron(gcpx, pluse)  # 将格雷序列与脉冲波形进行波形扩展
    s_y = np.kron(gcpy, pluse)  # 生成互补信号对
    Di = 3  # 2*ny.pi*fd*T
    N = 20
    M = N - 1
    Q = [1.256590093724545535e-16 - 1.265040131234983045e+00j,
    -1.318491083563192862e-16 + 1.891460655031030669e+00j,
    5.571089085478280186e-18 - 5.673594862907342939e-01j,
    8.542336597733363106e-17 - 1.603551318837289852e+00j,
    -1.386582172385705291e-16 + 2.322464669911263879e+00j,
    -1.671326725643483979e-17 - 8.152602156268817790e-01j,
    1.838459398207832377e-16 - 1.232740481071595173e+00j,
    -2.562700979320008932e-16 + 1.588220708198603948e+00j,
    1.145168312014979854e-17 - 5.799974657496299590e-01j,
    2.556510880336144322e-16 - 7.240145022135829889e-01j,
    -2.841255433593922764e-16 + 8.554220283434494920e-01j,
    4.085465329350738957e-17 - 3.406510499801998493e-01j,
    2.553415830844211771e-16 - 2.585563744044864776e-01j,
    -2.847445532577787866e-16 + 2.159319812745986444e-01j,
    5.857381163482025304e-17 - 4.495160696914529685e-02j,
    2.282599000300128577e-16 - 3.296289935604815313e-02j,
    -2.511245781516632918e-16 + 2.672529018844145739e-03j,
    5.957970271969827055e-17 - 3.393761289345887527e-02j,
    1.311720662799590807e-16 + 4.123734432179407966e-02j,
    -1.273612865930173581e-16 - 1.903361658532479531e-02j]
    P = np.array([ 1.000000000000000000e+00+0.000000000000000000e+00j,
 -1.000000000000000000e+00-1.224646799147353207e-16j,
 -1.000000000000000000e+00-1.224646799147353207e-16j,
 1.000000000000000000e+00+0.000000000000000000e+00j,
 -1.000000000000000000e+00-1.224646799147353207e-16j,
 -1.000000000000000000e+00-1.224646799147353207e-16j,
 1.000000000000000000e+00+0.000000000000000000e+00j,
 -1.000000000000000000e+00-1.224646799147353207e-16j,
 -1.000000000000000000e+00-1.224646799147353207e-16j,
 1.000000000000000000e+00+0.000000000000000000e+00j,
 -1.000000000000000000e+00-1.224646799147353207e-16j,
 -1.000000000000000000e+00-1.224646799147353207e-16j,
 1.000000000000000000e+00+0.000000000000000000e+00j,
 -1.000000000000000000e+00-1.224646799147353207e-16j,
 -1.000000000000000000e+00-1.224646799147353207e-16j,
 1.000000000000000000e+00+0.000000000000000000e+00j,
 -1.000000000000000000e+00-1.224646799147353207e-16j,
 -1.000000000000000000e+00-1.224646799147353207e-16j,
 1.000000000000000000e+00+0.000000000000000000e+00j,
 1.000000000000000000e+00+0.000000000000000000e+00j,])


    # print("p归一化一范数:\n",P)
    print("q归一化一范数:\n",Q)
    T = 40000 #对应信号的采样率
    x1, y1 = np.zeros(T, complex), np.zeros(T, complex)  # x1鐨勯暱搴︿负T
    x1[:len(s_x)] = s_x  # x1
    y1[:len(s_y)] = s_y

    s = np.zeros((N, T), complex)
    s_r = np.zeros((N, T), complex)
    for i in range(N):
        if P[i] == 1:
            s[i, :] = x1
        else:
            s[i, :] = y1#每行表示一个发射信号为x1或者y1
        s_r[i, :] = np.dot(Q[i].conjugate(), s[i, :])#接收信号的表示，每一行滤波器系数的共轭 * 信号(x1或者y1)，用于表示接受信号
    s = s.flatten() #发射序列,将x1或者y1的信号展开成一行进行发射 [x1,y1,x1,x1...]
    s_r = s_r.flatten() #接收序列,为np.dot(Q[i].conjugate(), s[i, :])，
    doppler_interval = np.linspace(-4.587*10**4, 4.587*10**4, 400)
    delay_interval = np.arange(-len(s_x) + len(pluse), len(s_x), len(pluse))
    t_interval = np.arange(len(s)) / 40000 * 2 * 10 ** -6  #40000 * 2 * 10 ** -6 是什么意思？？
    L = len(s)
    AF = np.zeros((len(doppler_interval), len(delay_interval)))

    mainlobe = np.abs(np.dot(s, s_r.conjugate()))
    print("mainlobe is \n", mainlobe)

    for i in range(len(doppler_interval)):
        fd = doppler_interval[i]
        s_tmp = s.T * np.exp(2 * pi * 1j * fd * t_interval)#fd为多普勒频移
        sidelobes1 = [(np.abs(np.correlate(s_tmp[np.abs(k):], (s_r[:L + k])))) + 10 ** -9 for k in
                      delay_interval[:len(delay_interval) // 2]]#k为负数发射序列左移
        sidelobes2 = [(np.abs(np.correlate(s_tmp[:L - k], (s_r[k:])))) + 10 ** -9 for k in
                      delay_interval[len(delay_interval) // 2:]]#k为正数发射序列右移补0，即可视为发射序列减少k个元素，为了保证序列长度一致接收序列从后往前减少k个元素
        # print("sidelobe1 is \n", sidelobes1)
        # print("sidelobe2 is \n", sidelobes2)
        AF1 = 20 * np.log10((sidelobes1) / mainlobe)
        AF2 = 20 * np.log10((sidelobes2) / mainlobe)
        AF[i][:] = np.append(AF1, AF2)


    for i in range(len(AF)):
        for j in range(len(AF[0])):
            if AF[i][j] <= -90:
                AF[i][j] = -90

    outfile = 'saved_array.npy'
    print("AF is \n", AF)
    np.save(outfile, AF)


    end_time = datetime.now()
    run_time = end_time - start_time
    run_time_str = str(run_time)
    # 去掉字符串中的空格

    run_time_str_without_space = run_time_str.replace(" ", "")
    run_time_str_without_dot = run_time_str_without_space.replace(".", "")
    # 去掉字符串中的冒号
    run_time_str_without_colon = run_time_str_without_dot.replace(":", "")
    plot_3D_AF(AF, delay_interval, doppler_interval, T, 2 * 10 ** -6,run_time_str_without_colon)


    print(run_time)
