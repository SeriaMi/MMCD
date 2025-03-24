import MM_H_Converge as MM
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz
from numpy.fft import fft,ifft
def compute_AF_fft(h_current, x_m1):

    H = h_current.conj()*x_m1#发射信号
    x_m1_reverse = x_m1[::-1]
    N = len(H)
    M = 2 * N - 1  # 输出长度满足线性卷积要求

    # 频域加速计算(包含共轭和补零)
    h_padded = np.pad(H.conj(), (0, M - N))  # 共轭并后补零[1](@ref)
    x_padded = np.pad(x_m1_reverse, (0, M - N))

    # FFT变换与频域相乘
    H_fft = fft(h_padded)
    X_fft = fft(x_padded)
    A_fft = H_fft * X_fft

    # 逆变换与时序调整
    A_full = ifft(A_fft)
    return A_full
def compute_gamma_max_term(w, N, h_m_i, m1, m, q):
    """
    计算 γ_max(L_{m₁mq}) * N * h_m^{(i)}

    参数:
        w_dict (dict): 权重字典 {k: w}，k ∈ [-N, N]
        N (int): 序列长度
        h_m_i (np.array): 接收滤波器向量，形状为 (N,)

    返回:
        np.array: γ_max * N * h_m^{(i)} 的结果向量
    """
    gamma_max = 0.0
    # 时延范围调整为 k ∈ [-(N-1), N-1]
    for k in range(-(N - 1), N):
        # 计算权重矩阵中的索引（w的第三维长度为 2N-1）
        delay_idx = k + (N - 1)
        w_value = w[m1, m, delay_idx, q]
        effective_length = N - abs(k)
        current_value = w_value * effective_length
        if current_value > gamma_max:
            gamma_max = current_value

    # 计算 γ_max * N * h_m^{(i)}
    return gamma_max * N * h_m_i
def compute_R_m1mqXm1q(w, h_current, x_m1, f_q, m1, m, q):
    """
    计算 R_{m1 m q} x_{m1 q}（完整实现，包含多普勒频移）
    """
    N = len(h_current)
    # 1. 多普勒频移相位
    n = np.arange(N)
    doppler_phase = np.exp(1j * 2 * np.pi * f_q * n / N)
    x_m1q = x_m1 * doppler_phase  # 频移后的发射信号
    #使用fft加速计算
    A_conj = compute_AF_fft(h_current,x_m1q)

    # 3. 构造 Toeplitz 矩阵
    weights = w[m1, m, :, q]

    first_col = weights[N - 1:] * A_conj[N - 1:] #A_conj[N - 1:]:表示获取延迟>=0的AF值的数组，数组的索引对应的延迟K = 0,1,2,3,...
    first_row = weights[N - 1::-1] * A_conj[N - 1::-1]#A_conj[N - 1::-1]:表示获取延迟>=0的AF值的数组，数组的索引对应的延迟K = 0,-1,-2,-3,...
    #R的主对角线及下方元素 R[i,j] = first_col[i-j]
    #主对角线上方元素,R[i,j]=first_row[j−i]
    R = toeplitz(first_col, first_row)
    # 4. 矩阵乘法
    return R @ x_m1q


def compute_u_m(w, N, h_m_i, x_all, f_q_all, M, Q, m, lambda_, a_max):
    """
    计算 u_m = sum_{m1=1}^M sum_{q=1}^Q [R x - γNh] - λ a_max x_m
    """
    sum_term = np.zeros(N, dtype=np.complex128)
    # 遍历所有发射通道和多普勒单元
    for m1 in range(M):
        for q in range(Q):
            # 计算 R_{m1 m q} x_{m1 q}
            R_term = compute_R_m1mqXm1q(
                w, h_m_i, x_all[m1], f_q_all[q], m1, m, q
            )
            # 计算 γ_max * N * h_m^{(i)}
            gamma_term = compute_gamma_max_term(w, N, h_m_i, m1, m, q)
            # 累加项
            sum_term += (R_term - gamma_term)

    # 最终 u_m 公式
    u_m = sum_term - lambda_ * a_max * x_all[m]
    return u_m
def optimize_H_MM(w, X, H_current, f_q_all, lambda_val, a_max, N, M, Q):
    """
    使用MM方法优化接收滤波器H

    参数:
        w: 权重张量，形状为(M, M, 2N-1, Q)
        X: 当前发射序列集合，形状(M, N)
        H_current: 当前接收滤波器，形状(N, M)
        f_q_all: 多普勒频率数组，形状(Q,)
        lambda_val: 正则化参数
        a_max: 最大幅度约束
        N: 序列长度
        M: 通道数
        Q: 多普勒单元数
    返回:
        H_updated: 更新后的接收滤波器，形状(N, M)
    """
    H_updated = np.zeros_like(H_current, dtype=np.complex128)
    for m in range(M):
        # 获取当前通道的接收滤波器
        h_m_i = H_current[:, m]
        # 计算u_m项
        u_m = compute_u_m(w, N, h_m_i, X, f_q_all, M, Q, m, lambda_val, a_max)
        # 计算归一化因子
        norm_u = np.linalg.norm(u_m)
        scaling_factor = -np.sqrt(N) / norm_u if norm_u != 0 else 0
        # 更新h_m
        H_updated[:, m] = scaling_factor * u_m
    return H_updated
def optimize_x_entry(xm, H, d, L, weights, fq_values, epsilon, M, N,a_max):
    """
    优化序列xm的第d个元素。
    参数:
        xm: 当前序列 (N维向量)
        H: 接收滤波器集合 (N x M矩阵)
        d: 待优化的元素索引 (0-based)
        L: 离散相位数量
        weights: 权重系数w_{m1m2}(k, q)
        fq_values: 多普勒频点数组
        epsilon: 帕累托权重
        mu: SNRL参数
        M: 通道数
        N: 序列长度
    返回:
        优化后的相位值 (e^{jφ})
    """

    lambda_val = (1 - epsilon) / epsilon
    v_total = np.zeros(L, dtype=np.float64)
    # 遍历所有通道组合和多普勒频点
    for m2 in range(M):
        for q, fq in enumerate(fq_values):
            D_q = np.exp(1j * 2 * np.pi * fq / N * np.arange(N))
            xm1_q = xm * D_q  # 应用多普勒频移
            h_m2 = H[:, m2]
            xm2_q = [n for n in xm1_q]
            xm2_q[d] = 0
            AF_result = compute_AF_fft(h_m2, xm2_q)
            for k in range(-N + 1, N):
                # 计算a_{m1m2dkq} = h*m(d+k)exp(j2πdfq\N)
                if 0 <= d + k < N:
                    a = np.conj(h_m2[d + k]) * np.exp(1j * 2 * np.pi * d * fq / N)
                else:
                    a = 0
                # # 计算c_{m1m2dkq} (排除n=d)
                c = AF_result[N - 1 + k]
                # 构造η向量并计算DFT
                eta = np.zeros(L, dtype=np.complex128)
                eta[0] = a
                eta[1] = c
                v_kq = np.abs(fft(eta)) ** 2
                # 累加加权后的结果
                weight = weights[m2,k + N-1, q]
                v_total += weight * v_kq
    # 处理SNRL项
    for m in range(M):
        h_m = H[:, m]
        a_md = h_m[d]
        a_md_1 = h_m[d].conj()
        x_conj = xm.conj()

        c_md_1 = np.dot(np.delete(x_conj,d),np.delete(h_m, d)) - a_max
        c_md_1 = c_md_1.conj()
        # h_conj = h_m.conj()
        # c_md_2 = np.dot(np.delete(xm,d),np.delete(h_conj, d)) - a_max
        eta_snrl = np.zeros(L, dtype=np.complex128)
        eta_snrl[0] = a_md
        eta_snrl[1] = c_md_1
        v_snrl = np.abs(fft(eta_snrl)) ** 2
        v_total += lambda_val * v_snrl

    # 找到最优相位
    i_star = np.argmin(v_total)
    phi_star = 2 * np.pi * (i_star - 1) / L
    return np.exp(1j * phi_star)

def generate_weight_matrix(M, N, K_range, Q):
    # nu_range为需要计算的延迟区间,论文实验中指示大于51旁瓣能量已经接近噪声基地
    #生成延迟单元和多普勒频率单元
    k_values = np.arange(-(N-1), N)  # 延迟范围 [-N+1, N-1]
    Q_values = np.arange(Q)
    # 初始化权重矩阵
    w = np.zeros((M, M, len(k_values), Q), dtype=np.float32)

    # 遍历所有通道组合和延迟-多普勒单元
    for m1 in range(M):
        for m2 in range(M):
            for k in k_values:
                for q in Q_values:
                    # 仅处理目标延迟和多普勒区域
                    delay_index = k + N - 1
                    if K_range[0] <= k <= K_range[1]:
                        if m1 == m2:
                            # 自通道：仅在 (k=0, q=0) 处保留主瓣（权重=0），其他位置抑制（权重=1）
                            if k == 0 and  q  == (Q-1) / 2:
                                w[m1, m2, delay_index, q] = 0
                            else:
                                w[m1, m2, delay_index, q] = 10
                        else:
                            # 交叉通道：强制所有位置正交（权重=1）
                            w[m1, m2, delay_index, q] = 10
                    else:
                        # 区域外的权重设为0（不参与优化）

                        w[m1, m2, delay_index, q] = 0
    return w

def optimize_X_CD(X, H, w, epision, mu, M, N, Q, fq_values, K_range,a_max):

    for i in range(M):
        for d in range(len(X[i])):
            optimized_phase = optimize_x_entry(X[i], H, d, L, w[i, :, :, :], fq_values, epision, M, N,a_max)
            X[i][d] = optimized_phase

    return X
def Cost_Curve(cost):
    output_dir = 'outputCurve/'
    os.makedirs(output_dir, exist_ok=True)
    # 绘制收敛曲线
    plt.figure()
    plt.plot(cost, 'b-o',markersize=2)
    plt.xlabel('Iteration')
    plt.ylabel('Total Cost')
    plt.title('Optimization Convergence')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'convergence_curve.png'))



def main_optimization_loop(X,H,M,N,Q,w,epision,lambda_val,fq_values,K_range,max_iter,a_max):
    # 输出文件夹
    output_dir = 'outputTXT001/'
    os.makedirs(output_dir, exist_ok=True)
    cost = []
    WISL = []
    SNRL = []
    for iter in range(max_iter):
        # 记录开始时间
        start_time = time.perf_counter()  # 或 time.time()
        # MM步骤：优化接收滤波器H
        H = optimize_H_MM(w, X, H, fq_values, lambda_val, a_max, N, M, Q)
        
        # CD步骤：优化发射序列X
        X = optimize_X_CD(X, H, w, epision, mu, M, N, Q, fq_values, K_range,a_max)
        # 在main_optimization_loop中添加
        if iter > 200 and abs(cost[-1] - cost[-2]) < 1e-15:
            print("收敛完成！")
            break
        # 每10次迭代保存复数数据
        if iter % 10 == 1:
            cost_iter,wisl,G = compute_cost(X,H,Q,fq_values,N,K_range,M,a_max,w,epision)
            cost.append(cost_iter)
            WISL.append(wisl)
            SNRL.append(G)
            # 计算总耗时
            end_time = time.perf_counter()
            total_time_CD = end_time - start_time
            # 保存接收滤波器H (NxM复数矩阵)
            # print(f"第{iter}次迭代  保存接收滤波器H,总运行时间: {total_time_CD:.6f} 秒")
            # print("H:\n", H.T)
            # print("X:\n", X)
            print(f"第{iter}次迭代,cost:{cost_iter},WISL:{wisl},SNRL:{G}总运行时间: {total_time_CD:.6f} 秒")

            for m in range(M):
                np.savetxt(os.path.join(output_dir, f'H_m{m+1}_iter_{iter+1}.txt'), H[:, m])
                np.savetxt(os.path.join(output_dir, f'X_m{m+1}_iter_{iter+1}.txt'), X[m])
            if iter % 100 == 1:
            # 在优化循环结束后添加绘图代码
                Cost_Curve(cost)

        # 计算目标函数值判断收敛
        # current_obj = compute_objective(...)
        # if converged: break

def compute_cost(X,H,Q,fq_values,N,K_range,M,a_max,w,epision):
    # 计算目标函数值
    # 计算WISL
    delays_count = K_range[1] - K_range[0] + 1
    WISL = 0
    G = 0
    AF_KQ = np.zeros((2*N-1, Q), dtype=np.float32)
    for m1 in range(M):
        for m2 in range(M):
            for q in range(Q):
                x_w = X[m1] * np.exp(1j * 2 * np.pi * fq_values[q] * np.arange(N))  # 频移后的发射信号
                h_w = H[:, m2]  # 待优化的接收滤波器
                # 计算AF并存入当前列
                AF_Q = MM.compute_AF_fft(h_w, x_w)
                # 截取指定延迟范围 [K_range[0], K_range[1]]
                AF_KQ[:, q] = np.abs(AF_Q)**2
                WISL += np.sum(w[m1, m2, :, q] * AF_KQ[:, q])

    for m in range(M):
        x_m = X[m]
        h_m = H[:, m]
        term = np.vdot(x_m, h_m) - a_max  # 使用vdot处理复数内积
        G += np.abs(term)**2  # 正确计算平方
    cost = epision*WISL + (1-epision)* G
    return cost ,WISL,G
if __name__ == '__main__':
    # 示例用法
    max_iter = 20000
    L = 2
    lambda_val  = 0
    epision = 1/(lambda_val+1)
    mu = 0.5
    Q = 11  # 多普勒单元数,Q必须为奇数
    fq_values = np.linspace(-0.5, 0.5, num=Q)  # 示例多普勒频点
    # 参数设置
    N = 20  # 序列长度
    M = 2  # 发射/接收通道数
    a_max = N * 10 ** (-mu / 20)
    K_range = (-4, 4)
    # 初始化参数
    # 步骤1：生成随机复数矩阵
    H_raw = np.random.randn(N, M) + 1j * np.random.randn(N, M)
    # 步骤2：计算原始范数
    norms_raw = np.linalg.norm(H_raw, axis=0)  # shape=(M,)
    #步骤3：归一化操作
    H_normalized = H_raw / (norms_raw / np.sqrt(N))[None, :]
    # 初始化离散相位序列 X (L=4相位)
    phases = np.exp(1j * 2 * np.pi * np.arange(L) / L)  # [1, j, -1, -j]
    X = np.array([[np.random.choice(phases) for _ in range(N)] for _ in range(M)])
    # 方案1：数值修约（推荐）
    X = np.round(X, decimals=15)  # 保留15位小数精度
    w = generate_weight_matrix(M, N, K_range, Q)

    main_optimization_loop(X,H_normalized,M,N,Q,w,epision,lambda_val,fq_values,K_range,max_iter,a_max)

