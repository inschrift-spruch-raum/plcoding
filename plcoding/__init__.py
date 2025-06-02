import numpy as np


# 给定概率分布，计算随机变量的base熵
def entropy_of(prob: np.ndarray, base: int = 2) -> float:
    prob = prob[np.nonzero(prob)]  # 避免 log(0)
    return np.sum(-prob * np.log(prob)) / np.log(base)

# 常用的base-2熵
def h2_of(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return entropy_of(np.array([p, 1 - p]), base=2)

# 给定一个base-2熵值，返回对应的概率
def p_of(h2: float, tol: float = 1e-10) -> tuple:
    low, high = 0.0, 0.5
    while high - low > tol:
        mid = (low + high) / 2
        h = h2_of(mid)
        if h < h2:
            low = mid
        else:
            high = mid
    p = (low + high) / 2
    return (p, 1 - p)

# 生成指定长度的比特翻转索引
def bitrev_perm(len: int) -> np.ndarray:
    n = int(np.log2(len))
    assert 2 ** n == len, "len must be a power of 2"
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(len)])

# 计算二进制擦除信道的极化容量
def bec_channels(level: int, e: float = 0.5, _log: bool = False) -> np.ndarray:
    log_e = np.log(e)
    z_list = [log_e]
    for _ in range(level):
        new_list = []
        for log_z in z_list:
            log_z0 = log_z + np.log(2 - np.exp(log_z) + 1e-100)
            log_z1 = 2 * log_z
            new_list.extend([log_z0, log_z1])
        z_list = new_list
    z_arr = np.array(z_list)
    return z_arr if _log else np.exp(z_arr)

# 最简单的二进制极化码编码（采用递归算法）
def basic_encode(input: np.ndarray) -> np.ndarray:
    N, W_size = len(input), len(input)
    u, x = np.copy(input), np.empty_like(input)
    while W_size > 1:
        for j in range(0, N, 2):
            x[j] = (u[j] + u[j + 1]) % 2
            x[j + 1] = u[j + 1]
        for j in range(0, N, W_size):
            for k in range(0, W_size, 2):
                u[j + int(k / 2)] = x[j + k]
                u[j + int((W_size + k) / 2)] = x[j + k + 1]
        W_size = int(W_size / 2)
    return u

# 计算给定矩阵的n次Kronecker积
def kron_power(mat: np.ndarray, n: int) -> np.ndarray:
    result = mat
    for i in range(1, n):
        result = np.kron(result, mat)
    return result

# 返回给定数组中前K大的bool指示数组
def topk_indicate(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros_like(arr, dtype=bool)
    if k >= len(arr):
        return np.ones_like(arr, dtype=bool)
    idx = np.argpartition(-arr, k)[:k]
    mask = np.zeros_like(arr, dtype=bool)
    mask[idx] = True
    return mask
