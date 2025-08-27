import numpy as np
from numpy.typing import NDArray
from typing import Any


def entropy_of(prob: NDArray[np.float64], base: int = 2) -> float:
    """
    Calculate the base-entropy of the given probability distribution.
    """
    prob = prob[np.nonzero(prob)]
    return np.sum(-prob * np.log(prob)) / np.log(base)


def h2_of(p: float) -> float:
    """
    Calculate the base-2 entropy of probability p.
    """
    if p <= 0 or p >= 1:
        return 0.0
    return entropy_of(np.array([p, 1 - p]), base=2)


def p_of(h2: float, tol: float = 1e-10) -> NDArray[np.float64]:
    """
    Calculate the binary probability distribution of entropy = h2.
    """
    low, high = 0.0, 0.5
    while high - low > tol:
        mid = (low + high) / 2
        h = h2_of(mid)
        if h < h2:
            low = mid
        else:
            high = mid
    p = (low + high) / 2
    return np.array((p, 1 - p))


def bitrev_perm(len: int) -> NDArray[np.int64]:
    """
    Generate the bit reversal permutation.
    """
    n = int(np.log2(len))
    assert 2 ** n == len, "len must be a power of 2"
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(len)])


def bec_channels(level: int, e: float = 0.5, _log: bool = False) -> NDArray[np.float64]:
    """
    Calculate the Bhattacharyya parameters of polarized erasure channels.

    Args:
        level (int): Number of polarization level.
        e (float): Erasure probability of the original channel.
        _log (bool): Enable high-precision calculations based on logarithmic operations.
    """
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


def basic_encode(input: NDArray[np.int64], base: int=2, inverse: bool=False) -> NDArray[np.int64]:
    """
    The basic low-complexity recursive encoding of polar codes.

    Args:
        input (NDArray[int]): Symbol sequence before polar transform.
        base (int): The alphabet size of the sequence.
        inverse (bool): Enable non-binary inverse transfrom.
    """
    N, W_size = len(input), len(input)
    u, x = np.copy(input), np.empty_like(input)
    while W_size > 1:
        for j in range(0, N, 2):
            x[j] = ((u[j] - u[j + 1] + base) if inverse else (u[j] + u[j + 1])) % base
            x[j + 1] = u[j + 1] % base
        for j in range(0, N, W_size):
            for k in range(0, W_size, 2):
                u[j + int(k / 2)] = x[j + k]
                u[j + int((W_size + k) / 2)] = x[j + k + 1]
        W_size = int(W_size / 2)
    return u


def kron_power(mat: NDArray[np.int64], n: int) -> NDArray[np.int64]:
    result = mat
    for _ in range(1, n):
        result = np.kron(result, mat)
    return result


def topk_indicate(arr: NDArray[Any], k: int) -> NDArray[np.bool_]:
    """
    Return the first k-largest bool indicator array of the input array.
    """
    if k <= 0:
        return np.zeros_like(arr, dtype=bool)
    if k >= len(arr):
        return np.ones_like(arr, dtype=bool)
    idx = np.argpartition(-arr, k)[:k]
    mask = np.zeros_like(arr, dtype=bool)
    mask[idx] = True
    return mask


def base_expansion_map(q: int, n: int) -> NDArray[np.int64]:
    """
    Returns a (q, q^n) mapping matrix where each column is the length-n base-q expansion of the column index.

    Args:
        q (int): The base for expansion (e.g., 2 for binary).
        n (int): The number of digits.

    Returns:
        NDArray[int64]: A mapping matrix of shape (q, q^n).
    """
    indices = np.arange(q ** n)
    map_matrix = np.zeros(shape=(n, q ** n), dtype=int)
    for i in range(n):
        divisor = q ** i
        map_matrix[i] = (indices // divisor) % q
