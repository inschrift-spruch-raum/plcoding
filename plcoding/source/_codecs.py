import torch
import numpy as np
from numpy.typing import NDArray
from plcoding.cpp_core.classics import PolarIterator


__all__ = ["encode_int16_cdf", "decode_int16_cdf", "encode_pmf", "decode_pmf"]


def encode_int16_cdf(cdf: torch.Tensor, sym: torch.Tensor) -> bytes:
    """
    Interface adapted to torchac. Compresses discrete symbols into bytes using their CDFs, represented as 16-bit integers scaled by 2^16.

    Args:
        cdf (Tensor.int16): Tensor of shape (..., q+1), where each row is a cumulative distribution function over q symbols, scaled to the range [0, 2^16]. Must start at 0 and end at 2^16.
        sym (Tensor): Discrete symbol tensor, with the shape of `cdf.shape[:-1]`.

    Returns:
        bytes: Compressed bitstream.

    Notes:
        For dtype-int16, we have 2^16 = 0.
    """
    # flatten leading dimension and check shape
    cdf_np = cdf.view(-1, cdf.shape[-1]).numpy().astype(np.int64)
    sym_np = sym.view(-1).numpy().astype(np.int64)
    assert cdf_np.shape[0] == sym_np.size, f"Mismatch: {cdf_np.shape[0]} CDFs vs {sym_np.size} symbols"
    # convert to PMF
    pmf_np = np.mod(np.diff(cdf_np, axis=1), 1 << 16) / (1 << 16)
    return encode_pmf(pmf_np, sym_np)


def encode_pmf(pmf: NDArray[np.float64], sym: NDArray[np.int64]) -> bytes:
    """
    Compress discrete symbols using their associated probability mass functions (PMFs).

    Args:
        pmf (NDArray[float64]): An array of shape (N, q), each row is a PMF over q symbols.
        sym (NDArray[int64]): A 1D array of N discrete symbols to encode.

    Returns:
        bytes: The compressed bitstream.
    """
    assert sym.ndim == 1, "sym must be a 1D array"
    assert pmf.shape[0] == sym.shape[0], "PMF and symbol count mismatch"
    # pad to next power of 2
    N = sym.size
    block_len = 1 << int(np.ceil(np.log2(N)))
    # pad PMFs with uniform zero rows
    pad_len = block_len - N
    pad_pmf = np.zeros((pad_len, pmf.shape[1]), dtype=pmf.dtype)
    pad_pmf[:, 0] = 1.0  # deterministic distribution on symbol 0
    pmf_ = np.concatenate([pmf, pad_pmf], axis=0)
    sym_ = np.concatenate([sym, np.zeros(pad_len, dtype=np.int64)], axis=0)
    # deterministic, reproducible random permutation based on block length
    rng = np.random.default_rng(seed=block_len)  # seed based on block_len
    permute = rng.permutation(block_len)
    pmf_ = pmf_[permute]
    sym_ = sym_[permute]
    return _polar_compress(pmf_, sym_)


def decode_int16_cdf(cdf: torch.Tensor, data: bytes) -> torch.Tensor:
    """
    Interface adapted to torchac. Decompresses bytes into discrete symbols using their CDFs, represented as 16-bit integers scaled by 2^16.

    Args:
        cdf (Tensor.int16): Tensor of shape (..., q+1), where each row is a cumulative distribution over q symbols, scaled to the range [0, 2^16]. Must start at 0 and end at 2^16.
        data (bytes): A compressed bitstream produced by the matching encoding function `encode_int16_cdf`.

    Returns:
        Tensor: Discrete symbol tensor, with shape equal to `cdf.shape[:-1]`.

    Notes:
        For dtype-int16, we have 2^16 = 0.
    """
    # flatten leading dimension and save shape
    leading_shape = cdf.shape[:-1]
    # convert tensor.int16 CDF to ndarray.float PMF
    cdf_np = cdf.view(-1, cdf.shape[-1]).numpy().astype(np.int64)
    pmf_np = np.mod(np.diff(cdf_np, axis=-1), 1 << 16) / (1 << 16)
    # get the 1D symbol array from bytes
    sym_np = decode_pmf(pmf_np, data)
    # reshape to original shape
    sym = torch.from_numpy(sym_np.astype(np.int64)).view(leading_shape)
    return sym


def decode_pmf(pmf: NDArray[np.float64], data: bytes) -> NDArray[np.int64]:
    """
    Decompress bytes into discrete symbols using their associated probability mass functions (PMFs).

    Args:
        pmf (NDArray[float64]): An array of shape (N, q), each row is a PMF over q symbols.
        data (bytes): The output bitstream of `encode_pmf`.

    Returns:
        NDArray[int64]: A 1D array of N discrete symbols.
    """
    N, q = pmf.shape
    block_len = 1 << int(np.ceil(np.log2(N)))
    # pad PMFs to match block_len (same deterministic padding as encode)
    pad_len = block_len - N
    pad_pmf = np.zeros((pad_len, q), dtype=pmf.dtype)
    pad_pmf[:, 0] = 1.0
    pmf_ = np.concatenate([pmf, pad_pmf], axis=0)
    # decompress (note: returns sym_ with permutation)
    sym_ = _polar_decompress(pmf_, data)  # shape: (block_len,)
    # inverse permutation
    rng = np.random.default_rng(seed=block_len)
    permute_inv = np.argsort(rng.permutation(block_len))
    sym = sym_[permute_inv]
    # remove padding and return
    return sym[:N]


def _polar_compress(pmf: NDArray[np.float64], sym: NDArray[np.int64]) -> bytes:
    """
    Our proposed lossless polar compression scheme.

    Args:
        pmf (NDArray[float]): A (N, q) array of PMFs for each symbol.
        sym (NDArray[int]): A length-N array of base-q discrete symbols.

    Returns:
        bytes: The compressed bitstream.
    """
    N, q = pmf.shape
    # apply polar transform on pmfs and symbols
    pIter = PolarIterator(N, q)
    pIter.set_priors(pmf)
    pIter.reset()
    sym_pl = pIter.transform_2u(sym)
    pmf_pl = np.empty((N, q), dtype=float)
    for i in range(N):
        pmf_pl[i] = pIter.get_prob(i)
        pIter.set_value(i, sym_pl[i])
    # obtain key parameters of our proposed scheme
    sym_ml = np.argmax(pmf_pl, axis=1)
    thresh = 1 - np.log(q) / (np.log(N) + np.log(q - 1))
    hold1 = np.max(pmf_pl, axis=1) <= thresh
    hold2 = ~hold1 & (sym_ml != sym_pl)
    # data segment-1: the true values of symbols with high uncertainty, ranging from (0) to (q-1)
    data1 = _ints_2_bytes(sym_pl[hold1], q)
    head_len = int(np.ceil(np.log2(N) / 8))
    data1 = int(hold1.sum()).to_bytes(head_len) + data1
    # data segment-2: indices and differences of the remaining error bits
    indices = np.flatnonzero(hold2)                                 # indices, ranging from (0) to (N-1)
    differences = np.mod(sym_pl[indices] - sym_ml[indices], q)      # differences, ranging from (1) to (q-1)
    merged = indices + (differences - 1) * N                        # merge the index and the difference together
    merged_base = N * (q - 1)
    data2 = _ints_2_bytes(merged, merged_base)
    # a header needs to be attached only when the bit width of the merged symbol is less than one byte
    if merged_base < 256:
        data2 = int(hold2.sum()).to_bytes(head_len) + data2
    return data1 + data2


def _polar_decompress(pmf: NDArray[np.float64], data: bytes) -> NDArray[np.int64]:
    """
    Our proposed lossless polar decompression scheme.

    Args:
        pmf (NDArray[float]): A (N, q) array of PMFs for each symbol.
        data (bytes): The bitstream obtained from `_polar_compress`.

    Returns:
        NDArray[int]: The original base-q discrete symbols.
    """
    N, q = pmf.shape
    head_len = int(np.ceil(np.log2(N) / 8))
    # parse data segment-1
    part1_len = int.from_bytes(data[:head_len])
    data1_len = int(np.ceil(part1_len * np.log2(q) / 8))
    data1 = data[head_len : head_len + data1_len]
    part1 = _bytes_2_n_ints(data1, q, part1_len)
    # parse data segment-2
    merged_base = N * (q - 1)
    if merged_base < 256:
        part2_len = int.from_bytes(data[head_len + data1_len : head_len + data1_len + head_len])
        data2 = data[head_len + data1_len + head_len:]
    else:
        data2 = data[head_len + data1_len :]
        part2_len = int(np.floor(len(data2) * 8 / np.log2(merged_base)))
    # parse indices and differences from data segment-2
    merged = _bytes_2_n_ints(data2, merged_base, part2_len)
    indices = np.mod(merged, N)
    differences = (merged - indices) // N + 1
    # successive cancellation decoding
    pIter = PolarIterator(N, q)
    pIter.set_priors(pmf)
    pIter.reset()
    sym_pl = np.empty((N,), dtype=int)
    tau, psi = 0, 0
    thresh = 1 - np.log(q) / (np.log(N) + np.log(q - 1))
    for i in range(N):
        prob = pIter.get_prob(i)
        if np.max(prob) <= thresh:
            sym_pl[i] = part1[tau]
            tau += 1
        else:
            sym_pl[i] = np.argmax(prob)
            if (psi < indices.size) and (i == indices[psi]):
                sym_pl[i] = (sym_pl[i] + differences[psi]) % q
                psi += 1
        pIter.set_value(i, sym_pl[i])
    return pIter.transform_2x(sym_pl)


def _ints_2_bytes(values: NDArray[np.int64], base: int) -> bytes:
    """
    Convert a sequence of integers in a given base into a compact byte stream.

    This function treats the input sequence `values` as digits of a large integer in base `base`, and encodes that integer into the minimal number of bytes needed to represent it (rounded up to the nearest whole byte).

    Args:
        values (np.ndarray): A 1D array of integers, each in the range [0, base-1].
        base (int): The radix (base) of the integer system used to represent the digits.

    Returns:
        bytes: A byte stream representing the packed integer, with length approximately ceil(N * log2(base) / 8), where N is the length of `values`.

    Notes:
        - The output is not human-readable and is intended for compact storage or transmission.
        - The function assumes `values` is a valid integer array with all elements < base.
    """
    num = 0
    for v in values:
        num = num * base + int(v)
    n_bytes = int(np.ceil(values.shape[0] * np.log2(base) / 8))
    return num.to_bytes(n_bytes)


def _bytes_2_n_ints(data: bytes, base: int, n_ints: int) -> NDArray[np.int64]:
    """
    Recover the original sequence of integers from a compact byte stream.

    This function decodes the byte stream created by `_ints_2_bytes` using the known
    base and number of original integers.

    Args:
        data (bytes): The byte stream to decode.
        base (int): The radix (base) used when encoding the original integers.
        n_ints (int): The number of integers that were originally encoded.

    Returns:
        NDArray[int]: A 1D array of `n_ints` integers in the range [0, base-1], reconstructing the original input to `_ints_2_bytes`.

    Notes:
        - The function assumes the data was encoded using `_ints_2_bytes` with the same base and n_ints.
        - If base=2, this behaves like unpacking a binary bit stream.
    """
    num = int.from_bytes(data)
    values = np.empty(n_ints, dtype=np.int64)
    for i in range(n_ints - 1, -1, -1):
        values[i] = num % base
        num //= base
    return values
