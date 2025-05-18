import numpy as np
from .source_core import compress


def pre_process(cdf, sym, int_width=16):
    # Convert torch.tensor to numpy.ndarray and cdf to pmf
    cdf_len = cdf.size()[-1]
    cdf = cdf.reshape([-1, cdf_len]).numpy() % (1 << int_width)
    cdf[:, -1] = (1 << int_width)
    pmf = np.diff(cdf, axis=1) / (1 << int_width)
    sym = sym.numpy().ravel()
    return pmf, sym

def padding(pmf, sym):
    # Padding the sequence length to an integer power of two
    code_len = (1 << int(np.ceil(np.log2(sym.size))))
    pmf_ = np.concatenate((pmf, np.tile(np.zeros(pmf.shape[-1]), (code_len - sym.size, 1))), axis=0)
    pmf_[sym.size:, 0] = 1
    sym_ = np.concatenate((sym, np.zeros(code_len - sym.size, dtype=int)), axis=0)
    # Random shuffling to improve numerical stability
    permute = np.random.permutation(code_len)
    return pmf_[permute], sym_[permute]

def polar_compress(cdf, sym):
    pmf, sym = pre_process(cdf, sym)
    pmf, sym = padding(pmf, sym)
    return compress(pmf, sym)
