import numpy as np


def padding(pmf: np.ndarray, sym: np.ndarray):
    # padding the sequence length to an integer power of two
    code_len = (1 << int(np.ceil(np.log2(sym.size))))
    pmf_ = np.concatenate((pmf, np.tile(np.zeros(pmf.shape[-1]), (code_len - sym.size, 1))), axis=0)
    pmf_[sym.size:, 0] = 1
    sym_ = np.concatenate((sym, np.zeros(code_len - sym.size, dtype=int)), axis=0)
    # random shuffling to improve numerical stability
    permute = np.random.permutation(code_len)
    return pmf_[permute], sym_[permute]

def get_bitstream(pmf: np.ndarray, sym: np.ndarray):
    # using our proposed polar compression scheme to generate bitstream
    code_len = sym.size
    base = pmf.shape[-1]
    threshold = 1 - np.log(base) / (np.log(code_len) + np.log(base - 1))
    main_part = (np.max(pmf, axis=1) <= threshold)
    segment_1 = np.log2(code_len)
    segment_2 = main_part.sum() * np.log2(base)
    segment_3 = ((sym != np.argmax(pmf, axis=1)) & (~main_part)).sum() * (np.log2(code_len) + np.log2(base - 1))
    totlen = int(segment_1 + segment_2 + segment_3)
    return bytes(int(totlen / 8))
