import numpy as np
import torch
from plcoding.cpp_core.source import prob_polarize
from plcoding.source.basic import padding, get_bitstream


def encode_cdf(cdf: torch.tensor, sym: torch.tensor, int_width: int=16):
    # convert cdf(torch.tensor, int16) to pmf(numpy.ndarray, double)
    cdf_np = cdf.reshape([-1, cdf.size()[-1]]).numpy()
    cdf_np[:, -1] = (1 << int_width)
    pmf = np.diff(cdf_np, axis=1) % (1 << int_width) / (1 << int_width) 
    sym = sym.numpy().ravel()
    # padding and polarize and get bitstream
    pmf, sym = padding(pmf, sym)
    pmf, sym = prob_polarize(pmf, sym)
    bitstream = get_bitstream(pmf, sym)
    return bitstream
