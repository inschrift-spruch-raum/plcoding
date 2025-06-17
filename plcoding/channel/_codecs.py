import numpy as np
from plcoding.cpp_core.classics import PolarIterator
from plcoding import topk_indicate, bec_channels
from numpy.typing import NDArray


__all__ = ["PolarCodec"]


class PolarCodec():
    def __init__(self, block_len: int, code_rate: float, base: int=2):
        """
        Error correction codec based on polar coding.

        Args:
            block_len (int): The block length of the codec, must be an integer power of 2.
            code_rate (float): The coding rate, which will be further refined to: floor(code_rate * block_len) / block_len.
            base (int): The base number processed by the codec, with the default being binary.

        Notes:
            The construction of this codec is implemented using the BEC algorithm under e = 1 - R, but you can improve the construction performance by manually configuring the boolean array self.frozen.
        """
        self.N = block_len
        self.n = int(np.log2(block_len))
        assert 2 ** self.n == self.N, f"Block length should be an integer powe of 2, but got {block_len}."
        self.info_len = int(np.floor(code_rate * block_len))
        self.frzn_len = block_len - self.info_len
        self.R = self.info_len / block_len
        assert (self.R > 0) and (self.R < 1), f"Invalid coding rate R={self.R:.3f}."
        self.q = base
        self.pIter = PolarIterator(block_len, base)
        Zs = bec_channels(self.n, 1 - self.R)
        self.frozen = topk_indicate(Zs, self.frzn_len)

    def encode(self, u: NDArray[np.int64]) -> NDArray[np.int64]:
        """
        Arikan's fast encoding algorithm.

        Args:
            u (NDArray[int]): Input 1D-dimensional array.

        Returns:
            NDArray[int]: Output = Input times G_N.
        """
        return self.pIter.transform_2x(u)

    def sc_decode(self, received_probs: NDArray[np.float64]) -> NDArray[np.int64]:
        """
        Arikan's successive cancellation decoding algorithm.

        Args:
            received_probs (NDArray[np.float64]): The conditional probability distribution obtained from the channel. Must be of shape (q, N).
        """
        assert received_probs.shape[0] == self.q and received_probs.shape[1] == self.N, f"The input received_probs should be of shape ({self.q}, {self.N}), but got {received_probs.shape}."
        self.pIter.set_priors(received_probs.T)
        self.pIter.reset()
        u = np.empty((self.N,), dtype=int)
        for i in range(self.N):
            prob = self.pIter.get_prob(i)
            u[i] = 0 if self.frozen[i] else np.argmax(prob)
            self.pIter.set_value(i, u[i])
        return u
