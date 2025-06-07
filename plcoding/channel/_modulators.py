from abc import ABC, abstractmethod
import numpy as np
from plcoding import base_expansion_map
from plcoding.channel._channels import _ChannelReal
from numpy.typing import NDArray
from typing import Any


__all__ = ["BPSK", "PAM"]


class _ModulatorReal(ABC):
    def __init__(self):
        pass

    def _check_1d(self, input: NDArray[Any]):
        if input.ndim != 1:
            raise ValueError(f"You must input a 1D array, but got shape {input.shape}.")


    @abstractmethod
    def modulate(self, input: NDArray[np.int64]) -> NDArray[np.float64]:
        """
        Modulate the input digits to constellation points.

        Args:
            input (NDArray[int64]): An array of digits.

        Returns:
            NDArray[float64]: An array of constellation points.

        Notes:
            The input digits do not have to be binary.
        """
        pass

    @abstractmethod
    def demodulate(self, received: NDArray[np.float64], channel: _ChannelReal) -> NDArray[np.float64]:
        """
        Perform soft demodulation on the disturbed constellation points.

        Args:
            received (NDArray[float64]): An array of disturbed constellation points.
            channel (ChannelReal): The real-number channel object through which the signal transmits.

        Returns:
            NDArray[np.float64]: The probabilities (or probability densities) that the received value originally belongs to each constellation points.

        Notes:
            The output array adds one dimension compared to the input array. For example, if the input shape is (n,), then the output shape is (q, n), where q is the number of valid constellation points.
        """
        pass


class _Universal(_ModulatorReal):
    def __init__(self, base: int, order: int):
        """
        Universal modulator for designated constellation points.

        Args:
            base (int): The base number of the input raw digits.
            order (int): The modulation order.
        """
        self.base = base
        self.order = order
        # constellation points
        self.points = np.empty(shape=(base ** order,))
        # map integers to digit arrays
        self.int2dit = base_expansion_map(base, order)
        # map digit array to integer
        self.dit2int = base ** np.arange(order).T

    def modulate(self, input):
        self._check_1d(input)
        assert input.size % self.order == 0, f"The length of the input: {input.size}, should be an integer multiple of the modulation order: {self.order}."
        indexes = np.matmul(np.reshape(input, (-1, self.order)), self.dit2int).ravel()
        return self.points[indexes.astype(np.int64)]

    def demodulate(self, received, channel):
        self._check_1d(received)
        # index likelihood
        likelies = np.empty(shape=(received.size, self.points.size))
        for p in range(self.points.size):
            test_input = np.zeros(shape=(received.size,)) + self.points[p]
            likelies[:, p] = channel._trans_prob(test_input, received)
        # digit likelihood
        probs = np.empty(shape=(self.base, self.order * received.size))
        for d in range(self.base):
            probs[d, :] = np.matmul(likelies, (self.int2dit == d).T).ravel()
        return probs / probs.sum(axis=0)


class BPSK(_Universal):
    def __init__(self):
        """
        Binary phase shift keying modulator with (0, 1) -> (+1, -1)
        """
        super().__init__(base=2, order=1)
        self.points = np.array([1, -1])


class PAM(_Universal):
    def __init__(self, M: int):
        """
        Pulse amplitude modulator with norm(-M/2, ..., M/2) -> (0, ..., M-1), where norm() is the power normalization function.
        """
        order = int(np.log2(M))
        assert 2 ** order == M, f"The modulation order should be an integer power of 2, but got M={M}."
        super().__init__(base=2, order=order)
        self.points = np.arange(M) - M / 2.0
        self.points /= np.sqrt(np.sum(self.points ** 2))
