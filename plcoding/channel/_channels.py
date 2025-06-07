from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Any


__all__ = ["AWGN", "Erasure", "BitFlip"]


class _ChannelReal(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transmit(self, input: NDArray[Any]) -> NDArray[np.float64]:
        """
        Transmit the input array through the channel.
        """
        pass

    @abstractmethod
    def _trans_prob(self, input: NDArray[Any], output: NDArray[Any]) -> NDArray[np.float64]:
        """
        Return the transition probabilities (or probability densities) of a given input-output array pair.

        Notes:
            - These two arrays must be with the same size.
        """
        pass


class AWGN(_ChannelReal):
    def __init__(self, noise_pwr: float):
        """
        The additive white Gaussian noise channel.

        Args:
            noise_pwr (float): The variance of the Gaussian noise.

        Notes:
            - The noise_pwr parameter is not the unilateral power spectral density N0. You should implement the conversion separately if needed.
        """
        self.sigma = np.sqrt(noise_pwr)

    def transmit(self, input):
        return input + np.random.randn(*input.shape) * self.sigma

    def _trans_prob(self, input, output):
        return np.exp(-(input - output) ** 2 / (2 * self.sigma ** 2))


class Erasure(_ChannelReal):
    def __init__(self, e: float):
        """
        Erasure channel, with erased symbol set to nan.

        Args:
            e (float): The erasure probability of the channel.
        """
        self.e = e

    def transmit(self, input):
        output = input.astype(float).copy()
        output[np.random.rand(*input.shape) < self.e] = np.nan
        return output

    def _trans_prob(self, input, output):
        prob = np.zeros(shape=input.shape)
        prob[input == output] = 1 - self.e
        prob[np.isnan(output)] = self.e
        return prob


class BitFlip(_ChannelReal):
    def __init__(self, p: float):
        """
        Bit flipping channel.

        Args:
            p (float): The bit flipping probability of the channel.
        """
        self.p = p

    def transmit(self, input):
        return np.mod(input + (np.random.rand(*input.shape) < self.p), 2)

    def _trans_prob(self, input, output):
        prob = np.empty(shape=input.shape)
        prob[input == output] = 1 - self.p
        prob[input != output] = self.p
        return prob
