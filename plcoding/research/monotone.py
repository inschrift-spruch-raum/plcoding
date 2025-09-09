import numpy as np
from numpy.typing import NDArray
import plcoding
from plcoding.cpp_core.monotone import ListIterator
from tqdm import trange


class JointProb():
    def __init__(self, probs: NDArray[np.float64], bases: NDArray[np.int64]):
        """
        Several basic operatios on joint probability distributions.

        Args:
            probs (NDArray[float]): The probability vector.
            bases (NDArray[int]): The base of each dimension.
        """
        self.probs = np.array(probs)
        self.bases = np.array(bases)
        self.M = self.bases.size
        self.size = self.probs.size

    def copy(self) -> "JointProb":
        """
        Generate an identical but independent copy.
        """
        new_obj = JointProb.__new__(JointProb)
        new_obj.probs = np.array(self.probs)
        new_obj.bases = np.array(self.bases)
        new_obj.M = self.M
        new_obj.size = self.size
        return new_obj

    def normalize(self):
        """
        Normalize the given probability vector, making the following operations valid.
        """
        self.probs = np.abs(self.probs) + 1e-100
        self.probs /= self.probs.sum()

    def gen_iids(self, N: int) -> NDArray[np.float64]:
        """
        Generate N independent copies.

        Args:
            N (int): Number of copies.

        Returns:
            NDArray[float]: np.array([prob, prob, ... prob])
        """
        return np.tile(self.probs, (N, 1))
    
    def marginal_by(self, var: int) -> NDArray[np.float64]:
        """
        Get the marginal distribution of the target variable.

        Args:
            var (int): The dimension of the target variable, ranging from 1 to M.
        
        Returns:
            NDArray[float]: A probability vector.
        """
        axes = tuple(i for i in range(self.M) if i != var)
        return self.probs.reshape(self.bases).sum(axis=axes).flatten()
    
    def condt_entropy(self, var: int) -> float:
        """
        Calculate the conditional entropy of the target variable. Note that this is not the entropy of marginal distributions.

        Args:
            var (int): The dimension of the target variable, ranging from 1 to M.
        
        Returns:
            float: Entropy with base-2.
        """
        base = self.bases[var]
        probs = np.array(self.probs)
        expansion = np.moveaxis(probs.reshape(self.bases), var, 0).reshape(base, -1)
        case_ps = expansion.sum(axis=0)
        expansion /= case_ps
        return ((-expansion * np.log2(expansion)).sum(axis=0) * case_ps).sum()
    
    def joint_entropy(self) -> float:
        """
        Calculate the entropy of the joint distribution.

        Returns:
            float: Entropy with base-2.
        """
        return (-self.probs * np.log2(self.probs)).sum()
    

class MCIterator():
    def __init__(self, block_len: int, jProb: JointProb):
        """
        The iterator for monotone chain polar codes.

        Args:
            block_len (int): The block length.
            jProb (JointProb): The base of each dimension.
        """
        self.N = block_len
        self.n = int(np.log2(block_len))
        assert 2 ** self.n == self.N, f"Block length should be an integer powe of 2, but got {block_len}."
        self.jProb = jProb.copy()
        self.pIter = ListIterator(block_len, 1, jProb.bases)

    def set_priors(self, priors: NDArray[np.float64]):
        """
        Set the prior distribution of the iterator.

        Args:
            priors (NDArray[float]): The given prior distribution, must be like [prob1, prob2, ...].
        """
        assert priors.shape == (self.N, self.jProb.size)
        self.pIter.set_priors(priors.ravel())

    def reset(self):
        """
        Reset the iterator.

        Notes:
            The prior distribution that has been set will not be reset.
        """
        self.pIter.reset()

    def get_prob(self, var: int, index: int) -> NDArray[np.float64]:
        """
        Get the next marginal probability distribution.

        Args:
            var (int): The variable class number of the next decoding step, ranging from 1 to M.
            index (int): The decoding index of the next decoding step, ranging from 1 to N.

        Returns:
            NDArray[float]: A nonnegative normalized vector of marginal probabilities.
        """
        self.jProb.probs = self.pIter.get_probs(var, index)
        self.jProb.normalize()
        return self.jProb.marginal_by(var)

    def set_value(self, var: int, index: int, value: int):
        """
        Set the known value for the given random variable.

        Args:
            var (int): The class number of the variable, ranging from 1 to M.
            index (int): The decoding index of the variable, ranging from 1 to N.
            value (int): The known value of the variable, ranging from 1 to q.
        """
        self.pIter.set_values(var, index, np.array([value]), np.array([0]))


class MNChain():
    def __init__(self, sups: NDArray[np.int64], M: int, N: int):
        """
        Wrapper for a monotone chain.

        Args:
            sups (NDArray[int]): The superscrips of the chain.
            M (int): The number of different variables.
            N (int): The number of independent copies of each variable.
        """
        self.sups = sups
        self.subs = np.empty_like(sups)
        self.M, self.N = M, N
        heads = np.zeros((M,), dtype=int)
        for t in range(M * N):
            self.subs[t] = heads[sups[t]]
            heads[sups[t]] += 1

    def __len__(self) -> int:
        return len(self.sups)
    
    def polarize(self, jProb: JointProb, _nsim: int=1000, _bar: bool=False) -> NDArray[np.float64]:
        """
        Monte-Carlo estimation on the polar entropies of the given joint probability distribution.
        Args:
            jProb (JointProb): The given joint probability distribution.
            _nsim (int): Simulation rounds.
            _bar (bool): Control bit for showing a progress bar.

        Returns:
            NDArray[float]: The polarized entropies [bit] along the monotone chain.
        """
        pIter = MCIterator(block_len=self.N, jProb=jProb)
        pIter.set_priors(jProb.gen_iids(self.N))
        entropies = np.empty(shape=[_nsim, self.M, self.N], dtype=float)
        my_range = trange(_nsim) if _bar else range(_nsim)
        for s in my_range:
            pIter.reset()
            for t in range(len(self)):
                var, index = self.sups[t], self.subs[t]
                condt_prob = pIter.get_prob(var=var, index=index)
                value = np.random.choice(a=len(condt_prob), p=condt_prob)
                entropies[s, var, index] = plcoding.entropy_of(condt_prob)
                pIter.set_value(var=var, index=index, value=value)
        return entropies.mean(axis=0)


class MCLIterator():
    def __init__(self, block_len: int, jProb: JointProb, list_size: int):
        """
        The iterator for monotone chain polar codes' list decoding.

        Args:
            block_len (int): The block length.
            jProb (JointProb): The base of each dimension.
            list_size (int): The size of the list.
        """
        self.N = block_len
        self.n = int(np.log2(block_len))
        self.L = list_size
        assert 2 ** self.n == self.N, f"Block length should be an integer powe of 2, but got {block_len}."
        self.jProb = jProb.copy()
        self.pIter = ListIterator(block_len, list_size, jProb.bases)
        self.perm = plcoding.bitrev_perm(block_len)
        self.llhs = np.zeros(shape=(1,))

    def set_priors(self, priors: NDArray[np.float64]):
        """
        Set the prior distribution of the iterator.

        Args:
            priors (NDArray): The given prior distribution, must be of shape (N, q^M).
        """
        assert priors.shape == (self.N, self.jProb.size)
        self.pIter.set_priors(priors.ravel())

    def reset(self):
        """
        Reset the iterator.

        Notes:
            The prior distribution that has been set will not be reset.
        """
        self.llhs = np.zeros(shape=(1,))
        self.pIter.reset()

    def _get_results(self, var: int, index: int) -> NDArray[np.float64]:
        """
        Get the step-wise decoding results and the corresponding likelihoods.

        Args:
            var (int): The variable class number of the next decoding step, ranging from 1 to M.
            index (int): The decoding index of the next decoding step, ranging from 1 to N.
        """
        nactive = self.llhs.size
        base = self.jProb.bases[var]
        results = self.pIter.get_probs(var, index).reshape([nactive, self.jProb.size])
        likelihoods = np.empty(shape=(nactive, base))
        for i in range(nactive):
            self.jProb.probs = results[i, :]
            self.jProb.normalize()
            marginal_probs = self.jProb.marginal_by(var)
            likelihoods[i, :] = np.log(marginal_probs) + self.llhs[i]
        if np.isnan(likelihoods).any():
            pass
        return likelihoods

    def explore_at(self, var: int, index: int):
        """
        Explore all possible candidates at the given decoding step, and the ones with top-L largest likelihoods are chosen to remain.

        Args:
            var (int): The variable class number of the next decoding step, ranging from 1 to M.
            index (int): The decoding index of the next decoding step, ranging from 1 to N.
        """
        # obtain the top-L values
        likelihoods = self._get_results(var, index)
        nactive_new = min(self.L, likelihoods.size)
        ranks = np.argsort(likelihoods.ravel())[::-1]
        topL_ranks = ranks[:nactive_new]
        # update the system
        self.llhs = likelihoods.ravel()[topL_ranks]
        froms, values = np.unravel_index(topL_ranks, likelihoods.shape)
        self.pIter.set_values(var, index, values, froms)

    def freeze_with(self, var: int, index: int, value: int):
        """
        Set the known value at the given decoding step.

        Args:
            var (int): The class number of the variable, ranging from 1 to M.
            index (int): The decoding index of the variable, ranging from 1 to N.
            value (int): The known value of the variable, ranging from 1 to q.
        """
        likelihoods = self._get_results(var, index)
        # update the system
        self.llhs = likelihoods[:, value]
        values = np.zeros_like(self.llhs, dtype=int) + value
        froms = np.arange(len(self.llhs))
        self.pIter.set_values(var, index, values, froms)

    def final_list(self) -> NDArray[np.int64]:
        """
        Get the list of codewords after decoding complete.

        Returns:
            NDArray[int]: A tensor of shape (L, M, N), where L is the list size, M is the number of variables, and N is the block length.
        """
        results = self.pIter.get_roots()
        list_size = results.shape[0]
        x = results.reshape(list_size, self.N, self.jProb.size).argmax(axis=-1)
        xs = np.empty(shape=(list_size, self.jProb.M, self.N))
        for i in range(list_size):
            xs[i] = self.split_on(x[i])
        return xs
    
    def final_llhs(self) -> NDArray[np.float64]:
        """
        Get the base-e likelihoods of the decoding results.

        Returns:
            NDArray[float]: A value vector of length L.
        """
        return self.llhs
    
    def transform(self, xs: NDArray[np.int64]) -> NDArray[np.int64]:
        """
        Get the transformed sequence of the multi-variate variable sequence.

        Args:
            xs (NDArray[int]): The target sequence of shape (M, N).

        Returns:
            NDArray[int]: The transformed sequence of shape (M, N).
        """
        us = np.empty_like(xs)
        for var in range(self.jProb.M):
            us[var, :] = plcoding.basic_encode(xs[var, :], base=self.jProb.bases[var], inverse=True)[self.perm]
        return us
    
    def joint_of(self, xs: NDArray[np.int64]) -> NDArray[np.int64]:
        """
        Get the joint variable sequence of a multi-variate variable sequence.

        Args:
            xs (NDArray[int]): The multi-variate variable sequence of shape (M, N).

        Returns:
            NDarray[int]: The joint variable sequence of shape (N,).
        """
        return np.ravel_multi_index(xs, self.jProb.bases)
    
    def split_on(self, x: NDArray[np.int64]) -> NDArray[np.int64]:
        """
        Split the joint variable sequence into a multi-variate variable sequence.

        Args:
            x (NDArray[int]): The joint variable sequence of shape (N,).

        Returns:
            NDarray[int]: The multi-variate variable sequence of shape (M, N).
        """
        return np.array(np.unravel_index(x, self.jProb.bases)).reshape(self.jProb.M, self.N)
