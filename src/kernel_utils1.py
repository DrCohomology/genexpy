import numpy as np
import warnings

from itertools import product
from numba import jit, njit
from numpy.random import default_rng  # to make njit happy
from typing import Any, Callable, Collection, Dict, Iterable, Literal, TypeAlias, Union

import src.rankings_utils as ru

Kernel: TypeAlias = Callable[[np.array, np.array, bool, Any], float]
RankFunction: TypeAlias = np.ndarray[int]
RankByte: TypeAlias = bytes
Ranking: TypeAlias = Union[RankFunction, RankByte]


# ---- Mallows kernel


@njit
def _mallows_rf(r1: RankFunction, r2: RankFunction, nu: float = "auto") -> float:
    n = len(r1)
    out = 0
    for i in range(n):
        for j in range(n):
            out += np.abs(np.sign(r1[i] - r1[j]) - np.sign(r2[i] - r2[j]))
    return np.exp(- nu * out)


def _mallows_bytes(b1: RankByte, b2: RankByte, nu: Union[float, Literal["auto"]] = "auto"):
    # if nu == "auto":
    #     nu = 2 / (np.sqrt(len(b1)) * (np.sqrt(len(b1)) - 1))
    i1 = np.frombuffer(b1, dtype=np.int8)
    i2 = np.frombuffer(b2, dtype=np.int8)
    return np.exp(- nu * np.sum(np.abs(i1 - i2)))


def mallows_kernel(x1: Ranking, x2: Ranking, use_rf: bool = True, nu: Union[float, Literal["auto"]] = "auto") -> float:
    """
    Mallows kernel adapted for ties.
    :param x1: first ranking
    :type x1: Ranking
    :param x2: second ranking
    :type x2: Ranking
    :param nu: kernel bandwidth. If 'auto', it's the number of
    :type nu: float or 'auto'
    :param use_rf: if True, use rank function
    :type use_rf: bool
    :return: the Mallows kernel adapted for ties
    :rtype: float
    """
    if len(x1) != len(x2):
        raise ValueError("The rankings have different number of alternatives.")
    if nu == "auto":
        if use_rf:
            # nu = 2 / (len(x1) * (len(x1) - 1))
            nu = 1 / len(x1)**2
        else:
            # nu = 2 / (np.sqrt(len(x1)) * (np.sqrt(len(x1)) - 1))
            nu = 1 / len(x1)
    if use_rf:
        return _mallows_rf(x1, x2, nu=nu)
    else:
        return _mallows_bytes(x1, x2, nu=nu)


# ---- Jaccard kernel


def _jaccard_rf(r1: RankFunction, r2: RankFunction, k: int = 1) -> float:
    """
    Supports tied rankings as columns of the output from SampleAM.to_rank_function_matrix().
    """
    topk1 = np.where(r1 < k)[0]
    topk2 = np.where(r2 < k)[0]
    # return len(np.intersect1d(topk1, topk2)) / len(np.union1d(topk1, topk2))
    return len(set(topk1).intersection(set(topk2))) / len(set(topk1).union(set(topk2)))


def _jaccard_bytes(b1: RankByte, b2: RankByte, k: int = 1):
    """
    Implementation is specific for AdjacencyMatrix objects, version of 25.01.2024.
    """
    na = np.sqrt(len(b1))
    if na != int(na):
        raise ValueError("Wrong length of bytestring or format.")
    na = int(na)

    topk1 = np.where(np.frombuffer(b1, dtype=np.int8).reshape((na, na)).sum(axis=1) > na - k)[0]
    topk2 = np.where(np.frombuffer(b2, dtype=np.int8).reshape((na, na)).sum(axis=1) > na - k)[0]

    # return len(np.intersect1d(topk1, topk2)) / len(np.union1d(topk1, topk2))
    return len(set(topk1).intersection(set(topk2))) / len(set(topk1).union(set(topk2)))


def jaccard_kernel(x1: Ranking, x2: Ranking, use_rf: bool = True, k: int = 1) -> float:
    """
    Jaccard similarity (intersection over union) of the top k tiers of x1 and x2.
    :param x1: first ranking
    :type x1: Ranking
    :param x2: second ranking
    :type x2: Ranking
    :param k: top tiers considered
    :type k: int
    :param use_rf: if True, use rank function
    :type use_rf: bool
    :return: the Jaccard kernel adapted for ties
    :rtype: float
    """
    if len(x1) != len(x2):
        raise ValueError("The rankings have different number of alternatives.")
    if use_rf:
        return _jaccard_rf(x1, x2, k=k)
    else:
        return _jaccard_bytes(x1, x2, k=k)


# ---- Borda kernel


@njit
def _borda_rf(r1: RankFunction, r2: RankFunction, idx: int = 0, nu: Union[float, Literal["auto"]] = "auto") -> float:
    return np.exp(- nu * np.abs(np.sum(r1 >= r1[idx]) - np.sum(r2 >= r2[idx])))


def _borda_bytes(b1: RankByte, b2: RankByte, idx: int = 0, nu: Union[float, Literal["auto"]] = "auto") -> float:
    raise NotImplementedError


def borda_kernel(x1: Ranking, x2: Ranking, use_rf: bool = True, idx: int = 0,
                 nu: Union[float, Literal["auto"]] = "auto") -> float:
    """
    Rescaled difference of the Borda counts for the alternative at position idx.
    :param nu: kernel bandwidth. auto is the number of alternatives
    :type nu:
    :param x1: first ranking
    :type x1: Ranking
    :param x2: second ranking
    :type x2: Ranking
    :param idx: Index of the alternative under consideration
    :type idx: int
    :param use_rf: if True, use rank function
    :type use_rf: bool
    :return: the Jaccard kernel adapted for ties
    :rtype: float
    """
    if len(x1) != len(x2):
        raise ValueError("The rankings have different number of alternatives.")
    if nu == "auto":
        if use_rf:
            nu = 1 / len(x1)
        else:
            nu = 1 / np.sqrt(len(x1))

    if use_rf:
        return _borda_rf(x1, x2, idx=idx, nu=nu)
    else:
        raise _borda_bytes(x1, x2, idx=idx, nu=nu)


# ---- Other kernels


def trivial_kernel(x: Ranking, y: Ranking, use_rf: bool = True, **kwargs) -> float:
    return float(int(np.all(x == y)))


def degenerate_kernel(x: Ranking, y: Ranking, use_rf: bool = True, **kwargs) -> float:
    return 1.0


# ---- Gram matrix


def gram_matrix(sample1: ru.SampleAM, sample2: ru.SampleAM, use_rf: bool = True,
                kernel: Kernel = trivial_kernel, **kernelargs) -> np.ndarray[float]:
    """
    Gram matrix of the two samples.
    out[i, j] = kernel(sample1[i], sample2[j]).
    """

    if use_rf:
        sample1 = sample1.to_rank_function_matrix().T  # rows: voters, cols: alternatives
        sample2 = sample2.to_rank_function_matrix().T  #

    out = np.zeros((len(sample1), len(sample2)))
    for (i1, x1), (i2, x2) in product(enumerate(sample1), enumerate(sample2)):
        out[i1, i2] = kernel(x1, x2, use_rf, **kernelargs)

    return out


def square_gram_matrix(sample: ru.SampleAM, use_rf: bool = True,
                       kernel: Kernel = trivial_kernel, **kernelargs) -> np.ndarray[float]:
    """
    Gram matrix of the two samples.
    out[i, j] = kernel(sample1[i], sample2[j]).
    """

    if use_rf:
        sample = sample.to_rank_function_matrix().T  # rows: voters, cols: alternatives

    lt = np.zeros((len(sample), len(sample)))
    for (i1, x1) in enumerate(sample):
        for i2, x2 in enumerate(sample):
            if i1 <= i2:
                break
            lt[i1, i2] = kernel(x1, x2, use_rf, **kernelargs)
    d = np.diag([kernel(x, x, use_rf, **kernelargs) for x in sample])

    return lt + d + lt.T


# ---- Variance


def var(sample: ru.SampleAM, use_rf: bool = True, kernel: Kernel = trivial_kernel, **kernelargs) -> float:
    """
    Sample variance of a distribution of adjacency matrices computed in a RKHS.
    """
    nv = len(sample)
    kxx = square_gram_matrix(sample=sample, use_rf=use_rf, kernel=kernel, **kernelargs)
    return 1 / (nv - 1) * np.sum(np.array([kxx[i, i] for i in range(nv)])) - 1 / (nv * (nv - 1)) * np.sum(kxx)


# ---- MMD


def mmdb(sample1: ru.SampleAM, sample2: ru.SampleAM, use_rf: bool = True,
         kernel: Kernel = trivial_kernel, **kernelargs) -> float:
    """
    Compute the biased estimator of MMD between two distributions.
    rfx and rfy must be matrices of ranks.
    kernel must take rank functions as input, without optional arguments.
    """
    kxx = square_gram_matrix(sample1, use_rf=use_rf, kernel=kernel, **kernelargs)
    kxy = gram_matrix(sample1, sample2, use_rf=use_rf, kernel=kernel, **kernelargs)
    kyy = square_gram_matrix(sample2, use_rf=use_rf, kernel=kernel, **kernelargs)

    return np.sqrt(np.abs(np.mean(kxx) + np.mean(kyy) - 2 * np.mean(kxy)))


def subsample_mmd_distribution(sample: ru.SampleAM, subsample_size: int,
                               seed: int = 42, rep: int = 1000, use_rf: bool = True, use_key: bool = False,
                               replace: bool = False, disjoint: bool = True,
                               kernel: Kernel = trivial_kernel, **kernelargs) -> np.ndarray[float]:
    """

    :param replace:
    :type replace:
    :param disjoint:
    :type disjoint:
    :param sample: Sample to compute MMD from
    :type sample:
    :param subsample_size: size of subsamples
    :type subsample_size:
    :param seed: to np.random.default_rng()
    :type seed:
    :param rep: number of repetitions
    :type rep:
    :param use_rf: if True, the kernel must support njit and rf (rank function)
    :type use_rf:
    :param kernel:
    :type kernel:
    :param use_key: if True, subsample using sample.key (instead of sampling from sample.index). subsample_size must be adjusted accordingly.
    :type use_key:
    :param kernelargs:
    :type kernelargs:
    :return:
    :rtype:
    """

    out = np.empty(rep)
    for ir in range(rep):
        sub1, sub2 = sample.get_subsamples_pair(subsample_size=subsample_size, seed=seed + 2 * ir, use_key=use_key,
                                                replace=replace, disjoint=disjoint, use_rf=False)  # so that it returns two SampleAM
        out[ir] = mmdb(sub1, sub2, use_rf=use_rf, kernel=kernel, **kernelargs)

    return out




# ---- Classes are too slow :(
# class Kernel:
#     """
#     Degenerate kernel: everything is the same.
#     """
#
#     def __init__(self, kernelargs: Dict, use_njit: bool = False):
#         self.use_njit = use_njit
#         self.kernelargs = kernelargs
#
#     @staticmethod
#     def _bytes(b1: bytes, b2: bytes, **kernelargs) -> float:
#         return 1.0
#
#     @staticmethod
#     @njit
#     def _rf(r1: np.ndarray[int], r2: np.ndarray[int], **kernelargs) -> float:
#         return 1.0
#
#     def __call__(self, x1: Union[np.ndarray[int], bytes], x2: Union[np.ndarray[int], bytes]):
#         if len(x1) != len(x2):
#             raise ValueError("The input rankings have different lengths.")
#         return self._rf(x1, x2, **self.kernelargs) if self.use_njit else self._bytes(x1, x2, **self.kernelargs)
#
#
# class MallowsKernel(Kernel):
#     """
#     The Mallows kernel adapted for ties.
#     """
#
#     def __init__(self, nu: float = 1, **superargs):
#         super().__init__(kernelargs={"nu": nu}, **superargs)
#
#     @staticmethod
#     @njit
#     def _rf(r1: np.ndarray[int], r2: np.ndarray[int], nu: float = 1) -> float:
#         n = len(r1)
#         out = 0
#         for i in range(n):
#             for j in range(n):
#                 out += np.abs(np.sign(r1[i] - r1[j]) - np.sign(r2[i] - r2[j]))
#         return np.exp(- nu * out / 2)
#
#     @staticmethod
#     def _bytes(b1: bytes, b2: bytes, nu: float = 1):
#         i1 = np.frombuffer(b1, dtype=np.int8)
#         i2 = np.frombuffer(b2, dtype=np.int8)
#         return np.exp(- nu * np.sum(np.abs(i1 - i2)))
#
#
# class JaccardKernel(Kernel):
#     """
#     The Jaccard similarity (intersection over union) between the top k tiers.
#     """
#     def __init__(self, k: float = 1, **superargs):
#         super().__init__(kernelargs={"k": k}, **superargs)
#
#     @staticmethod
#     @njit
#     def _rf(r1: np.ndarray[int], r2: np.ndarray[int], k: int = 1) -> float:
#         """
#         Supports tied rankings as columns of the output from SampleAM.to_rank_function_matrix().
#         """
#         topk1 = np.where(r1 < k)[0]
#         topk2 = np.where(r2 < k)[0]
#         # return len(np.intersect1d(topk1, topk2)) / len(np.union1d(topk1, topk2))
#         return len(set(topk1).intersection(set(topk2))) / len(set(topk1).union(set(topk2)))
#
#     @staticmethod
#     def _bytes(b1: bytes, b2: bytes, k: int = 1):
#         """
#         Implementation is specific for AdjacencyMatrix objects, version of 25.01.2024.
#         """
#         na = np.sqrt(len(b1))
#         if na != int(na):
#             raise ValueError("Wrong length of bytestring or format.")
#         na = int(na)
#
#         topk1 = np.where(np.frombuffer(b1, dtype=np.int8).reshape((na, na)).sum(axis=1) > na - k)[0]
#         topk2 = np.where(np.frombuffer(b2, dtype=np.int8).reshape((na, na)).sum(axis=1) > na - k)[0]
#
#         return len(set(topk1).intersection(set(topk2))) / len(set(topk1).union(set(topk2)))
