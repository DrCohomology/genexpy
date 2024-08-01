import numpy as np
import warnings

from itertools import product
from numba import jit, njit
from numpy.random import default_rng  # to make njit happy
from typing import Any, Callable, Collection, Dict, Iterable, Literal, TypeAlias, Union

from . import rankings_utils as ru

Kernel: TypeAlias = Callable[[np.array, np.array, bool, Any], float]

RankVector: TypeAlias = np.ndarray[int]
RankByte: TypeAlias = bytes
Ranking: TypeAlias = Union[RankVector, RankByte]


# ---- Mallows kernel


@njit
def _mallows_rv(r1: RankVector, r2: RankVector, nu: Union[float, Literal["auto"]] = "auto") -> float:
    n = len(r1)
    out = 0
    for i in range(n):
        for j in range(i):
            out += np.abs(np.sign(r1[i] - r1[j]) - np.sign(r2[i] - r2[j]))
    return np.exp(- nu * out)


def _mallows_bytes(b1: RankByte, b2: RankByte, nu: Union[float, Literal["auto"]] = "auto"):
    i1 = np.frombuffer(b1, dtype=np.int8)
    i2 = np.frombuffer(b2, dtype=np.int8)
    return np.exp(- nu * np.sum(np.abs(i1 - i2)))


def mallows_kernel(x1: Ranking, x2: Ranking, use_rv: bool = True, nu: Union[float, Literal["auto"]] = "auto") -> float:
    """
    Computes the Mallows kernel between two rankings, which is based on the difference in their rankings adjusted by a 
    kernel bandwidth parameter nu.
    
    Parameters:
    - x1 (Ranking): The first ranking as a RankVector or RankByte.
    - x2 (Ranking): The second ranking as a RankVector or RankByte.
    - nu (float, 'auto'): The decay parameter for the kernel. If 'auto', it adjusts based on the length of the rankings.
    - use_rv (bool): Determines whether to use the rank vector or byte representation for the calculation.
    
    Returns:
    - float: The computed Mallows kernel value.
    
    Raises:
    - ValueError: If the rankings do not have the same number of alternatives.
    """
    if len(x1) != len(x2):
        raise ValueError("The rankings must have the same number of alternatives.")
    
    if isinstance(nu, float) and nu <= 0:
        raise ValueError("nu must be a positive number when specified.")

    if nu == "auto":
        n = len(x1) if use_rv else np.sqrt(len(x1))
        nu = 2 / (n*(n-1))
    if use_rv:
        return _mallows_rv(x1, x2, nu=nu)
    else:
        return _mallows_bytes(x1, x2, nu=nu)


# ---- Jaccard kernel


def _jaccard_rv(r1: RankVector, r2: RankVector, k: int = 1) -> float:
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


def jaccard_kernel(x1: Ranking, x2: Ranking, use_rv: bool = True, k: int = 1) -> float:
    """
    Computes the Jaccard kernel between two rankings by considering the top k tiers of the rankings. This kernel 
    measures the similarity based on the intersection over union of the rankings within the top k tiers.
    
    Parameters:
    - x1 (Ranking): The first ranking as a RankVector or RankByte.
    - x2 (Ranking): The second ranking as a RankVector or RankByte.
    - k (int): The number of top tiers to consider for the similarity calculation.
    - use_rv (bool): Determines whether to use the rank vector or byte representation for the calculation.
    
    Returns:
    - float: The computed Jaccard similarity score.
    
    Raises:
    - ValueError: If the rankings do not have the same number of alternatives.
    """
    if len(x1) != len(x2):
        raise ValueError("The rankings must have the same number of alternatives.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if k > len(x1):
        raise ValueError("k cannot exceed the number of alternatives in the rankings.") #TODO: Is this legit?
    if use_rv:
        return _jaccard_rv(x1, x2, k=k)
    else:
        return _jaccard_bytes(x1, x2, k=k)


# ---- Borda kernel


@njit
def _borda_rv(r1: RankVector, r2: RankVector, idx: int = 0, nu: Union[float, Literal["auto"]] = "auto") -> float:
    return np.exp(- nu * np.abs(np.sum(r1 >= r1[idx]) - np.sum(r2 >= r2[idx])))


def _borda_bytes(b1: RankByte, b2: RankByte, idx: int = 0, nu: Union[float, Literal["auto"]] = "auto") -> float:
    raise NotImplementedError

# TODO: let hte function accept the name of an alternative instead of just indices
def borda_kernel(x1: Ranking, x2: Ranking, use_rv: bool = True, idx: int = 0, alternative=None,
                 nu: Union[float, Literal["auto"]] = "auto") -> float:
    """
    Computes a kernel based on the Borda count for a specific alternative indexed by 'idx'. This kernel considers the 
    rescaled difference of the Borda counts at a particular position and is adjusted by a kernel bandwidth 'nu'.
    
    Parameters:
    - x1 (Ranking): The first ranking, either as a RankVector or RankByte.
    - x2 (Ranking): The second ranking, either as a RankVector or RankByte.
    - idx (int): Index of the alternative under consideration within the ranking.
    - nu (float, 'auto'): The kernel bandwidth, adjusted automatically to the inverse of the number of alternatives squared if 'auto'.
    - use_rv (bool): If True, uses rank vector representation; otherwise expects a byte representation.
    
    Returns:
    - float: The computed kernel value.
    
    Raises:
    - ValueError: If the rankings do not have the same number of alternatives.
    """
    if len(x1) != len(x2):
        raise ValueError("The rankings must have the same number of alternatives.")
    if not isinstance(idx, int) or idx < 0 or idx >= len(x1):
        raise ValueError("idx must be a non-negative integer within the bounds of the rankings.")
    if isinstance(nu, str) and nu != "auto":
        raise ValueError("When a string, nu must be 'auto'.")
    if isinstance(nu, float) and nu <= 0:
        raise ValueError("nu must be a positive number when specified as a float.")
    if nu == "auto":
        if use_rv:
            nu = 1 / len(x1)
        else:
            nu = 1 / np.sqrt(len(x1))

    if use_rv:
        return _borda_rv(x1, x2, idx=idx, nu=nu)
    else:
        raise _borda_bytes(x1, x2, idx=idx, nu=nu)


# ---- Other kernels


def trivial_kernel(x: Ranking, y: Ranking, use_rv: bool = True, **kwargs) -> float:
    return float(int(np.all(x == y)))


def degenerate_kernel(x: Ranking, y: Ranking, use_rv: bool = True, **kwargs) -> float:
    return 1.0


# ---- Gram matrix


def gram_matrix(sample1: ru.SampleAM, sample2: ru.SampleAM, use_rv: bool = True,
                kernel: Kernel = trivial_kernel, **kernelargs) -> np.ndarray[float]:
    """
    Computes the Gram matrix between two samples of rankings, where each entry in the matrix represents the kernel 
    similarity between the rankings from each sample.
    
    Parameters:
    - sample1 (SampleAM): The first sample of rankings.
    - sample2 (SampleAM): The second sample of rankings.
    - use_rv (bool): If True, converts the rankings to rank function matrix format before processing.
    - kernel (Kernel): The kernel function to use for computing similarities.
    - **kernelargs: Additional keyword arguments for the kernel function.
    
    Returns:
    - np.ndarray[float]: A matrix of kernel similarities.
    """

    if use_rv:
        sample1 = sample1.to_rank_function_matrix().T  # rows: voters, cols: alternatives
        sample2 = sample2.to_rank_function_matrix().T  #

    out = np.zeros((len(sample1), len(sample2)))
    for (i1, x1), (i2, x2) in product(enumerate(sample1), enumerate(sample2)):
        out[i1, i2] = kernel(x1, x2, use_rv, **kernelargs)

    return out


def square_gram_matrix(sample: ru.SampleAM, use_rv: bool = True,
                       kernel: Kernel = trivial_kernel, **kernelargs) -> np.ndarray[float]:
    """
    Computes a symmetric square Gram matrix for a given sample of rankings. The matrix is calculated using a specified 
    kernel function, and each element [i, j] represents the kernel similarity between the i-th and j-th elements of the 
    sample.

    Parameters:
    - sample (SampleAM): The sample of rankings for which the Gram matrix is to be computed. This can be a collection 
      of rank vectors or adjacency matrices.
    - use_rv (bool): If True, converts the rankings to a rank function matrix before computation, which changes the 
      data structure to rows representing voters and columns representing alternatives.
    - kernel (Kernel): The kernel function used to compute the similarity between two rankings.
    - **kernelargs: Arbitrary keyword arguments for the kernel function to handle specific kernel configurations.
    
    Returns:
    - np.ndarray[float]: A symmetric square Gram matrix of kernel similarities.

    Notes:
    - The function first computes the lower triangle of the matrix, fills the diagonal with self-similarity values, 
      and then mirrors the lower triangle to the upper to complete the symmetric matrix.
    """

    if use_rv:
        sample = sample.to_rank_function_matrix().T  # rows: voters, cols: alternatives

    lt = np.zeros((len(sample), len(sample)))
    for (i1, x1) in enumerate(sample):
        for i2, x2 in enumerate(sample):
            if i1 <= i2:
                break
            lt[i1, i2] = kernel(x1, x2, use_rv, **kernelargs)
    d = np.diag([kernel(x, x, use_rv, **kernelargs) for x in sample])

    return lt + d + lt.T


# ---- Variance


def var(sample: ru.SampleAM, use_rv: bool = True, kernel: Kernel = trivial_kernel, **kernelargs) -> float:
    """
    Computes the sample variance of a distribution of rankings within a Reproducing Kernel Hilbert Space (RKHS). The 
    variance is derived from the Gram matrix of the sample, which is computed using a specified kernel function.

    Parameters:
    - sample (SampleAM): The sample of rankings from which to compute the variance.
    - use_rv (bool): If True, the rankings are converted using a rank function matrix representation before computation.
    - kernel (Kernel): The kernel function used for computing the Gram matrix.
    - **kernelargs: Additional keyword arguments for the kernel function.
    
    Returns:
    - float: The computed variance of the rankings within the RKHS.
    """
    nv = len(sample)
    kxx = square_gram_matrix(sample=sample, use_rv=use_rv, kernel=kernel, **kernelargs)
    return 1 / (nv - 1) * np.sum(np.array([kxx[i, i] for i in range(nv)])) - 1 / (nv * (nv - 1)) * np.sum(kxx)

