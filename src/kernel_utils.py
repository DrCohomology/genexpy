import numpy as np
import warnings

from itertools import product
from numba import jit, njit
from numpy.random import default_rng  # to make njit happy
from typing import Any, Callable, Collection, Iterable, TypeAlias

import src.rankings_utils as ru

Kernel: TypeAlias = Callable[[np.array, np.array, Any], float]


@njit
def mallows_kernel(b1: bytes, b2: bytes, nu: float = 1) -> float:
    """
    b1 and b2 are the np.tobytes() representation of two AdjaecncyMatrix objects.
    Adapted for ties.
    """
    i1 = np.frombuffer(b1, dtype=np.int8)
    i2 = np.frombuffer(b2, dtype=np.int8)
    return np.exp(- nu * np.sum(np.abs(i1 - i2)) / 2)


# @njit
# def mallows_kernel_rf(rf1: np.ndarray[int], rf2: np.ndarray[int], nu: float = 1) -> float:
#     """
#     r1 and r2 are numpy arrays representing rank functions.
#     The function returns the Mallows kernel between these two rank functions.
#     """
#
#     n = len(rf1)
#     discordant_pairs = 0
#
#     for i in range(n):
#         for j in range(n):
#             if i < j:
#                 if (rf1[i] < rf1[j] and rf2[i] > rf2[j]) or (rf1[i] > rf1[j] and rf2[i] < rf2[j]):
#                     discordant_pairs += 1
#
#     return np.exp(- nu * discordant_pairs)


@njit
def mallows_kernel_rf(rf1: np.ndarray[int], rf2: np.ndarray[int], nu: float = 1) -> float:
    """
    r1 and r2 are numpy arrays representing rank functions.
    Return the Mallows kernel between these two rank functions, adapted for ties.
    """
    if len(rf1) != len(rf2):
        raise ValueError("The rank functions should have the same length.")

    n = len(rf1)

    out = 0
    for i in range(n):
        for j in range(n):
            out += np.abs(np.sign(rf1[i] - rf1[j]) - np.sign(rf2[i] - rf2[j]))

    return np.exp(- nu * out / 2)


@njit
def borda_kernel_rf(rf1: np.ndarray[int], rf2: np.ndarray[int], idx: int = 0) -> float:
    """
    The kernel is the rescaled difference of the Borda count for the alternative at position idx.
    """
    if len(rf1) != len(rf2):
        raise ValueError("The rank functions should have the same length.")
    return 1 - np.abs(np.sum(rf1 <= rf1[idx]) - np.sum(rf2 <= rf2[idx])) / len(rf1)



@njit
def trivial_kernel(x: np.ndarray[Any], y: np.ndarray[Any], **kwargs) -> float:
    return float(int(np.all(x == y)))


@njit
def degenerate_kernel(x: np.ndarray[Any], y: np.ndarray[Any], **kwargs) -> float:
    return 1


def jaccard_kernel(b1: bytes, b2: bytes, k: int = 1) -> float:
    """
    Jaccard similarity (intersection over union) of the top k alternatives.
    Implementation is specific for AdjacencyMatrix objects, version of 25.01.2024.
    """
    if len(b1) != len(b2):
        raise ValueError("Wrong length of the bytestring.")

    na = np.sqrt(len(b1))
    if na != int(na):
        raise ValueError("Wrong length of bytestring or format.")
    na = int(na)

    topk1 = np.where(np.frombuffer(b1, dtype=np.int8).reshape((na, na)).sum(axis=1) > na - k)[0]
    topk2 = np.where(np.frombuffer(b2, dtype=np.int8).reshape((na, na)).sum(axis=1) > na - k)[0]

    return len(set(topk1).intersection(set(topk2))) / len(set(topk1).union(set(topk2)))


@njit
def jaccard_kernel_rf(r1: np.array, r2: np.array, k: int = 1) -> float:
    """
    Supports tied rankings as columns of the output from SampleAM.to_rank_function_matrix().
    Return the Jaccard similarity between the union of the top k tiers.
    """
    topk1 = np.where(r1 < k)[0]
    topk2 = np.where(r2 < k)[0]
    return len(set(topk1).intersection(set(topk2))) / len(set(topk1).union(set(topk2)))
    # return len(np.intersect1d(topk1, topk2)) / len(np.union1d(topk1, topk2))


def gram_matrix_rf(rfx: np.ndarray, rfy: np.ndarray, kernel: Kernel = trivial_kernel, **kernelargs) -> np.ndarray:
    if rfx.shape[0] != rfy.shape[0]:
        raise ValueError("Wrong number of alternatives.")

    print(kernelargs)

    out = np.zeros((rfx.shape[1], rfy.shape[1]))
    for i in range(rfx.shape[1]):
        for j in range(rfy.shape[1]):
            out[i, j] = kernel(rfx[:, i], rfy[:, j], **kernelargs)
    return out


@njit
def gram_matrix_rf_njit(rfx: np.ndarray, rfy: np.ndarray, kernel: Kernel, *kernelargs) -> np.ndarray:
    """
    Kernel must work with rank functions and without optional arguments.
    """

    if rfx.shape[0] != rfy.shape[0]:
        raise ValueError("Wrong number of alternatives.")

    out = np.zeros((rfx.shape[1], rfy.shape[1]))
    for i in range(rfx.shape[1]):
        for j in range(rfy.shape[1]):
            out[i, j] = kernel(rfx[:, i], rfy[:, j], *kernelargs)
    return out


def gram_matrix(lx: np.ndarray, ly: np.ndarray, kernel: Kernel = trivial_kernel, **kernelargs) -> np.array:
    """
    Compute the [name?] matrix of two samples lx and ly, i.e., M[i, j] = kernel(lx[i], ly[j]).
    Use when lx and ly are bytestrings.
    """
    return np.array([[kernel(x, y, **kernelargs) for y in ly] for x in lx])


def square_gram_matrix(lx: Collection, kernel: Kernel = trivial_kernel, **kernelargs) -> np.array:
    """
    Compute Gram matrix from a single sample
    """

    lt = np.zeros((len(lx), len(lx)))
    for ix, x in enumerate(lx):
        for iy, y in enumerate(lx):
            if iy >= ix:
                break
            lt[ix, iy] = kernel(x, y, **kernelargs)

    d = np.diag([kernel(x, x, **kernelargs) for x in lx])

    return lt + d + lt.T


def mmdb(lx: np.ndarray, ly: np.ndarray, kernel: Kernel, **kernelargs) -> float:
    """
    Compute the biased estimator of MMD between two samples of distributions.
    """
    kxx = gram_matrix(lx, lx, kernel, **kernelargs)
    kxy = gram_matrix(lx, ly, kernel, **kernelargs)
    kyy = gram_matrix(ly, ly, kernel, **kernelargs)

    # m = len(lx)
    # n = len(ly)

    a = np.mean(kxx)
    b = np.mean(kyy)
    c = np.mean(kxy)

    if np.isnan(np.sqrt(a + b - 2 * c)):
        warnings.warn(f"Biased MMD^2 is negative: {a+b-2*c}. Computation continues with the absolute value.")

    # a = 1 / (m * (m - 1)) * np.sum(kxx - np.diag(np.diagonal(kxx)))
    # b = 1 / (n * (n - 1)) * np.sum(kyy - np.diag(np.diagonal(kyy)))
    # c = 1 / (m * n) * np.sum(kxy)

    return np.sqrt(np.abs(a + b - 2 * c))


@njit
def mmdb_rf_njit(rfx: np.ndarray[int], rfy: np.ndarray[int], kernel: Kernel, *kernelargs) -> float:
    """
    Compute the biased estimator of MMD between two distributions.
    rfx and rfy must be matrices of ranks.
    kernel must take rank functions as input, without optional arguments.
    """
    kxx = gram_matrix_rf_njit(rfx, rfx, kernel, *kernelargs)
    kxy = gram_matrix_rf_njit(rfx, rfy, kernel, *kernelargs)
    kyy = gram_matrix_rf_njit(rfy, rfy, kernel, *kernelargs)

    # m = len(rfx)
    # n = len(rfy)

    a = np.mean(kxx)
    b = np.mean(kyy)
    c = np.mean(kxy)

    # a = 1 / (m * (m - 1)) * (np.sum(kxx) - np.sum(np.array([kxx[i, i] for i in range(m)])))
    # b = 1 / (n * (n - 1)) * (np.sum(kyy) - np.sum(np.array([kyy[i, i] for i in range(n)])))
    # c = 1 / (m * n) * np.sum(kxy)

    return np.sqrt(np.abs(a + b - 2 * c))


def var(sample: ru.SampleAM, kernel: Kernel = trivial_kernel, **kernelargs) -> float:
    """
    Sample variance of a distribution of adjacency matrices computed in a RKHS.
    """
    nv = len(sample)
    kxx = square_gram_matrix(sample, kernel, **kernelargs)
    return 1 / (nv - 1) * np.sum(np.diagonal(kxx)) - 1 / (nv * (nv - 1)) * np.sum(kxx)


@njit
def var_rf(rfx: np.ndarray[int], kernel: Kernel, *kernelargs) -> float:
    """
    Sample variance of a distribution of rank functions computed in a RKHS.
    """
    nv = rfx.shape[1]
    kxx = gram_matrix_rf_njit(rfx, rfx, kernel, *kernelargs)
    return 1 / (nv - 1) * np.sum(np.array([kxx[i, i] for i in range(nv)])) - 1 / (nv * (nv - 1)) * np.sum(kxx)


def subsample_mmd_distribution(sample: ru.SampleAM, subsample_size: int,
                               seed: int = 42, rep: int = 1000, use_njit: bool = False, use_key: bool = False,
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
    :param use_njit: if True, the kernel must support njit and rf (rank function)
    :type use_njit:
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
        sub1, sub2 = sample.get_subsamples_pair(subsample_size=subsample_size, seed=seed+2*ir, use_key=use_key,
                                                replace=replace, disjoint=disjoint, use_rf=use_njit)
        if use_njit:
            out[ir] = mmdb_rf_njit(sub1, sub2, kernel, *tuple(kernelargs.values()))
        else:
            out[ir] = mmdb(sub1, sub2, kernel=kernel, **kernelargs)
    return out


    # --- Checks
    # if use_key:
    #     try:
    #         if subsample_size > len(set(sample.key)) / 2:
    #             raise ValueError("Size of subsamples is too large, must be at most length of sample.key / 2.")
    #     except AttributeError:
    #         raise ValueError("The input sample has not key associated to it. Use sample.set_key to set one.")
    # else:
    #     if subsample_size > len(sample) / 2:
    #         raise ValueError("Size of subsamples is too large, must be at most length of sample / 2.")
    #
    # # --- Computation
    # keys = sample.key if use_key else range(len(sample))
    # if use_njit:
    #     sample = sample.to_rank_function_matrix()
    #
    # out = np.empty(rep)
    # for ir in range(rep):
    #     keys1 = np.random.default_rng(seed + ir).choice(keys, subsample_size, replace=False)
    #     keys2 = np.random.default_rng(rep * seed + ir).choice(np.setdiff1d(keys, keys1, assume_unique=not use_key),
    #                                                           subsample_size, replace=False)
    #     mask1 = np.isin(keys, keys1)
    #     mask2 = np.isin(keys, keys2)
    #
    #     if use_njit:
    #         out[ir] = mmdb_rf_njit(sample[:, mask1], sample[:, mask2], kernel, *tuple(kernelargs.values()))
    #     else:
    #         out[ir] = mmdb(sample[mask1], sample[mask2], kernel=kernel, **kernelargs)

    # sample = sample.copy()
    # sample.key = np.array(list(range(len(sample))))
    #
    # idxs = range(len(sample))
    # out = np.empty(rep)
    # for ir in range(rep):
    #     idxs1 = np.random.default_rng(seed + ir).choice(idxs, subsample_size, replace=False)
    #     idxs2 = np.random.default_rng(rep * seed + ir).choice(np.setdiff1d(idxs, idxs1, assume_unique=True),
    #                                                           subsample_size, replace=False)
    #     if use_njit:
    #         out[ir] = mmd_rf_njit(sample[:, idxs1], sample[:, idxs2], kernel, *tuple(kernelargs.values()))
    #     else:
    #         out[ir] = mmd(sample[idxs1], sample[idxs2], kernel=kernel, **kernelargs)

    # return out


def gram_matrix1(sample1: ru.SampleAM, sample2: ru.SampleAM, use_njit: bool = False,
                 kernel: Kernel = trivial_kernel, **kernelargs) -> np.ndarray[float]:
    """
    Gram matrix of the two samples.
    out[i, j] = kernel(sample1[i], sample2[j]).
    """

    if use_njit:
        sample1 = sample1.to_rank_function_matrix().T  # rows: voters, cols: alternatives
        sample2 = sample2.to_rank_function_matrix().T  #

    out = np.zeros((len(sample1), len(sample2)))
    for (i1, r1), (i2, r2) in product(enumerate(sample1), enumerate(sample2)):
        out[i1, i2] = kernel(r1, r2, **kernelargs)

    return out


def mmdb1(sample1: ru.SampleAM, sample2: ru.SampleAM, use_njit: bool = False,
          kernel: Kernel = trivial_kernel, **kernelargs) -> float:

    """
    Compute the biased estimator of MMD between two distributions.
    rfx and rfy must be matrices of ranks.
    kernel must take rank functions as input, without optional arguments.
    """
    kxx = gram_matrix1(sample1, sample1, use_njit=use_njit, kernel=kernel, **kernelargs)
    kxy = gram_matrix1(sample1, sample2, use_njit=use_njit, kernel=kernel, **kernelargs)
    kyy = gram_matrix1(sample2, sample2, use_njit=use_njit, kernel=kernel, **kernelargs)

    a = np.mean(kxx)
    b = np.mean(kyy)
    c = np.mean(kxy)

    return np.sqrt(np.abs(a + b - 2 * c))






