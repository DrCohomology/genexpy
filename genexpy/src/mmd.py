import numpy as np

from . import rankings_utils as ru
from .kernels import square_gram_matrix, gram_matrix, trivial_kernel, Kernel

def mmdb(sample1: ru.SampleAM, sample2: ru.SampleAM, use_rv: bool = True,
         kernel: Kernel = trivial_kernel, **kernelargs) -> float:
    """
    Compute the biased estimator of MMD between two distributions.
    rvx and rvy must be matrices of ranks.
    kernel must take rank functions as input, without optional arguments.
    """
    kxx = square_gram_matrix(sample1, use_rv=use_rv, kernel=kernel, **kernelargs)
    kxy = gram_matrix(sample1, sample2, use_rv=use_rv, kernel=kernel, **kernelargs)
    kyy = square_gram_matrix(sample2, use_rv=use_rv, kernel=kernel, **kernelargs)

    return np.sqrt(np.abs(np.mean(kxx) + np.mean(kyy) - 2 * np.mean(kxy)))


def subsample_mmd_distribution(sample: ru.SampleAM, subsample_size: int,
                               seed: int = 42, rep: int = 1000, use_rv: bool = True, use_key: bool = False,
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
    :param use_rv: if True, the kernel must support njit and rv (rank function)
    :type use_rv:
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
                                                replace=replace, disjoint=disjoint, use_rv=False)  # so that it returns two SampleAM
        out[ir] = mmdb(sub1, sub2, use_rv=use_rv, kernel=kernel, **kernelargs)

    return out

# Cumulative function of MMD
def generalizability(mmd_distr: np.ndarray[float], eps: float) -> float:
    return float(np.mean(mmd_distr <= eps))