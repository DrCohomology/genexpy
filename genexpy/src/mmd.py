import numpy as np

from . import kernels_classes as kcu
from . import rankings_utils as ru
from .kernels import square_gram_matrix, gram_matrix, trivial_kernel, Kernel
from .kernels_vectorized import AVAILABLE_VECTORIZED_KERNELS



def mmdb(sample1: ru.SampleAM, sample2: ru.SampleAM, use_rv: bool = True,
         kernel: Kernel = trivial_kernel, **kernelargs) -> float:
    """
    Compute the biased Maximum Mean Discrepancy (MMD) estimator between two distributions.

    This function uses kernel evaluations on ranked data to compute a biased estimate
    of the MMD between `sample1` and `sample2`.

    Parameters
    ----------
    sample1 : ru.SampleAM
        First sample of rankings.
    sample2 : ru.SampleAM
        Second sample of rankings.
    use_rv : bool, default=True
        Whether to use rank vectors instead of raw input.
    kernel : Kernel, default=trivial_kernel
        A kernel function to compute pairwise similarities between rankings.
    **kernelargs : dict
        Additional keyword arguments to pass to the kernel function.

    Returns
    -------
    float
        The biased MMD estimate between the two samples.
    """
    kxx = square_gram_matrix(sample1, use_rv=use_rv, kernel=kernel, **kernelargs)
    kxy = gram_matrix(sample1, sample2, use_rv=use_rv, kernel=kernel, **kernelargs)
    kyy = square_gram_matrix(sample2, use_rv=use_rv, kernel=kernel, **kernelargs)

    return np.sqrt(np.abs(np.mean(kxx) + np.mean(kyy) - 2 * np.mean(kxy)))


def mmdu_squared(sample1: ru.SampleAM, sample2: ru.SampleAM, use_rv: bool = True,
         kernel: Kernel = trivial_kernel, **kernelargs) -> float:
    """
    Compute the unbiased squared Maximum Mean Discrepancy (MMD) estimator between two distributions.

    This function uses kernel evaluations on ranked data to compute an unbiased estimate
    of the squared MMD between `sample1` and `sample2`.

    Parameters
    ----------
    sample1 : ru.SampleAM
        First sample of rankings.
    sample2 : ru.SampleAM
        Second sample of rankings.
    use_rv : bool, default=True
        Whether to use rank vectors instead of raw input.
    kernel : Kernel, default=trivial_kernel
        A kernel function to compute pairwise similarities between rankings.
    **kernelargs : dict
        Additional keyword arguments to pass to the kernel function.

    Returns
    -------
    float
        The unbiased squared MMD estimate between the two samples.
    """
    kxx = square_gram_matrix(sample1, use_rv=use_rv, kernel=kernel, **kernelargs)
    kxy = gram_matrix(sample1, sample2, use_rv=use_rv, kernel=kernel, **kernelargs)
    kyy = square_gram_matrix(sample2, use_rv=use_rv, kernel=kernel, **kernelargs)

    kxx = kxx - np.diag(kxx.diagonal())
    kyy = kyy - np.diag(kyy.diagonal())

    n = len(sample1)
    m = len(sample2)

    return np.sum(kxx)/(n*(n-1)) + np.sum(kyy)/(m*(m-1)) - 2 * np.mean(kxy)


def mmd_distribution(sample: ru.SampleAM, subsample_size: int,
                               seed: int = 42, rep: int = 1000, use_rv: bool = True, use_key: bool = False,
                               replace: bool = False, disjoint: bool = True,
                               kernel: Kernel = trivial_kernel, **kernelargs) -> np.ndarray[float]:
    """
    Estimate the distribution of the MMD by subsampling from a single sample.

    Performs repeated MMD computations on randomly drawn subsample pairs to approximate
    the distribution of the MMD under the null hypothesis (same distribution).

    Parameters
    ----------
    sample : ru.SampleAM
        Sample of rankings from which to draw subsamples.
    subsample_size : int
        Size of each subsample.
    seed : int, default=42
        Random seed used for reproducibility.
    rep : int, default=1000
        Number of repetitions.
    use_rv : bool, default=True
        Whether to use rank vectors instead of raw input.
    use_key : bool, default=False
        If True, sample using `sample.key` instead of `sample.index`.
    replace : bool, default=False
        Whether to sample with replacement.
    disjoint : bool, default=True
        Whether the two subsamples must be disjoint.
    kernel : Kernel, default=trivial_kernel
        A kernel function to compute pairwise similarities between rankings.
    **kernelargs : dict
        Additional keyword arguments to pass to the kernel function.

    Returns
    -------
    np.ndarray
        An array of MMD values from repeated subsampling.
    """

    out = np.empty(rep)
    for ir in range(rep):
        sub1, sub2 = sample.get_subsamples_pair(subsample_size=subsample_size, seed=seed + 2 * ir, use_key=use_key,
                                                replace=replace, disjoint=disjoint)  # so that it returns two SampleAM
        out[ir] = mmdb(sub1, sub2, use_rv=use_rv, kernel=kernel, **kernelargs)

    return out


def mmd_distribution_vectorized(sample: ru.SampleAM, n: int, rep: int, kernel_name: str,
                     seed: int = 0, disjoint: bool = True, replace: bool = False,
                     **kernelargs) -> np.ndarray[float]:
    """
    Computes the Maximum Mean Discrepancy (MMD) between multiple pairs of subsamples of a given sample.

    The MMD is a (pseudo-) distance between two probability distributions, it is the
    difference of their mean embeddings in a Reproducing Kernel Hilbert Space (RKHS).

    Parameters
    ----------
    sample : ru.SampleAM
        The sample from which to draw subsamples.
    n : int
        The size of each subsample.
    rep : int
        The number of repetitions to compute the MMD for.
    kernel_name : str
        The name of the kernel function to use. Must be a key in the
        `AVAILABLE_VECTORIZED_KERNELS` dictionary in `kernels_vectorized.py`.
    seed : int, optional
        The random seed to use for subsampling. The default is 0.
    disjoint : bool, optional
        Whether to draw disjoint subsamples. The default is True.
    replace : bool, optional
        Whether to draw subsamples with replacement. The default is False.
    **kernelargs : dict
        Additional keyword arguments to pass to the kernel function.

    Returns
    -------
    ndarray
        A 1D array of shape (rep,) containing the MMD values for each repetition.

    Raises
    ------
    ValueError
        If the kernel name is not found in the `AVAILABLE_VECTORIZED_KERNELS` dictionary.
        If the kernel format is not supported.

    See Also
    --------
    ru.SampleAM : Class representing a sample of adjacency matrices.
    kernels_vectorized : Module containing vectorized kernel functions.

    Notes
    -----
    The MMD is defined as:

    .. math::
        MMD(P, Q) = || \mathbb{E}_{x \sim P} [\phi(x)] - \mathbb{E}_{y \sim Q} [\phi(y)] ||_{\mathcal{H}}

    where :math:`P` and :math:`Q` are the two probability distributions, :math:`\phi` is the
    feature map associated with the kernel, and :math:`|| \cdot ||_{\mathcal{H}}` is the norm
    in the RKHS.

    In this function, we estimate the MMD by drawing two subsamples from the given sample
    and computing the difference in their means in the RKHS.
    """
    s1, s2 = sample.get_multisample_pair(subsample_size=n, rep=rep, seed=seed, disjoint=disjoint, replace=replace)

    s1 = ru.MultiSampleAM(s1)
    s2 = ru.MultiSampleAM(s2)

    na = int(np.sqrt(len(s1[0,0])))  # number of alternatives

    try:
        kernel_gram, kernel_format = AVAILABLE_VECTORIZED_KERNELS[kernel_name]
    except KeyError:
        raise ValueError(f"Kernel name {kernel_name} does not have a vectorized implementation in kernels_vectorized.py"
                         f"If it is a custom kernel, make sure to add it to "
                         f"kernels_vectorized.AVAILABLE_VECTORIZED_KERNELS in order to make it visible for this function.")

    if kernel_format == "adjmat":
        x1 = s1.to_adjacency_matrices(na=na)
        x2 = s2.to_adjacency_matrices(na=na)
    elif kernel_format == "vector":
        x1 = s1.to_rank_vectors()
        x2 = s2.to_rank_vectors()
    else:
        raise ValueError(f"Unsupported kernel format: {kernel_format}")

    Kxx = kernel_gram(x1, x1, **kernelargs)
    Kyy = kernel_gram(x2, x2, **kernelargs)
    Kxy = kernel_gram(x1, x2, **kernelargs)

    return np.sqrt(np.abs(np.mean(Kxx, axis=(1, 2)) + np.mean(Kyy, axis=(1, 2)) - 2 * np.mean(Kxy, axis=(1, 2))))


def mmd_distribution_vectorized_class(sample: ru.SampleAM, n: int, rep: int, kernel_obj: kcu.Kernel,
                                     seed: int = 0, disjoint: bool = True, replace: bool = False) -> np.ndarray[float]:


    s1, s2 = sample.get_multisample_pair(subsample_size=n, rep=rep, seed=seed, disjoint=disjoint, replace=replace)

    s1 = ru.MultiSampleAM(s1)
    s2 = ru.MultiSampleAM(s2)

    na = int(np.sqrt(len(s1[0,0])))  # number of alternatives

    match kernel_obj.vectorized_input_format:
        case "adjmat":
            x1 = s1.to_adjacency_matrices(na=na)
            x2 = s2.to_adjacency_matrices(na=na)
        case "vector":
            x1 = s1.to_rank_vectors()
            x2 = s2.to_rank_vectors()
        case _:
            raise ValueError(f"Unsupported input format for vectorized kernel: {kernel_obj.vectorized_input_format}")

    Kxx = kernel_obj.gram_matrix(x1, x1)
    Kyy = kernel_obj.gram_matrix(x2, x2)
    Kxy = kernel_obj.gram_matrix(x1, x2)

    return np.sqrt(np.abs(np.mean(Kxx, axis=(1, 2)) + np.mean(Kyy, axis=(1, 2)) - 2 * np.mean(Kxy, axis=(1, 2))))

def generalizability(mmd_distr: np.ndarray[float], eps: float) -> float:
    return np.mean(mmd_distr <= eps)
