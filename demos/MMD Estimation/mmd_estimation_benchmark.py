"""
# Estimation of the MMD

The module provides three ways to estimate the distribution of the MMD.
The distribution of the MMD is estimated by:
    Repeat rep times:
        1. Take two samples of rankings, X and Y
        2. Build the Gram matrices Kxx, Kyy, Kxy
        3. Compute the MMD between X and Y as the sqrt of the mean of Kxx + Kyy - 2Kxy
        4. Append the MMD to a vector
    The vector of MMDs is the sampled distribution

## 1  With basic kernels (not considered, way too slow)
A "basic" kernel is a function that computes the kernel between two rankings.
It has signature kernel(ranking, ranking, **parameters) -> float

Implemented in:
    - Kernels: in kernels.py, see borda_kernel, jaccard_kernel, mallows_kernel
    - Gram marices: in kernels.py, see gram_matrix, square_gram_matrix
    - MMD distribution: mmd, see mmd_distribution

## 2  With vectorized Gram matrices
A vectorized Gram matrix does not compute the kernel directly, instead, it directly computes the Gram matrix.
This approach is vectorized to be able to take a set of pairs of samples of rankings as inputs and compute the Gram
    matrix for each pair using np.vectorize.

Implemented in:
    - Gram matrices:
        - in kernels_vectorized.py, see borda_gram, jaccard_gram, mallows_gram
        - in kernels_classes.py, see BordaKernel, Jaccard_Kernel, MallowsKernel
    - MMD distribution: in mmd.py, see mmd_distribution_vectorized, mmd_distribution_vectorized_classes

## 3 Error-based
Using the newly found formula for the MMD of finite-dimensional RKHSs, we can provide a (hopefully) faster estimate
    if the true distribution is known.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import defaultdict
from importlib import reload
from itertools import product
from time import time
from tqdm.auto import tqdm

from genexpy import kernels as ku
from genexpy import kernels_vectorized as kvu
from genexpy import kernels_classes as kcu
from genexpy import mmd
from genexpy import probability_distributions as du
from genexpy import rankings_utils as ru

reload(du)
reload(ku)
reload(kvu)
reload(kcu)


N = 1000
rep = 1000
n = 200
seed = 444
disjoint = True
replace = False
kernel_obj = kcu.JaccardKernel(k=2)


def get_pmfs_df_from_multisample(ms: ru.MultiSampleAM) -> pd.DataFrame:
    """
    Create a dataframe. Index: rankings. Columns: samples in 'ms'.

    Parameters
    ----------
    ms : ru.MultiSample

    Returns
    -------
    a pd.DataFrame

    """
    tmps = []
    for i, s in enumerate(ms):
        universe, pmf = ru.SampleAM(s).get_universe_pmf()
        tmps.append(pd.Series(pmf, index=universe))
    return pd.concat(tmps, axis=1, ignore_index=True).fillna(0)


mmds_dict = defaultdict(lambda: [])
times_dict = defaultdict(lambda: [])
for na in tqdm(list(range(2, 8))):
    distr = du.UniformDistribution(na=na, ties=True, seed=seed)
    sample = distr.sample(N)

    # Multisampling
    ms1, ms2 = sample.get_multisample_pair(subsample_size=n, rep=rep, seed=seed, disjoint=disjoint, replace=replace)
    ms1 = ru.MultiSampleAM(ms1)
    ms2 = ru.MultiSampleAM(ms2)

    na = int(np.sqrt(len(ms1[0, 0])))  # number of alternatives

    # --- Vectorized

    match kernel_obj.vectorized_input_format:
        case "adjmat":
            x1 = ms1.to_adjacency_matrices(na=na)
            x2 = ms2.to_adjacency_matrices(na=na)
        case "vector":
            x1 = ms1.to_rank_vectors()
            x2 = ms2.to_rank_vectors()
        case _:
            raise ValueError(f"Unsupported input format for vectorized kernel: {kernel_obj.vectorized_input_format}")
    t0 = time()
    Kxx = kernel_obj.gram_matrix(x1, x1)
    Kyy = kernel_obj.gram_matrix(x2, x2)
    Kxy = kernel_obj.gram_matrix(x1, x2)

    mmds_vec = np.sqrt(np.abs(np.mean(Kxx, axis=(1, 2)) + np.mean(Kyy, axis=(1, 2)) - 2 * np.mean(Kxy, axis=(1, 2))))

    mmds_dict["vec"].append(mmds_vec)
    times_dict["vec"].append(time() - t0)


    # --- Error-based

    t0 = time()
    pmf_df1 = get_pmfs_df_from_multisample(ms1)
    pmf_df2 = get_pmfs_df_from_multisample(ms2)

    t1 = time()

    # ms1[0] is compared to ms2[0] etc...
    # equivalently, pmf_df1.iloc[:, 0] is compared with pmf_df2.iloc[:, 0]
    alpha_df = pmf_df1 - pmf_df2
    alpha_df = alpha_df.fillna(pmf_df1).fillna(-pmf_df2)  # if a ranking does not appear in both is an NaN

    t2 = time()

    # get the kernel matrix from the index of alpha (the universe)
    universe = ru.SampleAM(alpha_df.index.values)
    ams = universe.to_adjmat_array((na, na))
    K = kernel_obj.gram_matrix(ams, ams)

    t3 = time()

    alpha = alpha_df.values
    mmds_error = np.sqrt(np.diag(alpha.T @ K @ alpha))
    # mmds_extended = np.sqrt(alpha.T @ K @ alpha)
    """
    We are wasting all that is not on the diagonal: can we use it/remove it?
    """



    t4 = time()

    mmds_dict["error"].append(mmds_error)
    times_dict["error"].append(time() - t0)

diffs = np.zeros((len(mmds_dict), len(mmds_dict)))
for (i1, (m1, g1)), (i2, (m2, g2)) in product(enumerate(mmds_dict.items()), repeat=2):
    diffs[i1, i2] = np.linalg.norm(np.array(g1)-np.array(g2))

print("If not close to 0 there's a problem:",  np.linalg.norm(diffs))
times_print = {method: np.round(times, 2) for method, times in times_dict.items()}
for method, times in times_print.items():
    print(f"{method}: {times}")
