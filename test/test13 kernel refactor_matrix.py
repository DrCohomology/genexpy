"""
Refactor the Mallows kernel using adjacency matrices instead of rankings.
Try combining them all together in one tensor to calculate the Gram matrix.

USE XOR; IT'S BETTER
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns

from collections import Counter, defaultdict
from importlib import reload
from itertools import product
from numba import njit
from pathlib import Path
from scipy.special import binom
from scipy.stats import binomtest, friedmanchisquare, spearmanr, wilcoxon
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import time
from tqdm.auto import tqdm
from typing import Union, TypeAlias, Literal
from genexpy import lower_bounds as gu
from genexpy import kernels as ku
from genexpy import rankings_utils as ru
from genexpy import mmd as mmd
from genexpy import probability_distributions as du

reload(ku)


RankVector: TypeAlias = np.ndarray[int]
RankByte: TypeAlias = bytes
Ranking: TypeAlias = Union[RankVector, RankByte]

na = 30
kernel = ku.mallows_kernel
kernelargs = {}
distr = du.UniformDistribution(na=na, ties=True, seed=14)
s1 = distr.sample(1000)
s2 = distr.sample(1000)
rv1 = s1.to_rank_vector_matrix()
rv2 = s2.to_rank_vector_matrix()

r1 = rv1[:, 0]
r2 = rv1[:, 1]

am1 = ru.AdjacencyMatrix.from_rank_vector(r1)
am2 = ru.AdjacencyMatrix.from_rank_vector(r2)


nu = 2 / (na * (na - 1))

def mallows_featuremap(r: RankVector):
    """
    Feature map of the Mallows kernel: adjacency matrix.
    """
    # return np.array([[ri <= rj for rj in r] for ri in r], dtype=int)

    return (np.array([[ri < rj for rj in r] for ri in r], dtype=int) -
            np.array([[rj < ri for rj in r] for ri in r], dtype=int))

@njit
def mallows_am_xor(am1, am2, nu):
    return np.exp(-nu * np.sum(np.bitwise_xor(am1, am2)) / 2)

@njit
def mallows_am(am1, am2, nu):
    return np.exp(-nu * np.sum(np.abs(am1-am2)) / 2)

#%%
@njit
def matrix_pairs_numba_fixed(a, b):
    n_a = a.shape[0]
    n_b = b.shape[0]

    # Prepare an output array with the correct shape
    pairs = np.empty((n_a * n_b, 2, a.shape[1], a.shape[2]), dtype=a.dtype)

    index = 0
    for i in range(n_a):
        for j in range(n_b):
            pairs[index, 0] = a[i]
            pairs[index, 1] = b[j]
            index += 1

    return pairs

@njit
def hypergram(ams1, ams2):
     tmp = matrix_pairs_numba_fixed(ams1, ams2)
     xor = np.abs(tmp[:, 0]-tmp[:, 1])
     s = np.sum(xor, axis=(1, 2))
     rs = s.reshape(len(ams1), len(ams2))
     return np.exp(-nu / 2 * rs)

def broadcasting_gram(ams1, ams2):
    ndisc = np.abs(np.expand_dims(ams1, axis=1) - np.expand_dims(ams2, axis=0)).sum(axis=(-1, -2))
    return np.exp(-nu / 2 * ndisc)

@njit
def broadcasting_gram_njit(ams1: np.ndarray[int], ams2: np.ndarray[int]) -> np.ndarray[float]:
    ndisc = np.abs(np.expand_dims(ams1, axis=1) - np.expand_dims(ams2, axis=0))
    nd = ndisc.reshape((ndisc.shape[0], ndisc.shape[1], -1)).sum(-1)
    return np.exp(-nu / 2 * nd)

def broadcasting_xor(ams1, ams2, nu):
    ndisc = np.logical_xor(np.expand_dims(ams1, axis=1), np.expand_dims(ams2, axis=0)).sum(axis=(-1, -2))
    return np.exp(-nu / 2 * ndisc)


t0 = time.time()

g1 = ku.gram_matrix(s1, s2, use_rv=True, kernel=ku.mallows_kernel)

t1 = time.time()

ams1 = np.array([ru.AdjacencyMatrix.from_bytes(r, shape=(na, na)) for r in s1])
ams2 = np.array([ru.AdjacencyMatrix.from_bytes(r, shape=(na, na)) for r in s2])

t2 = time.time()

# tmp = matrix_pairs_numba_fixed(ams1, ams2)

t3 = time.time()

# g2 = np.exp(-nu / 2 * np.logical_xor(tmp[:, 0], tmp[:, 1]).astype(int).sum(axis=(1, 2)).reshape(len(ams1), len(ams2)))

t4 = time.time()

# g3 = hypergram(ams1, ams2)
g3 = broadcasting_gram(ams1, ams2)

t5 = time.time()

g4 = broadcasting_xor(ams1, ams2, nu=nu)

t6 = time.time()


times = {
    "old_rv": t1-t0,
    # "am": t2-t1,
    # "cartesian": t3-t2,
    # "math": t4-t3,
    # "am_xor": t4-t1,
    # "am_hyper": t5-t4 + t2-t1
    "am_broad": t5-t4 + t2-t1,
    "am_broad_xor": t6-t5 + t2-t1,
}


grams = {
    "old_rv": g1,
    # "am_xor": g2,
    "am_broad": g3,
    "am_broad_nb": g4
}


print("Not square")
for k, v in times.items():
    print(f"{k:15s} {v:03f}")
print()

for k1, v1 in grams.items():
    for k2, v2 in grams.items():
        if np.sum((v1-v2)**2) > 0:
            print(k1, k2)


#%%

def test_subsample_mmd_distribution(sample: ru.SampleAM, subsample_size: int,
                               seed: int = 42, rep: int = 1000, use_rv: bool = True, use_key: bool = False,
                               replace: bool = False, disjoint: bool = True) -> np.ndarray[float]:
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
                                                replace=replace, disjoint=disjoint)  # so that it returns two SampleAM
        ams1 = np.array([ru.AdjacencyMatrix.from_bytes(r, shape=(na, na)) for r in sub1])
        ams2 = np.array([ru.AdjacencyMatrix.from_bytes(r, shape=(na, na)) for r in sub2])

        kxx = broadcasting_xor(ams1, ams1, nu=nu)
        kxy = broadcasting_xor(ams1, ams2, nu=nu)
        kyy = broadcasting_xor(ams2, ams2, nu=nu)

        out[ir] = np.sqrt(np.abs(np.mean(kxx) + np.mean(kyy) - 2 * np.mean(kxy)))
    return out



tm0 = time.time()
mmd.subsample_mmd_distribution(sample=s1, seed=1444, subsample_size=10, rep=1000, use_rv=True, kernel=ku.mallows_kernel)
tm1 = time.time()
test_subsample_mmd_distribution(sample=s1, seed=1444, subsample_size=10, rep=1000, use_rv=True)
tm2 = time.time()

mmd_times = {
    "old_rv": tm1-tm0,
    "bc_xor": tm2-tm1,
}

print("MMD")
for k, v in mmd_times.items():
    print(f"{k:15s} {v:03f}")
print()