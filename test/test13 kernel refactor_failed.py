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

kernel = ku.mallows_kernel
kernelargs = {}
distr = du.UniformDistribution(na=100, ties=True)
s1 = distr.sample(100)
s2 = distr.sample(100)
rv1 = s1.to_rank_vector_matrix()
rv2 = s2.to_rank_vector_matrix()

class Kernel:

    def __init__(self, use_rv=True, **kernelargs):
        self.kernelargs = kernelargs

    def _bytes(self, b1: RankByte, b2: RankByte):
        pass

    def _rv(self, r1: RankVector, r2: RankVector):
        pass

    def __call__(self, **kernelargs):
        pass


class MallowsKernel:

    def __init__(self, use_rv=True, nu: Union[float, Literal["auto"]] = "auto"):
        # super().__init__(use_rv)
        self.nu = nu

    @staticmethod
    def _bytes(b1: RankByte, b2: RankByte, nu: float) -> float:
        i1 = np.frombuffer(b1, dtype=np.int8)
        i2 = np.frombuffer(b2, dtype=np.int8)
        return np.exp(- nu * np.sum(np.abs(i1 - i2)) / 2)


    @staticmethod
    @njit
    def _rv(r1: RankVector, r2: RankVector, nu: float) -> float:
        n = len(r1)
        out = 0  # twice the number of discordant pairs ((tie, not-tie) counts as 1/2 discordant)
        for i in range(n):
            for j in range(i):
                out += np.abs(np.sign(r1[i] - r1[j]) - np.sign(r2[i] - r2[j]))
        return np.exp(- nu * out / 2)

    def __call__(self, x1: Ranking, x2: Ranking, use_rv: bool = True,
                 nu: Union[float, Literal["auto"]] = "auto") -> float:
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
            nu = 2 / (n * (n - 1))

        self.use_rv = True
        self.nu = nu

        if use_rv:
            return self._rv(x1, x2, nu=nu)
        else:
            return self._bytes(x1, x2, nu=nu)


import time

mk = MallowsKernel()

# t0 = time.time()
# print(f"Old kernel      : {kernel(s1[0], s2[0], use_rv=False, **kernelargs)}")
# print(f"Old kernel rv   : {kernel(rv1[:, 0], rv2[:, 0], use_rv=True, **kernelargs)}")
# t1 = time.time()
#
# print(f"N kernel          : {mk(s1[0], s2[0], use_rv=False, **kernelargs)}")
# print(f"N kernel rv      : {mk(rv1[:, 0], rv2[:, 0], use_rv=True, **kernelargs)}")
# t2 = time.time()
#
# # print(f"New kernel      : {MallowsKernel()(s1[0], s2[0], use_rv=False, **kernelargs)}")
# # print(f"New kernel rv   : {MallowsKernel()(rv1[:, 0], rv2[:, 0], use_rv=True, **kernelargs)}")
# # t3 = time.time()

# print(t1-t0)
# print(t2-t1)
# print(t3-t2)

t0 = time.time()
# g1 = ku.square_gram_matrix(s1, use_rv=False, kernel=ku.mallows_kernel)
t1 = time.time()
g2 = ku.square_gram_matrix(s1, use_rv=True, kernel=ku.mallows_kernel)
t2 = time.time()

# g3 = ku.square_gram_matrix(s1, use_rv=False, kernel=MallowsKernel())
t3 = time.time()
g4 = ku.square_gram_matrix(s1, use_rv=True, kernel=MallowsKernel())
t4 = time.time()

# g5 = ku.square_gram_matrix(s1, use_rv=False, kernel=mk)
t5 = time.time()
g6 = ku.square_gram_matrix(s1, use_rv=True, kernel=mk)
t6 = time.time()


sq_times = {
    # "old_bytes": t1-t0,
    "old_rv": t2-t1,
    # "new_bytes": t3-t2,
    "new_rv": t4-t3,
    # "eff_bytes": t5-t4,
    "eff_rv": t6-t5
}

sq_grams = {
    # "old_bytes": g1,
    "old_rv": g2,
    # "new_bytes": g3,
    "new_rv": g4,
    # "eff_bytes": g5,
    "eff_rv": g6
}

t0 = time.time()
# g1 = ku.gram_matrix(s1, s2, use_rv=False, kernel=ku.mallows_kernel)
t1 = time.time()
g2 = ku.gram_matrix(s1, s2, use_rv=True, kernel=ku.mallows_kernel)
t2 = time.time()

# g3 = ku.gram_matrix(s1, s2, use_rv=False, kernel=MallowsKernel())
t3 = time.time()
g4 = ku.gram_matrix(s1, s2, use_rv=True, kernel=MallowsKernel())
t4 = time.time()

# g5 = ku.gram_matrix(s1, s2, use_rv=False, kernel=mk)
t5 = time.time()
g6 = ku.gram_matrix(s1, s2, use_rv=True, kernel=mk)
t6 = time.time()

times = {
    # "old_bytes": t1-t0,
    "old_rv": t2-t1,
    # "new_bytes": t3-t2,
    "new_rv": t4-t3,
    # "eff_bytes": t5-t4,
    "eff_rv": t6-t5
}

grams = {
    # "old_bytes": g1,
    "old_rv": g2,
    # "new_bytes": g3,
    "new_rv": g4,
    # "eff_bytes": g5,
    "eff_rv": g6
}

print("Square")
for k, v in sq_times.items():
    print(f"{k:15s} {v:03f}")
print()

print("Not square")
for k, v in times.items():
    print(f"{k:15s} {v:03f}")
print()

for k1, v1 in grams.items():
    for k2, v2 in grams.items():
        if np.sum((v1-v2)**2) > 0:
            print(k1, k2)

for k1, v1 in sq_grams.items():
    for k2, v2 in sq_grams.items():
        if np.sum((v1-v2)**2) > 0:
            print(k1, k2)

#%%

tm0 = time.time()
mmd.subsample_mmd_distribution(sample=s1, seed=1444, subsample_size=20, rep=1000, use_rv=True, kernel=ku.mallows_kernel)
tm1 = time.time()
mmd.subsample_mmd_distribution(sample=s1, seed=1444, subsample_size=20, rep=1000, use_rv=True, kernel=MallowsKernel())
tm2 = time.time()
mmd.subsample_mmd_distribution(sample=s1, seed=1444, subsample_size=20, rep=1000, use_rv=True, kernel=mk)
tm3 = time.time()

mmd_times = {
    "old_rv": tm1-tm0,
    "new_rv": tm2-tm1,
    "eff_rv": tm3-tm2
}

print("MMD")
for k, v in mmd_times.items():
    print(f"{k:15s} {v:03f}")
print()