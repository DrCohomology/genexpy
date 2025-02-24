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
from genexpy import kernels as ku
from genexpy import rankings_utils as ru
from genexpy import mmd as mmd
from genexpy import probability_distributions as du

from genexpy import kernels_vectorized as kvu

na = 2      # number of alternatives
rep = 10     # number of subsamples
N = 100      # universe size
n = 50       # subsample size

distr = du.UniformDistribution(na=na, ties=True, seed=14)
sample = distr.sample(N)

# estimation (biased MMD)
# mmds = kvu.mmd_distribution(sample, n, rep, "mallows", seed=42, disjoint=True, replace=False, nu="auto")
mmds_sq = mmd.subsample_mmdu_squared_distribution(sample, n, seed=42, rep=rep, disjoint=True, replace=False, nu="auto")


# s1, s2 = sample.get_subsamples_pair(subsample_size=n seed=43, replace=False, disjoint=True)
s1 = sample[:N//2]
s2 = sample[N//2:]

ams1 = np.array([ru.AdjacencyMatrix.from_bytes(r, shape=(na, na)) for r in s1])
ams2 = np.array([ru.AdjacencyMatrix.from_bytes(r, shape=(na, na)) for r in s2])

kxx = kvu.mallows_gram(ams1, ams1, nu="auto")
kyy = kvu.mallows_gram(ams2, ams2, nu="auto")
kxy = kvu.mallows_gram(ams1, ams2, nu="auto")

#%%

alpha = 0.99
eps = 0.01

h = kxx + kyy - kxy - kxy.T  # Gretton Lemma 6
var = np.var(h)

def mmd_ub_cum(eps, var, n):
    return 1 - np.exp(- (n//2) * (eps**2 / var + eps/3))

epss = np.linspace(0, 0.2, 10000)
mmds_ub_sq = np.array([mmd_ub_cum(eps, var, n) for eps in epss])
genu_sq = np.array([mmd.generalizability(mmds_sq, eps) for eps in epss])


fig, ax = plt.subplots()

# sns.ecdfplot(mmds_sq, ax=ax)
ax.plot(epss, mmds_ub_sq)
ax.plot(epss, genu_sq)

ax.axvline(eps)
ax.axhline(alpha)

fig.savefig("upper bound test 1.pdf")
fig.show()

#%%

def nstar(eps, alpha, var):
    return - np.log(alpha) * (var + eps/3) / eps**2


print(nstar(eps, alpha, var))




