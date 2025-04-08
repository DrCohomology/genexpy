import numpy as np
try:
    import cupy
    if cupy.cuda.is_available():
        np = cupy
except:
    pass


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

na = 33      # number of alternatives
rep = 332     # number of subsamples
N = 120      # universe size
n = 5       # subsample size

k = 1
kernel = ku.jaccard_kernel
kernelargs = {}
distr = du.UniformDistribution(na=na, ties=True, seed=14)
sample = distr.sample(N)

for d, r in product([True, False], repeat=2):
    s1, s2 = sample.get_multisample_pair(subsample_size=n, rep=123, seed=42, disjoint=d, replace=r)

ams1 = s1.to_adjacency_matrices(na=na)
ams2 = s2.to_adjacency_matrices(na=na)

rvs1 = s1.to_rank_vectors()
rvs2 = s2.to_rank_vectors()

Kxx = kvu.mallows_gram(ams1, ams2, nu=0.1)
Kxy = kvu.gram_jaccard(rvs1, rvs2, k=1)
