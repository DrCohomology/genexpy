"""
Refactor subsample_mmd_distribution to remove the loop. Instead: vectorize the functions within
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
from genexpy import kernels as ku
from genexpy import rankings_utils as ru
from genexpy import mmd as mmd
from genexpy import probability_distributions as du

reload(ku)

RankVector: TypeAlias = np.ndarray[int]
RankByte: TypeAlias = bytes
Ranking: TypeAlias = Union[RankVector, RankByte]


na = 33      # number of alternatives
rep = 332     # number of subsamples
N = 120      # universe size
n = 5       # subsample size

k = 1
kernel = ku.jaccard_kernel
kernelargs = {}
distr = du.UniformDistribution(na=na, ties=True, seed=14)
sample = distr.sample(N)
rv = sample.to_rank_vector_matrix()

s1 = distr.sample(N)
rv1 = s1.to_rank_vector_matrix()
ams1 = np.array([ru.AdjacencyMatrix.from_bytes(r, shape=(na, na)) for r in s1])
s2 = distr.sample(N)
rv2 = s2.to_rank_vector_matrix()
ams2 = np.array([ru.AdjacencyMatrix.from_bytes(r, shape=(na, na)) for r in s2])

r1 = rv[:, 0]
r2 = rv[:, 1]

am1 = ru.AdjacencyMatrix.from_rank_vector(r1)
am2 = ru.AdjacencyMatrix.from_rank_vector(r2)

topk1 = np.where(r1 < k)[0]
topk2 = np.where(r2 < k)[0]
# return len(np.intersect1d(topk1, topk2)) / len(np.union1d(topk1, topk2))
jaccard = len(set(topk1).intersection(set(topk2))) / len(set(topk1).union(set(topk2)))


k1 = rv1 < k
k2 = rv2 < k
# intersection = (k1.T @ k2).astype(int)
intersection = np.logical_and(np.expand_dims(k1, 2), np.expand_dims(k2, 1)).astype(int).sum(axis=0)
union = np.logical_or(np.expand_dims(k1, 2), np.expand_dims(k2, 1)).astype(int).sum(axis=0)
j1 = intersection / union


def gram_jaccard(rv1: RankVector, rv2: RankVector, k: int):
    k1 = rv1 < k
    k2 = rv2 < k
    intersection = np.logical_and(np.expand_dims(k1, 2), np.expand_dims(k2, 1)).astype(int).sum(axis=0)
    union = np.logical_or(np.expand_dims(k1, 2), np.expand_dims(k2, 1)).astype(int).sum(axis=0)
    return intersection / union


# j1 = gram_jaccard(rv1, rv2, k)
int2 = np.empty((N, N))
uni2 = np.empty((N, N))
for i in range(N):
    for j in range(N):
        topk1 = np.where(rv1[:, i] < k)[0]
        topk2 = np.where(rv2[:, j] < k)[0]
        int2[i, j] = len(set(topk1).intersection(set(topk2)))
        uni2[i, j] = len(set(topk1).union(set(topk2)))




j2 = ku.gram_matrix(s1, s2, use_rv=True, kernel=ku.jaccard_kernel, k=1)

print(np.sum((j1-j2)**2))
print(np.sum((int2-intersection)**2))
print(np.sum((uni2-union)**2))

#%%
t0 = time.time()

mmd_veryold = mmd.subsample_mmd_distribution(sample, n, rep=rep, kernel=ku.jaccard_kernel, k=1, disjoint=True, replace=True)

t1 = time.time()

mmd_old = np.empty(rep)
for ir in tqdm(range(rep)):
    sub1, sub2 = sample.get_subsamples_pair(subsample_size=n, seed=422332131 + 2 * ir, replace=True, disjoint=True)  # so that it returns two SampleAM
    rv1 = sub1.to_rank_vector_matrix()
    rv2 = sub2.to_rank_vector_matrix()

    kxx = gram_jaccard(rv1, rv1, k=1)
    kxy = gram_jaccard(rv1, rv2, k=1)
    kyy = gram_jaccard(rv2, rv2, k=1)

    mmd_old[ir] = np.sqrt(np.abs(np.mean(kxx) + np.mean(kyy) - 2 * np.mean(kxy)))

t2 = time.time()

amss = np.broadcast_to(np.expand_dims(s1, axis=0), (rep, *s1.shape))
t11 = time.time()
# print("expand", t11 - t1)

shuffled = rng.permuted(amss, axis=1)
t12 = time.time()
print("shuffle", t12 - t11)

subs1 = np.array([rng.choice(sub, n, replace=True) for sub in shuffled[:, :N//2]])
subs2 = np.array([rng.choice(sub, n, replace=True) for sub in shuffled[:, N//2:]])
t13 = time.time()
print("sample", t13 - t12)

rvs1 = np.array([ru.SampleAM(subs_).to_rank_vector_matrix() for subs_ in subs1])
rvs2 = np.array([ru.SampleAM(subs_).to_rank_vector_matrix() for subs_ in subs2])

Kxx = gram_jaccard(rvs1, rvs1, k=1)
Kxy = gram_jaccard(rvs1, rvs2, k=1)
Kyy = gram_jaccard(rvs2, rvs2, k=1)

mmd_new = np.sqrt(np.abs(np.mean(Kxx, axis=(1, 2)) + np.mean(Kyy, axis=(1, 2)) - 2 * np.mean(Kxy, axis=(1, 2))))

t3 = time.time()

print(t1-t0, t2-t1, t3-t2)


# %% borda kernel

na = 13      # number of alternatives
rep = 332     # number of subsamples
N = 100      # universe size
n = 5       # subsample size
nu = 1 / na

kernel = ku.jaccard_kernel
kernelargs = {}
distr = du.UniformDistribution(na=na, ties=True, seed=14)
sample = distr.sample(N)
rv = sample.to_rank_vector_matrix()

s1 = distr.sample(N)
rv1 = s1.to_rank_vector_matrix()
ams1 = np.array([ru.AdjacencyMatrix.from_bytes(r, shape=(na, na)) for r in s1])
s2 = distr.sample(N)
rv2 = s2.to_rank_vector_matrix()
ams2 = np.array([ru.AdjacencyMatrix.from_bytes(r, shape=(na, na)) for r in s2])

r1 = rv[:, 0]
r2 = rv[:, 1]

am1 = ru.AdjacencyMatrix.from_rank_vector(r1)
am2 = ru.AdjacencyMatrix.from_rank_vector(r2)


idx = 0     # alternative


@np.vectorize(signature="(na, n), (na, n), (), () -> (n, n)")
def gram_borda(rv1, rv2, nu, idx):
    d1 = np.sum(rv1 >= rv1[idx], axis=0)  # dominated
    d2 = np.sum(rv2 >= rv2[idx], axis=0)

    return np.exp(- nu * np.abs(np.expand_dims(d1, axis=1) - np.expand_dims(d2, axis=0)))

@np.vectorize(signature="(na, n), (na, n), (), () -> (n, n)", otypes=[float])
def gram_borda2(rv1, rv2, nu, idx):
    d1 = np.sum(rv1 >= rv1[idx], axis=0)  # dominated
    d2 = np.sum(rv2 >= rv2[idx], axis=0)

    return np.exp(- nu * np.abs(np.expand_dims(d1, axis=1) - np.expand_dims(d2, axis=0)))


t0 = time.time()

b1 = gram_borda(rv1, rv2, nu, idx)

t1 = time.time()

# b2 = ku.gram_matrix(s1, s2, use_rv=True, kernel=ku.borda_kernel, idx=0, nu=nu)

t2 = time.time()

b3 = gram_borda2(rv1, rv2, nu, idx)

t3 = time.time()


print(t1-t0, t2-t1, t3-t2)
print(np.sum((b1-b2)**2))

#%%

rep = 123

rng = np.random.default_rng()

t0 = time.time()

mmd_veryold = mmd.subsample_mmd_distribution(sample, n, rep=rep, kernel=ku.borda_kernel, nu=nu, idx=0, disjoint=True, replace=True)

t1 = time.time()

mmd_old = np.empty(rep)
for ir in tqdm(range(rep)):
    sub1, sub2 = sample.get_subsamples_pair(subsample_size=n, seed=422332131 + 2 * ir, replace=True, disjoint=True)  # so that it returns two SampleAM
    rv1 = sub1.to_rank_vector_matrix()
    rv2 = sub2.to_rank_vector_matrix()

    kxx = gram_borda(rv1, rv1, nu, idx)
    kxy = gram_borda(rv1, rv2, nu, idx)
    kyy = gram_borda(rv2, rv2, nu, idx)

    mmd_old[ir] = np.sqrt(np.abs(np.mean(kxx) + np.mean(kyy) - 2 * np.mean(kxy)))

t2 = time.time()

amss = np.broadcast_to(np.expand_dims(s1, axis=0), (rep, *s1.shape))
t11 = time.time()
# print("expand", t11 - t1)

shuffled = rng.permuted(amss, axis=1)
t12 = time.time()
print("shuffle", t12 - t11)

subs1 = np.array([rng.choice(sub, n, replace=True) for sub in shuffled[:, :N//2]])
subs2 = np.array([rng.choice(sub, n, replace=True) for sub in shuffled[:, N//2:]])
t13 = time.time()
print("sample", t13 - t12)

rvs1 = np.array([ru.SampleAM(subs_).to_rank_vector_matrix() for subs_ in subs1])
rvs2 = np.array([ru.SampleAM(subs_).to_rank_vector_matrix() for subs_ in subs2])

Kxx = gram_borda(rvs1, rvs1, nu, idx)
Kxy = gram_borda(rvs1, rvs2, nu, idx)
Kyy = gram_borda(rvs2, rvs2, nu, idx)

mmd_new = np.sqrt(np.abs(np.mean(Kxx, axis=(1, 2)) + np.mean(Kyy, axis=(1, 2)) - 2 * np.mean(Kxy, axis=(1, 2))))

t3 = time.time()

print(t1-t0, t2-t1, t3-t2)


#%% compare

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp


test = ks_2samp(mmd_old, mmd_new)


fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
fig.suptitle(f"KS p-value: {test}")

sns.histplot(mmd_old, ax=ax, label="old")
sns.histplot(mmd_new, ax=ax, label="new")
sns.histplot(mmd_veryold, ax=ax, label="very old")

plt.legend()

fig.show()




#%% new implementation







def multisample_disjoint_replace(sample: ru.SampleAM, rep: int, n: int, rng: np.random.Generator):
    """
    Get 'rep' pairs of subsamples of size 'n', sampled with replacement from disjoint subsamples of 'sample'.
    'sample' has shape (N. ).

    Algorithm:
    1. Get rep copies of sample (rep, N).
    2. Shuffle each row independently.
    3. Split every row (roughly) in half and sample from each half independently.
    """
    N = len(sample)
    samples = np.broadcast_to(np.expand_dims(sample, axis=0), (rep, N))     # (rep, N)
    shuffled = rng.permuted(samples, axis=1)
    subs1 = np.array([rng.choice(sub, n, replace=True) for sub in shuffled[:, :N//2]])      # (rep, n)
    subs2 = np.array([rng.choice(sub, n, replace=True) for sub in shuffled[:, N//2:]])      # (rep, n)

    return subs1, subs2


def multisample_disjoint_not_replace(sample: ru.SampleAM, rep: int, n: int, rng: np.random.Generator):
    """
    Get 'rep' pairs of subsamples of size 'n', sampled with replacement from disjoint subsamples of 'sample'.
    'sample' has shape (N. ).

    Algorithm:
    1. Get rep copies of sample (rep, N).
    2. Shuffle each row independently.
    3. Split every row (roughly) in half and sample from each half independently.
    """
    N = len(sample)
    samples = np.broadcast_to(np.expand_dims(sample, axis=0), (rep, N))     # (rep, N)
    shuffled = rng.permuted(samples, axis=1)
    subs1 = np.array([rng.choice(sub, n, replace=False) for sub in shuffled[:, :N//2]])      # (rep, n)
    subs2 = np.array([rng.choice(sub, n, replace=False) for sub in shuffled[:, N//2:]])      # (rep, n)

    return subs1, subs2


def multisample_not_disjoint_replace(sample: ru.SampleAM, rep: int, n: int, rng: np.random.Generator):
    """
    Get 'rep' pairs of samples of size 'n', sampled with replacement from 'sample'.
    'sample' has shape (N. ).

    Algorithm:
    1. Get rep copies of sample (rep, N).
    2. Get a sample of size 2n from each row independently.
    3. Split the rows in half.
    """
    N = len(sample)
    samples = np.broadcast_to(np.expand_dims(sample, axis=0), (rep, N))  # (rep, N)
    tmp = np.array([rng.choice(sub, 2*n, replace=True) for sub in samples])     # (rep, 2*n)
    subs1 = tmp[:, :N//2]
    subs2 = tmp[:, N//2:]

    return subs1, subs2


def multisample_not_disjoint_not_replace(sample: ru.SampleAM, rep: int, n: int, rng: np.random.Generator):
    """
    Get 'rep' pairs of samples of size 'n', sampled with replacement from 'sample'.
    'sample' has shape (N. ).

    Algorithm:
    1. Get rep copies of sample (rep, N).
    2. Get a sample without replacement of size 2n from each row independently.
    3. Split the rows in half.
    """
    N = len(sample)
    samples = np.broadcast_to(np.expand_dims(sample, axis=0), (rep, N))  # (rep, N)
    subs1 = np.array([rng.choice(sub, n, replace=False) for sub in samples])    # (rep, n)
    subs2 = np.array([rng.choice(sub, n, replace=False) for sub in samples])    # (rep, n)

    return subs1, subs2



#%% final test

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



