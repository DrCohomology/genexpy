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
from genexpy import probability_distributions as du
from genexpy import rankings_utils as ru

reload(du)
reload(ku)

N = 2000
n = 200
seed = 444
disjoint = True
replace = False

results = []
for na in [5, 10, 15]:
    t_ = time()
    distr = du.UniformDistribution(na=na, ties=True, seed=seed * na)
    sample = distr.sample(N)
    t__ = time()
    for rep in tqdm(list(range(100, 1001, 200)), desc=f"na = {na}"):
        for n in (list(range(10, 211, 40))):

            kernel_obj = ku.MallowsKernel(nu="auto")

            t0 = time()
            if n < 30 and na < 12 and rep < 500:
                kernel_obj._mmd_distribution_naive(sample, n, rep)
            t1 = time()
            kernel_obj._mmd_distribution_vectorized(sample, n, rep)
            t2 = time()
            kernel_obj._mmd_distribution_embedding(sample, n, rep)
            t3 = time()

            results.append({
                "n_v": n,
                "n_a": na,
                "n_rep": rep,
                "sampling_time": t__ - t_,
                "naive_time": t1 - t0,
                "vectorized_time": t2 - t1,
                "embedding_time": t3 - t2
            })

results = pd.DataFrame(results)
results.to_parquet("mmd_benchmark.parquet")

results = pd.read_parquet("mmd_benchmark.parquet").rename(columns={"vectorized_naive_time": "vectorized_time"})

#%%


dfplot = (results.melt(id_vars=["n_v", "n_a", "n_rep"], value_vars=["naive_time", "vectorized_time", "embedding_time"], value_name="time", var_name="method"))
dfplot = dfplot.query("time > 0")

fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharey="all")
axes = axes.flatten()

ax = axes[0]
# df_ = dfplot.query("n_a == 10 and n_rep == 300")
df_ = dfplot.query("method != 'naive_time'")
sns.lineplot(df_, x="n_v", y="time", hue="method", style="method", ax=ax, estimator=np.median, errorbar=("pi", 50))
# ax.set_title("n_a = 10, n_rep = 300")


ax = axes[1]
# df_ = dfplot.query("n_v == 10 and n_rep == 300")
sns.lineplot(df_, x="n_a", y="time", hue="method", style="method", ax=ax, estimator=np.median, errorbar=("pi", 50))
# ax.set_title("n_v = 10, n_rep = 300")


ax = axes[2]
# df_ = dfplot.query("n_v == 10 and n_a == 10")
sns.lineplot(df_, x="n_rep", y="time", hue="method", style="method", ax=ax, estimator=np.median, errorbar=("pi", 50))
# ax.set_title("n_a = 10, n_v = 10")

sns.despine()

plt.tight_layout()
fig.savefig("mmd_method_benchmark2.pdf")
plt.show()




