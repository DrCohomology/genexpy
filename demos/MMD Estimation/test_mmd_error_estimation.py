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
reload(ru)


def get_pmfs_df_from_multisample(ms: ru.MultiSampleAM, pmf_df_base = pd.Series(), with_base_pmf=False) -> pd.DataFrame:
    """
    Create a dataframe. Index: rankings. Columns: samples in 'ms'.
    The index can be given with pmf_df_base (optional). It is useful if you have a universe and want to keep the pmfs in
        line with it.

    Parameters
    ----------
    ms : ru.MultiSample

    Returns
    -------
    a pd.DataFrame

    """
    tmps = [pmf_df_base]
    for i, s in enumerate(ms):
        universe, pmf = ru.SampleAM(s).get_universe_pmf()
        tmps.append(pd.Series(pmf, index=universe))

    out = pd.concat(tmps, axis=1, ignore_index=False).fillna(0)
    return out if with_base_pmf else out.drop(columns=pmf_df_base.name)


rep = 1000
seed = 444
ns = [2**m for m in range(2, 11)]
N = 16 * max(ns)
na = 3
disjoint = True
replace = False
ps = np.linspace(2, 10, 101)

kernels = [
    # kcu.BordaKernel(nu="auto", idx=0),
    kcu.JaccardKernel(k=1),
    # kcu.JaccardKernel(k=3),
    # kcu.MallowsKernel(nu="auto"),
    # kcu.MallowsKernel(nu=0.3),
    # kcu.MallowsKernel(nu=1.0),
]

distr = du.UniformDistribution(na=na, ties=True, seed=seed)
# distr = du.MDegenerateDistribution(na=na, m=1, ties=True, seed=seed)

sample = distr.sample(N)
universe_base, pmf_base = sample.get_universe_pmf()
pmf_df_base = pd.Series(pmf_base, index=universe_base, name="universe")

kernel_obj = kernels[0]     # TODO Assuming one kernel
match kernel_obj.vectorized_input_format:
    case "adjmat":
        x = universe_base.to_adjmat_array((na, na))
        # x1 = ms1.to_adjacency_matrices(na=na)
        # x2 = ms2.to_adjacency_matrices(na=na)
    case "vector":
        x = universe_base.to_rank_vector_matrix()
        # x1 = ms1.to_rank_vectors()
        # x2 = ms2.to_rank_vectors()
    case _:
        raise ValueError(f"Unsupported input format for vectorized kernel: {kernel_obj.vectorized_input_format}")
K = kernel_obj.gram_matrix(x, x)

mmds_dict = defaultdict(lambda: [])
times_dict = defaultdict(lambda: [])
norms = defaultdict(lambda: dict())
out = []
Ks = defaultdict(lambda: {})
Gs = defaultdict(lambda: {})
Hs = defaultdict(lambda: {})
nVns = defaultdict(lambda: {})
mmds = defaultdict(lambda: {})
for n in tqdm(ns):
    # Multisampling
    ms1, ms2 = sample.get_multisample_pair(subsample_size=n, rep=rep, seed=seed+n, disjoint=disjoint, replace=replace)
    ms1 = ru.MultiSampleAM(ms1)
    ms2 = ru.MultiSampleAM(ms2)

    # --- Error-based approach

    t0 = time()
    pmf_df1 = get_pmfs_df_from_multisample(ms1, pmf_df_base=pmf_df_base, with_base_pmf=False)
    pmf_df2 = get_pmfs_df_from_multisample(ms2, pmf_df_base=pmf_df_base, with_base_pmf=False)

    t1 = time()

    # ms1[0] is compared to ms2[0] etc...
    # equivalently, pmf_df1.iloc[:, 0] is compared with pmf_df2.iloc[:, 0]
    alpha_df = pmf_df1 - pmf_df2
    # alpha_df = alpha_df.fillna(pmf_df1).fillna(-pmf_df2)  # NaN = does not appear in either pmf. should not be a problem anymore

    # get the kernel matrix from the index of alpha (the universe)
    universe = ru.SampleAM(alpha_df.index.values)
    ams = universe.to_adjmat_array((na, na))
    rvs = universe.to_rank_vector_matrix()

    t2 = time()

    alpha = alpha_df.values
    for kernel_obj in kernels:
        match kernel_obj.vectorized_input_format:
            case "adjmat":
                x = universe.to_adjmat_array((na, na))
                # x1 = ms1.to_adjacency_matrices(na=na)
                # x2 = ms2.to_adjacency_matrices(na=na)
            case "vector":
                x = universe.to_rank_vector_matrix()
                # x1 = ms1.to_rank_vectors()
                # x2 = ms2.to_rank_vectors()
            case _:
                raise ValueError(f"Unsupported input format for vectorized kernel: {kernel_obj.vectorized_input_format}")

        # Ks[kernel_obj.__repr__()][n] = kernel_obj.gram_matrix(x, x)  # TODO can be moved out of loop
        Ks[kernel_obj.__repr__()][n] = K

        # Gram matrix stuff
        # G = kernel_obj.gram_matrix(x1, x2)
        # H = G - np.mean(G, axis=2, keepdims=True) - np.mean(G, axis=1, keepdims=True) + np.mean(G, axis=(1, 2), keepdims=True)  # centered Gram matrix
        # nVn = n * H.mean(axis=(1, 2))

        # Kxx = kernel_obj.gram_matrix(x1, x1)
        # Kyy = kernel_obj.gram_matrix(x2, x2)
        # Kxy = kernel_obj.gram_matrix(x1, x2)
        # mmds[kernel_obj.__repr__()][n] = np.abs(np.mean(Kxx, axis=(1, 2)) + np.mean(Kyy, axis=(1, 2)) - 2 * np.mean(Kxy, axis=(1, 2)))

        # Gs[kernel_obj.__repr__()][n] = G
        # Hs[kernel_obj.__repr__()][n] = H
        # nVns[kernel_obj.__repr__()][n] = nVn

    t3 = time()

    for aname in alpha_df.columns:
        a = alpha_df[aname].values.reshape(-1, 1)

        for kernelname, K in Ks.items():
            K = K[n]
            out.append({
                "n": n,
                "realization": aname,
                "norm": kernelname,
                # "value": np.sqrt(np.diag(a.T @ K @ a))[0],  # TODO re-vectorize
                "value": np.sqrt((a.T @ K @ a)).flatten()[0]
            })

        # test different norms
        for p in ps:
            out.append({
                "n": n,
                "realization": aname,
                "norm": p,
                "value": np.sum(np.abs(a)**p)**(1/p),
            })

    t4 = time()

    # print(t1 - t0, t2 - t1, t3 - t2, t4 - t3)


out = pd.DataFrame(out)



# %% 1. Plots

x = "n"
y = "value"
hue = "norm"

plt.close("all")

fig, ax = plt.subplots(figsize=(15, 10))

df_plot_p = out.query("norm in @ps")
df_plot_K = out.query("norm not in @ps")
df_plot_K = (df_plot_K.groupby(["n", "norm"]).value.quantile([0.05, 0.1, 0.5, 0.9, 0.95]).reset_index()
             .rename(columns={"level_2": "quantile"}))

sns.lineplot(data=out, x=x, y=y, hue=hue, ax=ax, errorbar=None, legend=False, palette="Blues")
sns.lineplot(data=df_plot_K, x=x, y=y, ax=ax, hue="quantile", estimator="mean", errorbar="sd", palette="dark:salmon_r")

ax.set_xscale("log")
ax.set_yscale("log")

sns.despine()
plt.tight_layout()
wm = plt.get_current_fig_manager()
wm.window.state('zoomed')
fig.show()

# %% 2. CLT-like result: # Does the distribution of sqrt(n)||.||_K resemble an infinite sum of chi^2?


dfK = out.query("norm not in @ps").reset_index(drop=True)
dfK.loc[:, "limit"] = pd.Series(np.sqrt(dfK["n"]) * dfK["value"], index=dfK.index)

plt.close("all")

fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex="all", sharey="all")

fig.suptitle("Distribution of sqrt(n) * MMD(p_n, q_n)")

axes = axes.flatten()
for ax, n in zip(axes, [256, 1024, 4096, 16384]):
    ax.set_title(f"n={n}")
    sns.histplot(data=dfK.query("n == @n"), x="limit", ax=ax)

    ax.set_xlabel("")

sns.despine()
plt.tight_layout()
wm = plt.get_current_fig_manager()
wm.window.state('zoomed')
plt.show()

# %% 2b. With proper limiting distribution

rng = np.random.default_rng(144444)
repchi = rep

dfK = out.query("norm not in @ps").reset_index(drop=True)

# --- Refomat nVn in a pandas dataframe

# dfV = []
# for kernel, nVns_ in nVns.items():
#     for n, nVn in nVns_.items():
#         tmp = pd.DataFrame(columns=["n", "nVn"], index=range(len(nVn)))
#         tmp["n"] = n
#         tmp["nVn"] = nVn
#         # tmp["kernel"] = kernel
#         dfV.append(tmp)
# dfV = pd.concat(dfV, axis=0)

# dfV = []
# for kernel, mmds_ in mmds.items():
#     for n, mmds__ in mmds_.items():
#         tmp = pd.DataFrame(columns=["n", "nVn"], index=range(len(mmds__)))
#         tmp["n"] = n
#         tmp["nVn"] =  n * mmds__
#         # tmp["kernel"] = kernel
#         dfV.append(tmp)
# dfV = pd.concat(dfV, axis=0)

dfV = dfK.copy().rename(columns={"value": "nVn"}).drop(columns=["realization", "norm"])
dfV["nVn"] = dfV["nVn"]**2 * dfV["n"]

# TODO shuffle the rankings: are the eigenvalues of K the same?



universe_base, pmf_base = sample.get_universe_pmf()
pmf_df_base = pd.Series(pmf_base, index=universe_base, name="universe")

ub = universe_base.copy()

kernel_obj = kernels[0]     # TODO Assuming one kernel
match kernel_obj.vectorized_input_format:
    case "adjmat":
        x = ub.to_adjmat_array((na, na))
        # x1 = ms1.to_adjacency_matrices(na=na)
        # x2 = ms2.to_adjacency_matrices(na=na)
    case "vector":
        x = ub.to_rank_vector_matrix()
        # x1 = ms1.to_rank_vectors()
        # x2 = ms2.to_rank_vectors()
    case _:
        raise ValueError(f"Unsupported input format for vectorized kernel: {kernel_obj.vectorized_input_format}")
K = kernel_obj.gram_matrix(x, x)

np.random.shuffle(ub)

kernel_obj = kernels[0]     # TODO Assuming one kernel
match kernel_obj.vectorized_input_format:
    case "adjmat":
        x = ub.to_adjmat_array((na, na))
        # x1 = ms1.to_adjacency_matrices(na=na)
        # x2 = ms2.to_adjacency_matrices(na=na)
    case "vector":
        x = ub.to_rank_vector_matrix()
        # x1 = ms1.to_rank_vectors()
        # x2 = ms2.to_rank_vectors()
    case _:
        raise ValueError(f"Unsupported input format for vectorized kernel: {kernel_obj.vectorized_input_format}")
K2 = kernel_obj.gram_matrix(x, x)

eigv, _ = np.linalg.eig(K)
eigv2, _ = np.linalg.eig(K2)

print(np.linalg.norm(eigv-eigv2))

K = K2

#%%

# Get the eigenvalues of the operator
n = max(ns)
m = K.shape[0]

Tk = K  @ np.diag(pmf_df_base.values)
# eigvals, eigvecs = np.linalg.eig(Tk)  # imaginary part ~ 10e-19
# eigvals = np.real(eigvals)
eigvals, eigvecs = np.linalg.eigh(Tk)

# have an independent sample on every row
chisq = rng.chisquare(df=1, size=(m, repchi))

# multiply every column by the corresponding eigenvalue
chisq_eig = np.diag(eigvals) @ chisq

# sum along the rows
chisq_limit = chisq_eig.sum(axis=0)

chiout = []
for ic, c in enumerate(chisq_limit):
    chiout.append({
        "n": "limit",
        "nVn": c
    })
chiout = pd.DataFrame(chiout)

# --- Plotting

ns_plot = ns[1:] + ["limit"]

dfV = pd.concat([dfV, chiout], axis=0)
dfplot = dfV.query("n in @ns_plot")

plt.close("all")

fig, ax = plt.subplots()
fig.suptitle(f"Limiting distribution of n * MMD_n^2\nna={na}, kernel={repr(kernel_obj)}\nreplace={replace}, disjoint={disjoint}")

# sns.histplot(dfplot, x="nVn", hue="n", ax=ax, stat="probability")
sns.ecdfplot(dfplot, x="nVn", hue="n", ax=ax, palette="rocket")

ax.set_xlim(-0.5, 3)

sns.despine()
plt.tight_layout()
wm = plt.get_current_fig_manager()
wm.window.state('zoomed')
plt.show()
#
# plt.close("all")
#
# fig, axes = plt.subplots(2, 4, sharex="all", sharey="all" )
# fig.suptitle("Limiting distribution of nVn (avg of centered kernel)")
# axes = axes.flatten()
#
# for (ax, n) in zip(axes, ns_plot):
#     ax.set_title(f"n = {n}")
#     # sns.histplot(dfplot.query("n == @n"), x="nVn", ax=ax, stat="probability", binwidth=0.05)
#     sns.ecdfplot(dfplot.query("n == @n"), x="nVn", ax=ax)
#
# sns.despine()
# plt.tight_layout()
# wm = plt.get_current_fig_manager()
# wm.window.state('zoomed')
# plt.show()

