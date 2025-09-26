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
ns = [2**m for m in range(2, 14)]
N = 16 * max(ns)
na = 5
ties = True
disjoint = True
replace = False
ps = np.linspace(2, 10, 11)
rng = np.random.default_rng(seed)

# kernel_obj = kcu.JaccardKernel(k=1)
kernel_obj = kcu.MallowsKernel(nu="auto")

distr = du.UniformDistribution(na=na, ties=ties, seed=seed+1)
# distr = du.MDegenerateDistribution(na=na, m=2, ties=ties, seed=seed)

sample = distr.sample(N)
universe_base, pmf_base = sample.get_universe_pmf()
pmf_df_base = pd.Series(pmf_base, index=universe_base, name="universe")

match kernel_obj.vectorized_input_format:
    case "adjmat":
        x = universe_base.to_adjmat_array((na, na))
    case "vector":
        x = universe_base.to_rank_vector_matrix()
    case _:
        raise ValueError(f"Unsupported input format for vectorized kernel: {kernel_obj.vectorized_input_format}")
K = kernel_obj.gram_matrix(x, x)

out = []
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

    t2 = time()

    alpha = alpha_df.values

    t3 = time()

    for aname in alpha_df.columns:
        a = alpha_df[aname].values.reshape(-1, 1)

        out.append({
            "n": n,
            "realization": aname,
            "norm": repr(kernel_obj),
            # "value": np.sqrt(np.diag(a.T @ K @ a))[0],  # TODO re-vectorize
            "value": np.sqrt((a.T @ K @ a)).flatten()[0]
        })

        # test different norms
        # for p in ps:
        #     out.append({
        #         "n": n,
        #         "realization": aname,
        #         "norm": p,
        #         "value": np.sum(np.abs(a)**p)**(1/p),
        #     })

    t4 = time()

    # print(t1 - t0, t2 - t1, t3 - t2, t4 - t3)

out = pd.DataFrame(out)


# %% 1b. Plots for chi square

repchi = rep * 10
rng = np.random.default_rng(189)

dfK = out.query("norm not in @ps").reset_index(drop=True)
dfV = dfK.copy().rename(columns={"value": "nVn"}).drop(columns=["realization", "norm"])
dfV["nVn"] = dfV["nVn"]**2 # * dfV["n"]
m = len(pmf_df_base)

# Get the eigenvalues of the operator
# H = K - np.mean(K, axis=0, keepdims=True) - np.mean(K, axis=1, keepdims=True) + np.mean(K, axis=(0, 1), keepdims=True)
C = np.eye(m) - np.ones((m, m)) / m
H = C @ K @ C

Tk = 2 * H @ np.diag(pmf_df_base.values)
# Tk = H
# eigvals = np.sqrt(np.maximum(np.round(np.linalg.eigvalsh(Tk), 1), 0))
# eigvals = (np.maximum(np.linalg.eigvalsh(H), 0)) ** 1/32
eigvals = np.maximum(np.linalg.eigvalsh(Tk), 0)
# eigvals = 2/m * eigvals

# Get the limiting distribution (weighted sum of chi squares)
chisq = rng.chisquare(df=1, size=(Tk.shape[0], repchi))    # have an independent sample on every row
chisq_eig = np.diag(eigvals) @ chisq    # multiply every column by the corresponding eigenvalue
chisq_limit = chisq_eig.sum(axis=0)     # sum along the rows
# chisq_limit = chisq_limit / dfV["n"].max()
chisq_limit = chisq_limit / 64


# Reformat into a dataframe
normalout = []
for  ic, c in enumerate(chisq_limit):
    normalout.append({
         "n": "limit",
         "nVn": c
    })
dfV = pd.concat([dfV, pd.DataFrame(normalout)], axis=0)

# --- Plotting

ns_plot = ns[::] + ["limit"]
dfplot = dfV.query("n in @ns_plot")
dfplot["n"] = dfplot["n"].astype(str)

plt.close("all")

fig, ax = plt.subplots()
fig.suptitle(f"Limiting distribution of n * MMD_n^2\n"
              f"na={na}, m={m}, kernel={repr(kernel_obj)}\n")

 # sns.histplot(dfplot, x="nVn", hue="n", ax=ax, stat="probability")
#  sns.ecdfplot(dfplot, x="nVn", hue="n", ax=ax, palette="rocket")
sns.ecdfplot(dfplot.query("n != 'limit'"), x="nVn", hue="n", ax=ax, palette="rocket")
sns.ecdfplot(dfplot.query("n == 'limit'"), x="nVn", label="limit", color="lime", ls="--")
# ax.set _xlim(-0.5, 3)

ax.set_xscale("log")

sns.despine()
plt.tight_layout()
wm = plt.get_current_fig_manager()
wm.window.state('zoomed')
plt.show ()

 # %% 1c. Eigenvalues of Tk

eigvals, eigvecs = np.linalg.eig(K)
eigvals = np.real(eigvals)
eigvecs = np.real(eigvecs)


# Get the eigenvalues of the operator
H = K - np.mean(K, axis=0, keepdims=True) - np.mean(K, axis=1, keepdims=True) + np.mean(K, axis=(0, 1), keepdims=True)
Tk = H @ np.diag(pmf_df_base.values)
# Tk = H
eigvals = np.linalg.eigvalsh(Tk)

 # %% 1d. https://pages.stat.wisc.edu/~shao/stat709/stat709-18.pdf

repchi = rep * 100

dfK = out.query("norm not in @ps").reset_index(drop=True)
dfV = dfK.copy().rename(columns={"value": "nVn"}).drop(columns=["realization", "norm"])
dfV["nVn"] = dfV["nVn"] * np.sqrt(dfV["n"])
m = len(pmf_df_base)

v = np.var((K - K.mean()).mean(axis=1))

normal_limit = rng.normal(loc=0, scale=((4 * v) ** 2 ), size=repchi)

# Reformat into a dataframe
normalout = []
for ic, c in enumerate(normal_limit):
    normalout.append({
        "n": "limit",
        "nVn": c
    })
dfV = pd.concat([dfV, pd.DataFrame(normalout)], axis=0)

# --- Plotting

ns_plot = ns[::2] + ["limit"]
dfplot = dfV.query("n in @ns_plot")

plt.close("all")

fig, ax = plt.subplots()
fig.suptitle(f"Limiting distribution of n * MMD_n^2\n"
             f"na={na}, kernel={repr(kernel_obj)}\n")

# sns.histplot(dfplot, x="nVn", hue="n", ax=ax, stat="probability")
sns.ecdfplot(dfplot, x="nVn", hue="n", ax=ax)

# ax.set_xlim(-0.5, 3)

sns.despine()
plt.tight_layout()
wm = plt.get_current_fig_manager()
wm.window.state('zoomed')
plt.show()



#  %% 2. Test eigenvalues for different permutations

seeds = range(500)
repchi = 1000

# sample = distr.sample(N)
 # universe_base, pmf_base = sample.get_universe_pmf()
#  pmf_df_base = pd.Series(pmf_base, index=universe_base, name="universe")
m =  len(pmf_df_base)

eigs = np.empty(shape=(len(seeds), m))   # spectra of are stored as rows
chis_df = []
chis = []
for ( i, seed) in tqdm(list(enumerate(seeds))):
    pmf_df = pmf_df_base.sample(frac=1, random_state=seed)

    universe = ru.SampleAM(pmf_df.index.values)
    pmf = pmf_df.values

    match kernel_obj.vectorized_input_format:
         case "adjmat":
             x = universe.to_adjmat_array((na, na))
         case "vector":
             x = universe.to_rank_vector_matrix()
         case _:
             raise ValueError(f"Unsupported input format for vectorized kernel: {kernel_obj.vectorized_input_format}")
    K = kernel_obj.gram_matrix(x, x)
    Tk = K  @  np.diag(pmf)
    eigvals, eigvecs = np.linalg.eig(Tk)  # imaginary part ~ 10e-19
    eigvals = np.real(eigvals)

    eigs[i] = eigvals

    # Get the limiting distribution (weighted sum of chi squares, always the same)
    rng = np.random.default_rng(10)
    chisq = rng.chisquare(df=1, size=(m, repchi))    # have an independent sample on every row
    chisq_eig = np.diag(eigvals) @ chisq    # multiply every column by the corresponding eigenvalue
    chisq_limit = chisq_eig.sum(axis=0)     # sum along the rows

    chis.append(chisq_limit)

    normalout = []
    for ic, c in enumerate(chisq_limit):
     normalout.append({
         "seed": seed,
         "n": "limit",
         "nVn": c
    })
    chis_df.append(pd.DataFrame(normalout))
chis_df = pd.concat(chis_df, axis=0)

plt.close("all")
fig, ax = plt.subplots()
fig. suptitle(f"Limiting distribution for different shuffles of the universe")

# sns.histplot(dfplot, x="nVn", hue="n", ax=ax, stat="probability")
sns.ecdfplot(chis_df, x="nVn", hue="seed", ax=ax, legend=False)

 ax.set_xlim(-0.5, 3)

 sns.despine()
p lt.tight_layout()
wm  = plt.get_current_fig_manager()
wm.wi ndow.state('zoomed')
plt.sho w()

 # %% 3. Convergence as in Gretton (2012) Thm 12 (for the U-statistic)

 repchi = rep * 10
rn g = np.random.default_rng(189)

dfK = out.query("norm not in @ps").reset_index(drop=True)
 dfV = dfK.copy().rename(columns={"value": "nVn"}).drop(columns=["realization", "norm"])
df V["nVn"] = dfV["nVn"]**2 * 2 * dfV["n"]
m =  len(pmf_df_base)

 # Get the eigenvalues of the operator
H  = K - np.mean(K, axis=0, keepdims=True) - np.mean(K, axis=1, keepdims=True) + np.mean(K, axis=(0, 1), keepdims=True)
# H2  = K - K.mean()
# H =  K
Tk = H  @ np.diag(pmf_df_base.values)
eigvals,  eigvecs = np.linalg.eig(Tk)  # imaginary part ~ 10e-19
eigvals = n p.real(eigvals)

# Get the limiting distribution (weighted sum of [sth])
n ormal1 = rng.standard_normal(size=(Tk.shape[0], repchi))
nor mal2 = rng.standard_normal(size=(Tk.shape[0], repchi))
norma l = (np.sqrt(2) * normal1 - np.sqrt(2)*normal2)**2 - 4
normal_ limit_eig = np.diag(eigvals) @ normal
normal_l imit = normal_limit_eig.sum(axis=0)

# Reformat into a dataframe
 normalout = []
fo r ic, c in enumerate(normal_limit):
    n ormalout.append({
         "n": "limit",
        " nVn": c
    })
 dfV = pd.concat([dfV, pd.DataFrame(normalout)], axis=0)

# --- Plotting

ns_plot = ns[::2] + ["limit"]
 dfplot = dfV.query("n in @ns_plot")

 plt.close("all")

 fig, ax = plt.subplots()
fi g.suptitle(f"Limiting distribution of n * MMD_n^2\n"
              f"na={na}, kernel={repr(kernel_obj)}\n")

# sns.histplot(dfplot, x="nVn", hue="n", ax=ax, stat="probability")
 sns.ecdfplot(dfplot, x="nVn", hue="n", ax=ax)

# ax.set_xlim(-0.5, 3)

sns.despine()
p lt.tight_layout()
wm  = plt.get_current_fig_manager()
wm.wi ndow.state('zoomed')
plt.sho w()

#%% 4. GPT

import numpy as np
 import matplotlib.pyplot as plt
fr om scipy.stats import chi2, norm

 # --- Step 1: Generate data under the null hypothesis P = Q ---
np .random.seed(42)
n =  200
X = np .random.normal(0, 1, size=(n, 1))
Y = np.r andom.normal(0, 1, size=(n, 1))

Z = np.vstack([X, Y])  # Combined for kernel computation

# --- Step 2: Define the kernel and compute kernel matrix ---
d ef rbf_kernel(x, y, sigma=1.0):
     sq_dist = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
    r eturn np.exp(-sq_dist / (2 * sigma ** 2))

 K = rbf_kernel(Z, Z)  # Full 2n x 2n kernel matrix

# --- Step 3: Center the kernel matrix ---
 n_total = 2 * n
H  = np.eye(n_total) - np.ones((n_total, n_total)) / n_total
K_cen tered = H @ K @ H

 # --- Step 4: Compute biased MMD² statistic ---
K_ XX = K[:n, :n]
K_YY  = K[n:, n:]
K_XY  = K[:n, n:]

 mmd2_biased = (
     K_XX.sum() / (n ** 2) +
     K_YY.sum() / (n ** 2) -
    2  * K_XY.sum() / (n ** 2)
)

 print(f"Biased MMD²: {mmd2_biased:.5f}")

# --- Step 5: Estimate null distribution (simulate asymptotic distribution) ---

 # Extract eigenvalues from centered full kernel matrix
ei gvals = np.linalg.eigvalsh(K_centered / n_total)  # Normalize
eigva ls = np.clip(eigvals, 0, None)  # Ensure non-negativity

# Simulate distribution: sum λ_i Z_i²
 n_sim = 1000
Z_ samples = np.random.normal(0, 1, size=(n_sim, len(eigvals)))
simu lated_null = (eigvals * Z_samples**2).sum(axis=1)

 # Scale back to match n * MMD²
si mulated_null_scaled = simulated_null / n_total

 # --- Step 6: Compare the statistic to the null ---
pl t.hist(simulated_null_scaled, bins=50, density=True, alpha=0.5, label="Simulated null")
plt .axvline(mmd2_biased, color='red', linestyle='--', label="Observed MMD²")
plt.t itle("Asymptotic Distribution of Biased MMD² under $H_0$")
plt.xla bel("MMD²")
plt.ylabel ("Density")
plt.legend()
 plt.grid(True)
pl t.show()
