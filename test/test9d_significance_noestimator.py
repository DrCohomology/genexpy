"""

Hypothesis 1: there is no monotonic relation between generalizability and significance.
    test with actual generalizability values, no estimators



Generalizability is result trustworthiness
    will we observe similar results on another sample?
Significance is strength of results on a given sample
    Does our hypothesis hold on this sample?
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
import warnings

from collections import Counter, defaultdict
from functools import reduce
from importlib import reload
from itertools import product
from pathlib import Path
from rich.progress import Progress
from scipy.special import binom
from scipy.stats import binomtest, entropy, friedmanchisquare, spearmanr, wilcoxon
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
from typing import Union

from genexpy import lower_bounds as gu
from genexpy import kernels as ku
from genexpy import rankings_utils as ru
from genexpy import mmd as mmd
from genexpy import probability_distributions as du

warnings.filterwarnings("ignore")

def get_number_rankings(na):
    """
    Number of rankings of na alternatives (with ties), https://oeis.org/A000670
    """
    return du.get_unique_ranks_distribution(na, normalized=False).sum().astype(int)


def get_number_samples(n: int, k: int) -> int:
    """
    Number of samples of size k from a pool of choices of size n (with replacement).
    """
    return int(binom(n + k -1, k))


def universe_tied_rankings(na: int) -> ru.UniverseAM:
    """
    Generate all 'numbers' in base na, up to na digits.
    Reduce these 'numbers' and remove duplicates.
    These are the rankings.
    """
    universe, _ = ru.SampleAM.from_rank_vector_matrix(np.array(list(product(range(na), repeat=na))).T).get_universe_pmf()
    return universe


def samples_pmf(universe: ru.UniverseAM, n: int, universe_pmf: np.ndarray = None):
    """
    Return all possible samples of size n from universe.
    universe_pmf contains the probability of observing a given element of the universe, as returned by universe_tied_rankings.
        It is used to modify 'pmf' and must have the same length as 'universe'
    """

    if universe_pmf is None:
        universe_pmf = np.ones_like(universe, dtype=float)

    assert len(universe) == len(universe_pmf)

    u_pmf_dict = dict(zip(universe, universe_pmf))

    # return 1, 1, u_pmf_dict

    counter = Counter(tuple(sorted(x)) for x in product(universe, repeat=n))
    # samples = np.array(list(counter.keys()))
    samples = list(counter.keys())
    pmf = np.fromiter(counter.values(), float)

    pmfs = []
    for sample, p in zip(samples, pmf):
        ps = 1
        for r in sample:
            ps *= u_pmf_dict[r]
        pmfs.append(p * ps)
    pmfs = np.array(pmfs)

    assert np.isclose(pmfs.sum(), 1.0), pmfs.sum()

    # if normalized:
    #     pmf /= pmf.sum()

    return samples, pmfs


def dense2avg(ranking):
    """ Fabian's code to convert a ranking from dense to average representation --- for Friedman"""
    cts = Counter(ranking)
    results = {}

    min_rank, max_rank = (min(ranking), max(ranking))

    next_int = 1
    for i in range(min_rank, max_rank + 1):
        freq = cts[i]
        acc = 0
        for j in range(next_int, next_int + freq):
            acc += next_int
            next_int += 1
        results.update({i: acc / freq})

    return np.array([results[i] for i in ranking])


def dense2avg_matrix(rv):
    return np.array([dense2avg(r) for r in rv.T]).T


# distr = du.UniformDistribution(na=10, ties=True)
OUTPUT_DIR = Path(os.getcwd()) / "test" / "outputs" / f"test9d_significance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %% 1. Actual values of generalizability from an empirical distribution


seeds = range(20)
n_samples = 20

nmin = 2
nmax = 6
ns = range(nmin, nmax+1)

na = 3
dfout = []
base_distrs = {}
sampled_universes = {}
for seed in tqdm(list(seeds), desc="Distribution: "):
    rng = np.random.default_rng(seed)

    base_universe, _ = ru.SampleAM.from_rank_vector_matrix(np.array(list(product(range(na), repeat=na))).T).get_universe_pmf()
    # distr = du.PMFDistribution(universe=base_universe, pmf=np.array([0.3, 0.7, 0]))
    tokeep = 0.2
    base_pmf = np.zeros(len(base_universe))
    while len(np.nonzero(base_pmf)[0]) == 0:  # get a valid pmf
        base_pmf = np.maximum(rng.uniform(-1+tokeep, tokeep, len(base_universe)), 0)
    base_pmf = base_pmf / base_pmf.sum()
    distr = du.PMFDistribution(universe=base_universe, pmf=base_pmf, seed=seed + 1)

    base_distrs[seed] = distr

    out = []
    for i_sample in range(n_samples):

        # sample of experimental results
        sampled_universe = distr.sample(nmax)
        sampled_universes[(seed, i_sample)] = sampled_universe

        # now we are computing the exact generalizability for the sample distribution observed in the sample of experimental results
        universe, universe_pmf = sampled_universe.get_universe_pmf()
        universe_pmf /= universe_pmf.sum()

        for n in ns:
            samples, pmf = samples_pmf(universe, n, universe_pmf=universe_pmf)

            # precompute gram matrices
            grams = {}
            for s in samples:
                grams[s] = ku.square_gram_matrix(s, use_rv=False, kernel=kernel)

            # compute MMD
            for (s1, p1), (s2, p2) in list(product(zip(samples, pmf), repeat=2)):
                kxx = grams[s1]
                kyy = grams[s2]
                kxy = ku.gram_matrix(s1, s2, use_rv=False, kernel=kernel)

                out.append({
                    "na": na,
                    "n": n,
                    "seed": seed,
                    "i_sample": i_sample,
                    "mmd": np.sqrt(kxx.mean() + kyy.mean() - 2 * kxy.mean()),
                    "p_mmd": p1 * p2,
                    "kept_alternatives": len(np.nonzero(base_pmf)[0])
                })

    dftmp = pd.DataFrame(out).groupby(["na", "n", "seed", "i_sample", "mmd"])["p_mmd"].sum().reset_index().sort_values(by="mmd")
    dftmp["gen"] = dftmp["p_mmd"].cumsum()

    dfout.append(dftmp)

dfmmd = pd.concat(dfout, axis=0).reset_index(drop=True)
# dfmmd.to_parquet(OUTPUT_DIR / f"dfmmd_na={na}_pmf={distr.pmf}.parquet")
dfmmd.to_parquet(OUTPUT_DIR / f"dfmmd_na={na}_pmf=random.parquet")





#%% n - log(eps) relation

epss = np.linspace(0.0001, np.sqrt(2 * (1-1/np.e)), 1000)

out = []
for eps in epss:
    tmp = dfmmd.query("mmd <= @eps").groupby(["na", "n"])["gen"].max().reset_index()
    tmp["eps"] = eps
    out.append(tmp)

dfgen = pd.concat(out, axis=0).reset_index(drop=True).sort_values(["na", "n", "eps"])

#%%
pass
# df_ = dfgen.query("n in [2, 4, 8, 16]")
# df_ = dfgen.query("n in [5, 10, 20]")
df_ = dfgen

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

sns.lineplot(df_, ax=ax, x="eps", y="gen", hue="n", palette="rainbow")

fig.show()



#%% quantiles

alpha = 0.8

df_ = dfgen.query("gen >= @alpha").groupby(["na", "n", "gen"])["eps"].min().reset_index()
genmin = df_.groupby(["na", "n"])["gen"].min().reset_index()
df_ = pd.merge(df_, genmin, on=["na", "n"], how="left", suffixes=("", "_min"))
df_ = df_.query("gen == gen_min")


df_["logn"] = np.log(df_["n"])
df_["logeps"] = np.log(df_["eps"])

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

sns.scatterplot(df_, x="logeps", y="logn", hue="gen", ax=ax, palette="rainbow")


# for gen in df_["gen"].unique():
#     df__ = df_.query("gen == @gen")
#     sns.scatterplot(df__, x="logeps", y="logn", ax=ax, legend=False)

fig.show()





#%% 2. Significance is not conclusive: significant in different ways

"""
significance is not able to tell you if your sample is representative
significance does not tell you if your results will hold on other samples of data
take two samples, show that results are significant in both cases, show that the actual conclusions are, however, different
    in sample i encoder i is the best
"""
ndistrs = 100
nsamples = 100  # number of samples
pvals = [0.01, 0.05, 0.1, 0.5]     # pvalue for significance
na = 6  # also used to check significance of conover
tokeeps = np.array([5, 10, 20, 40]) / get_number_rankings(na)    # apprx fraction of rankings with non-zero probability of being drawn
sample_sizes = [5, 10, 20, 40, 80, 160]

out = []
out_distrs = {}
out_conover = {}
out_samples = {}
for seed in tqdm(list(range(ndistrs)), desc="Distribution: "):
    for ik, tokeep in enumerate(tokeeps):
        rng = np.random.default_rng(seed)
        base_universe = universe_tied_rankings(na=na)
        # base_pmf = np.maximum(rng.uniform(-1+tokeep, tokeep, len(base_universe)), 0)
        # base_pmf = base_pmf / base_pmf.sum()
        base_pmf = np.zeros(len(base_universe))

        while len(np.nonzero(base_pmf)[0]) == 0:  # get a valid pmf
            base_pmf = np.maximum(rng.uniform(-1 + tokeep, tokeep, len(base_universe)), 0)
        base_pmf = base_pmf / base_pmf.sum()

        distr = du.PMFDistribution(universe=base_universe, pmf=base_pmf, seed=seed * len(tokeeps) + ik + 1000)
        out_distrs[(seed, tokeep)] = distr

        for isa in list(range(nsamples)):
            for sas in sample_sizes:
                sample = distr.sample(sas)
                rv = sample.to_rank_vector_matrix()

                tmp_friedman = friedmanchisquare(*rv)[1]
                # tmp_best_avgrk = rv.mean(axis=1).argmin()  # only keeping one (if multiple are tied)
                tmp_best_avgrk = np.where(rv.sum(axis=1) == rv.sum(axis=1).min())[0]

                # regardless of whether friedman is significant, run conover-iman
                tmp_conover = sp.posthoc_conover_friedman(rv.T)  # transitive?

                for pval in pvals:
                    adj = np.array(tmp_conover <= pval).astype(int)  # not transitive adjacency matrix for "a != b"

                    # check if all/any of the best alternatives are significantly better than the others
                    # the best alternative (avg rank) is significantly best if conover is rejected for every other alternative
                    tmp_best_sig_all = (adj[tmp_best_avgrk].sum(axis=1) == na - len(tmp_best_avgrk)).all()
                    tmp_best_sig_any = (adj[tmp_best_avgrk].sum(axis=1) == na - len(tmp_best_avgrk)).any()

                    tmp = {
                        "distr": seed,
                        "size": sas,
                        "tokeep": tokeep,
                        "significance": pval,
                        "n_rankings": len(np.nonzero(base_pmf)[0]),
                        "friedman": tmp_friedman <= pval,
                        "best_avgrk": tuple(tmp_best_avgrk),  # best has lowest average rank
                        "conover_all": tmp_best_sig_all,
                        "conover_any": tmp_best_sig_any,
                    }

                    out.append(tmp)
                out_conover[(distr, isa, sas, tokeep)] = tmp_conover
                out_samples[(distr, isa, sas, tokeep)] = sample

df = pd.DataFrame(out)
df.to_parquet(OUTPUT_DIR / f"dfsig.parquet", index=False)

# %% 2a. get number of significant samples per distribution and best alternatives

df = pd.read_parquet(OUTPUT_DIR / f"dfsig.parquet")
df["best_avgrk"] = df["best_avgrk"].map(lambda x: tuple(x))
df_sig = df.query("friedman and conover_any")

pk = ["distr", "size", "tokeep", "significance", "best_avgrk"]

df_tmp1 = df.groupby(pk)[["n_rankings"]].count()  # distributions with significant samples. n_rankings can be replaced with any other column
df_tmp1["n_rankings"] = 0
df_tmp2 = df_sig.groupby(pk)[["n_rankings"]].count()  # all distributions

# number of significant samples per distribution and best alternative
df_numsig_da = (df_tmp1 + df_tmp2).fillna(0).reset_index().rename(columns={"n_rankings": "n_samples"})
# number of significant samples per distribution
df_numsig_d = df_numsig_da.groupby(pk[:-1])["n_samples"].sum().reset_index()
# number of significant best alternatives per distribution
tmp_func = lambda x: len(reduce(lambda x, y: set(x).union(y), x))  # merge the best alternatives and get their number
df_numalt_d = df_numsig_da.query("n_samples > 0").groupby(pk[:-1])["best_avgrk"].agg(tmp_func).reset_index().rename(columns={"best_avgrk": "n_best"})
# entropy of significant best alternatives (higher = more pathological). entropy considers as completely different the best alternatives (4, ) and (4, 1)
df_entropyalt_d = df_numsig_da.query("n_samples > 0").groupby(pk[:-1])["n_samples"].agg(lambda x: entropy(x)).reset_index().rename(columns={"best_avgrk": "entropy_best"})

# %%
"""
For every distribution, find the best alternative in terms of average rank, then compare to thesignificant and not samples. 
"""

out = []
for (idx, tokeep), distr in tqdm(list(out_distrs.items())):
    # get the true best alternative
    avgrk = np.sum(base_universe.to_rank_vector_matrix() * distr.pmf, axis=1)
    best_avgrk = np.where(avgrk == avgrk.min())[0][0]
    out.append({
        "distr": idx,
        "tokeep": tokeep,
        "true_best_avgrk": best_avgrk,
    })

#%%
df_truebest = pd.merge(pd.DataFrame(out), df, on=["distr", "tokeep"], how="left")
df_sig = df_truebest.query("significance == 0.05 and friedman and conover_any")
# df_faulty_sig = df_sig[df_sig["true_best_avgrk"].apply(lambda x: x in df_sig["best_avgrk"])]

faulty = []
for idx, x in df_sig.iterrows():
    if x["true_best_avgrk"] not in x["best_avgrk"]:
        faulty.append(idx)
df_faulty_sig = df_sig.loc[faulty]





# %% Plot

"""
For distribution and sample size, plot how many significant alternatives are found
Also, plot the 
"""

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r"""\usepackage{helvet}"""
mpl.rcParams["font.family"] = "Serif"
sns.set()
sns.set_style("white")
palette = "flare_r"
sns.set_palette("flare_r")

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 7))

axes = axes.flatten()

ax = axes[0]
# ax.set_title("Number of significant samples per distribution")
# sns.lineplot(data=sig_best, x="size", y="n_samples", hue="distr", style="distr", ax=ax, legend=False, palette="rainbow")
# sns.scatterplot(data=sig_best, x="size", y="n_samples", hue="distr", ax=ax, legend=True, palette="rainbow")
sns.boxplot(data=df_numsig_d, x="size", y="n_samples", ax=ax, native_scale=False, hue="significance", fill=True, palette="flare_r", legend=False)
ax.set_ylabel("No. of samples with significant results\nper distribution")


ax = axes[1]
# ax.set_title("Number of significant (at least in one sample) best alternatives per distribution")
sns.boxplot(data=df_numalt_d, x="size", y="n_best", ax=ax, native_scale=False, hue="significance", palette="flare_r")
ax.set_xlabel("sample size")
ax.set_ylabel("No. of significant best alternatives\nper distribution")

# ax = axes[2]
# ax.set_title("Fraction of distributions with more than one significant best alternative")
# df_plot = df_numalt_d.query("n_best > 1").groupby(["significance", "size"])["n_best"].count().reset_index()
# # sns.boxplot(data=df_plot, x="size", y="n_best", ax=ax, hue="significance", palette="deep")
# sns.lineplot(data=df_plot, x="size", y="n_best", ax=ax, hue="significance", palette="deep")
# ax.axhline(0)
# ax.set_ylabel("")

sns.despine()
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "significant_alternatives.png")
fig.savefig(OUTPUT_DIR / "significant_alternatives.pdf")
# fig.show()





#%% 2b. Can we detect "faulty" distributions with generalizability?
"""
A faulty distribution is one for which different samples yield different best alternatives. 
df_faulty contains the number of faulty (distribution, sample_size, significance) triplets 
find, within the faulty distributions, the worst one according to the entropy of the distribution of best alternatives 
    for instance, given samples from the same distribution: sample1 -> a1, sample2 -> a2, sample3 -> a1; then compute entropy(a1=2, a2=1)

"""




pval_max = 0.05
size_min = 40

df_faulty = df_numalt_d.query("significance <= @pval_max and "
                              "size >= @size_min and "
                              "n_best > 1")
df_pathological = df_faulty.query("n_best == @df_faulty.n_best.max()")

# distr_idx, tokeep = tuple(df_pathological.iloc[0][["distr", "tokeep"]].values)
# distr = out_distrs[distr_idx, tokeep]
# samples_pathological = [v for k, v in out_samples.items() if k[0] == distr]

df_faulty_da = pd.merge(df_numsig_da.query("n_samples > 0"), df_faulty, on=pk[:-1])
df_faulty_entr = df_faulty_da.groupby(pk[:-1])["n_samples"].agg(entropy).reset_index().rename(columns={"n_samples": "entropy"})
idx_maxentr = df_faulty_entr["entropy"].argmax()  # only four significant samples

idx_maxentr = 2
distr_idx, tokeep = df_faulty_entr.iloc[idx_maxentr][["distr", "tokeep"]]
distr = out_distrs[distr_idx, tokeep]
samples_pathological = [v for k, v in out_samples.items() if k[0] == distr]

# check number of best alternatives
# ba = [x["best_avgrk"] for x in sigs]

# share of samples contradicting the best alternative in the distribution

#%% 2b1. Compute generalizability to check if we could have known the different sample behavior

# keep reasonable sized samples
samples_pathological = [s for s in samples_pathological if len(s) == 40]

SEED = 14444444
DISJOINT = True
REPLACE = False
kernel = ku.jaccard_kernel
kernelargs = {"k": 1}
pval = 0.05
mmdrep = 1000

out = []
tests_f = []
tests_ci = []
for sample in tqdm(samples_pathological, desc="Computing the MMD"):
    mmds = {
            n: mmd.subsample_mmd_distribution(
                sample, subsample_size=n, rep=mmdrep, use_rv=True, use_key=False,
                seed=SEED, disjoint=DISJOINT, replace=REPLACE, kernel=kernel, **kernelargs
            )
            for n in range(2, min(sample.nv // 2 + 1, 20), 2)
        }

    rv = sample.to_rank_vector_matrix()
    tmp_friedman = friedmanchisquare(*rv)[1]
    tmp_conover = sp.posthoc_conover_friedman(rv.T)

    tmp_best_avgrk = np.where(rv.sum(axis=1) == rv.sum(axis=1).min())[0]

    adj = np.array(tmp_conover <= pval).astype(int)  # not transitive adjacency matrix for "a != b"

    # check if all/any of the best alternatives are significantly better than the others
    # the best alternative (avg rank) is significantly best if conover is rejected for every other alternative
    tmp_best_sig_all = (adj[tmp_best_avgrk].sum(axis=1) == na - len(tmp_best_avgrk)).all()
    tmp_best_sig_any = (adj[tmp_best_avgrk].sum(axis=1) == na - len(tmp_best_avgrk)).any()

    out.append({
        "mmd": mmds,
        "best_avgrk": tuple(tmp_best_avgrk),
        "friedman": tmp_friedman <= pval,
        "conover_all": tmp_best_sig_all,
        "conover_any": tmp_best_sig_any,
    })
    tests_f.append(tmp_friedman)
    tests_ci.append(tmp_conover)

#%% 2b1a. Take only the significant samples, check generalizability

"""
hopefully, the generalizability of every sample is the same. we are sample-independent (unlike statistical tests). 
    we are not fooled by weird samples. 
"""

alphas = [0.8, 0.9, 0.95, 0.99]
epss = [np.sqrt(2*(1-x)) for x in [0.99, 0.95, 0.9, 0.8]]

# --- SIGNIFICANT
sigs = [s for s in out if s["friedman"] and s["conover_any"]]

tmp = []
for idx, s in enumerate(sigs):
    for n, mmdn_ in s["mmd"].items():
        for alpha in alphas:
            tmp.append({
                "sample": idx,
                "size": n,
                "alpha": alpha,
                "q": np.quantile(mmdn_, alpha)
            })
df_qvar_sig = pd.DataFrame(tmp)

tmp = []
for idx, s in enumerate(sigs):
    for n, mmdn_ in s["mmd"].items():
        for eps in epss:
            tmp.append({
                "sample": idx,
                "size": n,
                "eps": eps,
                "gen": np.mean(mmdn_ <= eps)
            })
df_genvar_sig = pd.DataFrame(tmp)

# --- ALL
samples = out.copy()

tmp = []
for idx, s in enumerate(samples):
    for n, mmdn_ in s["mmd"].items():
        for alpha in alphas:
            tmp.append({
                "sample": idx,
                "size": n,
                "alpha": alpha,
                "q": np.quantile(mmdn_, alpha)
            })
df_qvar = pd.DataFrame(tmp)

tmp = []
for idx, s in enumerate(samples):
    for n, mmdn_ in s["mmd"].items():
        for eps in epss:
            tmp.append({
                "sample": idx,
                "size": n,
                "eps": eps,
                "gen": np.mean(mmdn_ <= eps)
            })
df_genvar = pd.DataFrame(tmp)


#%%

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

ax = axes[0]
ax.set_title("Estimated quantiles, significant")
sns.boxplot(df_qvar_sig, x="size", y="q", hue="alpha", ax=ax, native_scale=True)

ax = axes[1]
ax.set_title("Estimated gen, significant")
sns.boxplot(df_genvar_sig, x="size", y="gen", hue="eps", ax=ax, native_scale=True)

ax = axes[2]
ax.set_title("Estimated quantiles")
sns.boxplot(df_qvar, x="size", y="q", hue="alpha", ax=ax, native_scale=True)

ax = axes[3]
ax.set_title("Estimated gen")
sns.boxplot(df_genvar, x="size", y="gen", hue="eps", ax=ax, native_scale=True)


plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"variance_generalizability_mmdrep={mmdrep}.png")






