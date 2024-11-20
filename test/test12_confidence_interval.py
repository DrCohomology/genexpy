"""
1. What confidence interval can we place on our estimate of generalizability?
    1.1. Estimate generalizability of a single sample
    1.2. Estimate gneralizability of a distribution (can I get the theoretical value for simple distributions?)

2. what can we say on the generalizability for 2n if we know that for n?
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
from pathlib import Path
from scipy.special import binom
from scipy.stats import binomtest, friedmanchisquare, spearmanr, wilcoxon
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

kernel = ku.mallows_kernel
distr = du.UniformDistribution(na=10, ties=True)

OUTPUT_DIR = Path(os.getcwd()) / "test" / "outputs" / f"test12_{kernel.__name__}_uniform"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#%% 1.1. Estimate generalizability of a single sample
pass

# universe = ru.SampleAM.from_rank_vector_matrix(np.array([[0, 1], [1, 0]]).T)
# pmf = np.array([5, 5])
# distr = du.PMFDistribution(universe=universe, pmf=pmf, ties=False)

repnum = 100
N = 1000
sample = distr.sample(N)

mmds = []
for rep in tqdm(list(range(repnum))):
    mmd_ = mmd.subsample_mmd_distribution(sample, subsample_size=5, rep=100, seed=rep, kernel=kernel)
    mmds.append(mmd_)

#%% 1.1.a Plot

eps = np.linspace(0, np.sqrt(2), 100)
gens = np.array([mmd.generalizability(mmd_, eps) for mmd_ in mmds]).T

dfg = pd.DataFrame(gens)
dfg["eps"] = eps
dfg = dfg.melt(id_vars="eps", var_name="rep", value_name="gen")





fig, axes = plt.subplots(1, 2)

ax = axes[0]
# for gen_ in gens.T:
#     sns.lineplot(x=eps, y=gen_, ax=ax, color="black")
# sns.boxplot(dfg, x="eps", y="gen", ax=ax, native_scale=True)

sns.lineplot(dfg, x="eps", y="gen", errorbar="sd", ax=ax)

ax = axes[1]
# sns.lineplot(gens.mean(axis=1), ax=ax, c="red")
sns.lineplot(x=eps, y=gens.var(axis=1), ax=ax, c="blue")

plt.tight_layout()

plt.show()



#%% 1.2 Theoretical derivation

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


def samples_pmf(universe: ru.UniverseAM, n: int, normalized=True):
    """
    Return all possible samples of size n from universe.
    """
    counter = Counter(tuple(sorted(x)) for x in product(universe, repeat=n))
    # samples = np.array(list(counter.keys()))
    samples = list(counter.keys())
    pmf = np.fromiter(counter.values(), float)
    if normalized:
        pmf /= pmf.sum()
    return samples, pmf


kernel = ku.mallows_kernel

na = 3
ns = range(2, 5)
dfout = []
for n in ns:
    universe, _ = ru.SampleAM.from_rank_vector_matrix(np.array(list(product(range(na), repeat=na))).T).get_universe_pmf()
    samples, pmf = samples_pmf(universe, n, normalized=True)

    # precompute gram matrices
    out = []
    grams = {}
    for s in tqdm(samples, desc="Building kxx"):
        grams[s] = ku.square_gram_matrix(s, use_rv=False, kernel=kernel)

    for (s1, p1), (s2, p2) in tqdm(list(product(zip(samples, pmf), repeat=2)), desc="MMD"):
        kxx = grams[s1]
        kyy = grams[s2]
        kxy = ku.gram_matrix(s1, s2, use_rv=False, kernel=kernel)

        out.append({
            "na": na,
            "n": n,
            "mmd": np.sqrt(kxx.mean() + kyy.mean() - 2 * kxy.mean()),
            "p_mmd": p1 * p2,
        })

    dftmp = pd.DataFrame(out).groupby(["na", "n", "mmd"])["p_mmd"].sum().reset_index().sort_values(by="mmd")
    dftmp["gen"] = dftmp["p_mmd"].cumsum()

    dfout.append(dftmp)

dfmmd = pd.concat(dfout, axis=0).reset_index(drop=True)
dfmmd.to_parquet(OUTPUT_DIR / f"dfmmd_na={na}_uniform.parquet")

# %% get gen for many epss (slow-ish) OUTDATED

dfmmd = pd.read_parquet(OUTPUT_DIR / "dfmmd_na=2_uniform_MEGA.parquet")
ns_ = dfmmd["n"].unique()

epss = np.linspace(0, np.sqrt(2), 1000)
epss = np.logspace(-5, np.log10(np.sqrt(2)), 1000)
out = []
for (n, na), idx in dfmmd.groupby(["n", "na"]).groups.items():
    for eps in epss:
        out.append({
            "n": n,
            "na": na,
            "eps": eps,
            "gen": dfmmd.iloc[idx].query("mmd <= @eps").sort_values("mmd")["p_mmd"].sum()
        })

dfgen = pd.DataFrame(out)

# %% quantiles
pass
# qs = np.linspace(0, 1, 1000)
# # qs = np.linspace(0, 1, 6)
# dfgb = dfmmd.groupby(["n", "na"])["mmd"]
# out = []
# for q in qs:
#     tmp = dfgb.agg(lambda x: np.quantile(x, q=q, method="lower")).reset_index()  # on a uniform sample!!!!
#     tmp["q"] = q
#     out.append(tmp)
# dfq = pd.concat(out, axis=0).reset_index(drop=True)

# %% gen plots


palette = sns.color_palette("rocket", n_colors=len(ns_))

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()

ax = axes[0]
ax.set_title("Generalizability vs eps, N")
sns.lineplot(dfgen.query("n in @ns_"), x="eps", y="gen", ax=ax, hue="n", palette=palette, legend=False)
# sns.scatterplot(dfq.query("n in @ns_"), x="mmd", y="q", ax=ax, hue="n", palette="rainbow")

ax = axes[1]
ax.set_title("Generalizability vs eps, N")
sns.scatterplot(dfgen, x="n", y="gen", ax=ax, hue="eps", palette="rainbow", legend=False)

ax = axes[2]
ax.set_title("Quantile behavior")
dfplot2 = dfgen.query("n in @ns_").groupby(["n", "na", "gen"]).max().reset_index()
# sns.lineplot(dfplot2, x="gen", y="eps", ax=ax, hue="n", palette=palette, legend=False)
sns.scatterplot(dfplot2, x="gen", y="eps", ax=ax, hue="n", palette=palette, legend=False)
# sns.lineplot(dfq.query("n in @ns_"), x="q", y="mmd", ax=ax, hue="n", palette="rainbow")
# ax.set_ylabel("quantile")
# ax.set_xscale("log")
# ax.set_yscale("log")

ax = axes[3]
ax.set_title("Quantile behavior")
dfplot2 = dfgen.groupby(["eps", "na", "gen"]).max().reset_index()
# sns.lineplot(dfplot2, x="gen", y="eps", ax=ax, hue="n", palette=palette)
sns.scatterplot(dfplot2, x="gen", y="n", ax=ax, hue="eps", palette="rainbow", legend=False)
# sns.lineplot(dfq.query("n in @ns_"), x="q", y="mmd", ax=ax, hue="n", palette="rainbow")
# ax.set_ylabel("quantile")

plt.tight_layout()
fig.show()

#%% investigate kink patterns
pass

# # try to isolate the patterns by "upscaling" around 1
# dfplot2 = dfgen.query("n in @ns_").groupby(["n", "na", "gen"]).max().reset_index()
#
# ks = range(5)
# for k in ks:
#     if k == 0:
#         dfplot2["leps0"] = dfplot2["eps"]
#         continue
#     dfplot2[f"leps{k}"] = (np.abs(dfplot2[f"leps{k-1}"])) ** (1/20)
#
# dfplot2["gen"] = np.exp(dfplot2["gen"])
#
# fig, axes = plt.subplots(2, len(ks)//2 , sharex="all")
#
# for k, ax in enumerate(axes.flatten()):
#     ax.set_title(f"k = {k}")
#     sns.scatterplot(dfplot2, x="gen", y=f"leps{k}", ax=ax, hue="n", palette=palette, legend=False, size=0.001)

dfmmd = pd.read_parquet(OUTPUT_DIR / "dfmmd_na=2_uniform_MEGA.parquet")
ns_ = dfmmd["n"].unique()
na = dfmmd["na"].max()

# --- dfplot2, dfplot3
# dfplot2 = dfgen.query("n in @ns_").groupby(["n", "na", "gen"]).min().reset_index()
# dfplot22 = dfgen.query("n in @ns_").groupby(["n", "na", "gen"]).max().reset_index()
# dfplot2 = pd.merge(dfplot2, dfplot22, on=["n", "na", "gen"], suffixes=("_min", "_max"))
#
# out = []
# for n in dfplot2["n"].unique():
#     tmp = dfplot2.query("n == @n").sort_values("gen").reset_index(drop=True)
#     tmp["kink_nr"] = np.arange(len(tmp))
#     out.append(tmp)
# dfplot3 = pd.concat(out, axis=0).reset_index(drop=True)
# dfplot4 = pd.concat(out, axis=1).reset_index(drop=True)

# --- dfplot5 directly from dfmmd
out = []
for n in dfmmd["n"].unique():
    tmp = dfmmd.query("n == @n").sort_values("gen").reset_index(drop=True)
    tmp["kink_nr"] = np.arange(len(tmp)) + 1
    out.append(tmp)
dfplot5 = pd.concat(out, axis=0).reset_index(drop=True)

palette = "rainbow"

# dfplot5["x"] = 1 - 1 / dfplot5["gen"]
dfplot5["x"] = dfplot5["gen"]
# dfplot5["x"] = 1 / (1 - dfplot5["gen"] + 0.1)

kinks = range(0, 1001, 10)
dfplot5 = dfplot5.query("kink_nr in @kinks")
nmax = 20
dfplot5 = dfplot5.query("n <= @nmax")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

ax = axes[0]

# margin = 0.02
# ax.set_title(f"ns: {dfplot5['n'].unique()}")
# ax.set_ylim(-margin, 1 + margin)
# ax.set_xlim(-margin, np.sqrt(2)+margin)

sns.scatterplot(dfplot5, x="mmd", y="gen", ax=ax, hue="kink_nr", style="n", palette=palette, legend=False)
# sns.scatterplot(dfplot5, x="n", y="eps_min", ax=ax, hue="gen", palette=palette, legend=False)

ax.vlines(0.1, 0, 1, ls="--", color="grey")
ax.hlines(0.9, 0, np.sqrt(2), ls="--", color="grey")

# ax.set_yscale("log")

ax = axes[1]

for kink in kinks:
    df_ = dfplot5.query("kink_nr == @kink")
    sns.lineplot(df_, x="mmd", y="gen", ax=ax)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / f"kinks_na={na}_uniform_nmax={nmax}.pdf")
fig.show()


#%%

fig, ax = plt.subplots()

sns.lineplot(dfplot3.groupby("n")["kink_nr"].max().reset_index(), x="n", y="kink_nr", ax=ax)
sns.lineplot(dfmmd.groupby("n")["mmd"].nunique().reset_index(), x="n", y="mmd", ax=ax)
sns.lineplot(dfgen.groupby("n")["gen"].nunique().reset_index(), x="n", y="gen", ax=ax)

fig.show()



#%%

import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"

fig = px.scatter(dfplot3, x="x", y="eps_max", color="n")
fig.show()

#%% plotly visualization

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

# dfplotly = dfgen.melt(id_vars="n", var_name="eps", value_name="gen").pivot(index="eps", columns="n", values="gen").iloc[::25]
dfplotly = dfgen.pivot(index="eps", columns="n", values="gen")

# fig = go.Figure(data=go.Surface(z=z, x=x, y=y))
# fig = px.scatter_3d(dfplotg, x="eps", y="n", z="gen", color=z, log_y=True)
# fig.show()

fig = go.Figure(data=[go.Surface(z=dfplotly.values, x=dfplotly.columns, y=dfplotly.index,
                                 coloraxis="coloraxis2",
                                 contours={
                                     "x": {"show": True},
                                     "y": {"show": True},
                                     "z": {"show": True},
                                 },
                                 opacity=0.6)],
                layout=go.Layout(template="plotly_white"))
fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
fig.show()

OUTPUT_DIR = Path(os.getcwd()) / "test" / "outputs" / f"test12_{kernel.__name__}_uniform"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig.write_html(OUTPUT_DIR / "gen_theoretical_na=2_smalln.html")


# %%




# number of alternatives, corresponding Fubini number, and number of samples
na = 2
nfa = get_number_rankings(na)
nv = 2
ns = get_number_samples(nfa, nv)


unique_symbols = du.get_unique_ranks_distribution(na, normalized=False)


kernel = ku.jaccard_kernel



