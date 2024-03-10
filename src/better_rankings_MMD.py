"""
- use np.random!        ✔️
- Sampling rankings:    ✔️
    - Uniformly         ✔️
    - Degenerately      ✔️
    - Bidigenerately    ✔️
    - Center            ✔️
    - Ball              ✔️
    - AntiBall          ✔️
- Types of Rankings:
    - Reflexive Antisymmetric Transitive ✔️
    - Total Transitive
    - Reflexive Transitive
- Sampling
    - TODO Implement lazy sampling, without needing to first get all rankings. Draw an item at random, keep it with given
        probability.

- MMD ✔️
- Plots ✔️
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import defaultdict
from importlib import reload
from numpy.random import Generator
from pathlib import Path
from time import time
from tqdm.notebook import tqdm
from typing import Iterable

import src.generalizability_utils as gu
import src.kernel_utils as ku
import src.probability_distributions as prob
import src.rankings_utils as ru

DATA_DIR = Path("./data/benchmark_encoders")
FIGURES_DIR = Path("./Figures")
mpl.use("TkAgg")

# %% 1. Compare restricted encoder_rankings with the other distributions

df = pd.read_parquet(DATA_DIR / "results.parquet")

# get a subset
encoders = ["OHE", "DTEM5", "DE", "CBE", "TE", "SE"]
models = ["LR"]
tunings = ["no"]
df = df.query("encoder in @encoders and model in @models and tuning in @tunings").reset_index(drop=True)

# resolve ties at random
df["cv_score"] = df["cv_score"] + np.random.default_rng(125).normal(0, 0.001, len(df["cv_score"]))

# get rankings
rf = ru.get_rankings_from_df(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                             target="cv_score", lower_is_better=False, impute_missing=True)

# fill missing values, all with the same number. !!! THEY ARE TIED.
rf = rf.fillna(rf.max())

# get synthetic distributions
na = len(encoders)
nv = len(rf.columns)
shape = (na, na)

rankings = ru.universe_untied_rankings(na)

center = rankings[0]
radius = 0.3
nu = 0.5
elements = ru.UniverseAM(rankings[:2])

samples = {
    "uniform": prob.UniformDistribution(rankings).sample(nv),
    "degenerate": prob.DegenerateDistribution(rankings).sample(nv, element=rankings[0]),
    "2degenerate_close": prob.MDegenerateDistribution(rankings).sample(nv, m=2, elements=rankings[:2]),
    "2degenerate_opposite": prob.MDegenerateDistribution(rankings).sample(nv, m=2,
                                                                          elements=ru.UniverseAM(np.array([rankings[0],
                                                                                                           rankings[
                                                                                                               -1]],
                                                                                                          dtype=object))),
    "5degenerate": prob.MDegenerateDistribution(rankings).sample(nv, m=5),
    f"center_nu={nu}": prob.CenterProbabilityDistribution(rankings).sample(nv, center=center, kernel=ku.mallows_kernel,
                                                                           nu=nu),
    f"ball_nu={nu}": prob.BallProbabilityDistribution(rankings).sample(nv, center=center, radius=radius,
                                                                       kernel=ku.mallows_kernel, nu=nu),
    f"antiball_nu={nu}": prob.BallProbabilityDistribution(rankings).sample(nv, center=center, radius=radius,
                                                                           kind="antiball", kernel=ku.mallows_kernel,
                                                                           nu=nu),
    "encoders": ru.SampleAM.from_rank_function_dataframe(rf)
}
variances = {distr_name: ku.var(sample, ku.mallows_kernel, nu=0.5) for distr_name, sample in samples.items()}
""" !!!
The mallows_kernel of two rankings of encoders is so low (10e-200) that it is virtually impossible to compare them. 
The result is almost exactly the same as if we were using the trivial kernel.
It's also interesting to see that MMD does NOT go to 0 if te kernel used is the trivial kernel. Instead, it converges
    to a value which seems fixed at 2^{subsample_size-1}, at least for the encoders BECAUSE THERE ARE NO OVERLAPPINGS
    BETWEEN SUBSAMPLES, as we have a lot of rankings to choose from. 
    It does not hold for the other (smaller) samples as we are getting the same ranking in both subsamples. 
These values could serve as "base neutral values" for MMD, meaning that anything above that indicates contrasting 
    rankings (negative kernel) --- actually, ther kernel CANNOT be negative, meaning that thesw values ARE a hard 
    upper bound for MMD (i.e., the trivial kernel).
Actually, MMD > value means that there is extremely high within-sample kernel (kxx.mean() is high ) and low 
    between-sample kernel (kxy.mea() is low). 
"""

# %% 1a1. Plot distribution of rankings

plt.close("all")

fig, axes = plt.subplots(3, int(np.ceil(len(samples) / 3)), figsize=(6, 4), sharex="all", sharey="all")
fig.suptitle("Distributions of rankings (logistic regression, no tuning)")
with sns.axes_style("white"):
    for ax, (distr_name, sample) in zip(axes.flatten(), samples.items()):
        ax = sns.histplot(sample, stat="probability", discrete=True, ax=ax, color="r")
        # ax = sns.kdeplot(sample, ax=ax)
        ax.set(xticks=range(len(rankings)), xticklabels=[])
        ax.tick_params(bottom=False)
        ax.set_title(f"{distr_name}, variance={variances[distr_name]:.02f}")
        ax.set_label("Ranking")
        ax.set_yscale("log")

    sns.despine(left=True, trim=True)

plt.tight_layout()
plt.show(block=True)

# %% 1b. Compute MMD

plt.close("all")

rep = 100
subsample_size = 25
subsample_mmd = {
    distr_name: ku.subsample_mmd_distribution(sample, subsample_size, rep=rep, kernel=ku.jaccard_kernel, k=1)
    for distr_name, sample in samples.items()}

# %% 1b1. Plot MMD

fig, axes = plt.subplots(3, int(np.ceil(len(subsample_mmd) / 3)), figsize=(6, 4), sharey="all", sharex="all")
fig.suptitle("Distributions of MMD^2 (logistic regression, no tuning) with mallows_kernel(nu=0.5)")
# fig.suptitle("Distributions of MMD (logistic regression. no tuning) with jaccard_kernel(k=1)")
with sns.axes_style("white"):
    for ax, (distr_name, sample) in zip(axes.flatten(), subsample_mmd.items()):
        if distr_name in ["degenerate", "2degenerate_close"]:
            bins = [0, 0.005]
            binwidth = None
        else:
            bins = "auto"
            binwidth = 0.005
        ax = sns.histplot(sample, stat="probability", ax=ax, color="r", binwidth=binwidth, bins=bins)
        ax.set_ylabel("P(MMD^2)")
        ax.set_title(f"{distr_name}, variance={variances[distr_name]:.02f}")
        ax.set_ylim((0, 0.5))

    sns.despine()

plt.tight_layout()
plt.show(block=True)

# %% 2. Encoders with top-k kernels

df = pd.read_parquet(DATA_DIR / "results.parquet")

# get a subset
# encoders = ["OHE", "DTEM5", "DE", "CBE", "TE", "SE"]
# models = ["LR"]
# tunings = ["no"]
# df = df.query("encoder in @encoders and model in @models and tuning in @tunings").reset_index(drop=True)

# resolve ties at random
df["cv_score"] = df["cv_score"] + np.random.default_rng(125).normal(0, 0.001, len(df["cv_score"]))

# get rankings
rf = ru.get_rankings_from_df(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                             target="cv_score",
                             lower_is_better=False, impute_missing=True)

# fill missing values, all with the same number. !!! THEY ARE TIED.
rf = rf.fillna(rf.max())

encoder_rankings = ru.SampleAM.from_rank_function_dataframe(rf)

mmd_encoders = [
    ku.subsample_mmd_distribution(encoder_rankings, subsample_size=20, kernel=ku.jaccard_kernel, k=k, rep=100)
    for k in range(1, 33)]

mmde = pd.DataFrame(mmd_encoders).reset_index(names="k").melt(id_vars="k", value_name="MMD")
mmde["k"] = mmde["k"] + 1
mallows = ku.subsample_mmd_distribution(encoder_rankings, subsample_size=20, kernel=ku.mallows_kernel, nu=0.001,
                                        rep=100)
trivial = ku.subsample_mmd_distribution(encoder_rankings, subsample_size=20, kernel=ku.trivial_kernel, rep=100)

fig, ax = plt.subplots(1, 1)
fig.suptitle("Average MMD for top-k jaccard kernel")

sns.lineplot(mmde, x="k", y="MMD", errorbar="sd", ax=ax)
ax.axhline(mallows.mean(), c="purple", label="mallows")
ax.axhline(trivial.mean(), c="silver", label="trivial")
ax.set_ylabel("Average MMD")

plt.legend()
plt.tight_layout()
plt.show(block=True)

""" 
Very interesting: initia drop in standard deviation is followed by plateau at k~6.
MMD with trivial kernel is 0.01 (same reps, same subsample size)
MMD with mallows kernel(nu=10e-3) is 0.037
"""

# %% 2c. Encoders with mallows kernel, varying nu

xs = np.logspace(-7, 3, 11)

mmd_encoders = [
    ku.subsample_mmd_distribution(encoder_rankings, subsample_size=20, kernel=ku.mallows_kernel, nu=nu, rep=100)
    for nu in xs]

mmde = pd.DataFrame(mmd_encoders, index=xs).reset_index(names="nu").melt(id_vars="nu", value_name="MMD")
trivial = ku.subsample_mmd_distribution(encoder_rankings, subsample_size=20, kernel=ku.trivial_kernel, rep=100)

fig, ax = plt.subplots(1, 1)
fig.suptitle("Average MMD for varying nu of Mallows kernel")

sns.lineplot(mmde, x="nu", y="MMD", errorbar="sd", ax=ax)
ax.axhline(trivial.mean(), c="silver", ls=":", label="trivial")
ax.set_ylabel("Average MMD")
ax.set_xscale("log")

plt.legend()
plt.tight_layout()
plt.show(block=True)

# %% 3. Robustness of results

DATA_DIR = Path("./data/benchmark_encoders")
df = pd.read_parquet(DATA_DIR / "results.parquet")

df = df.query("tuning == 'no' and model == 'DTC' ").reset_index(drop=True)

rep = 10
noisy_samples = []
for seed in tqdm(range(rep)):
    # resolve ties at random
    df[f"cv_score_{seed}"] = df["cv_score"] + np.random.default_rng(seed).normal(0, 0.001, len(df["cv_score"]))

    # get rankings
    rf = ru.get_rankings_from_df(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                                 target=f"cv_score_{seed}", lower_is_better=False, impute_missing=True)

    # fill missing values, all with the same number. !!! THEY ARE TIED.
    rf = rf.fillna(rf.max())
    noisy_samples.append(ru.SampleAM.from_rank_function_dataframe(rf))

# %% 3a. Distribution of MMD for noisy results

kernel = ku.jaccard_kernel

noisy_mmd = []
for i1, s1 in tqdm(list(enumerate(noisy_samples))):
    for i2, s2 in enumerate(noisy_samples):
        if i2 >= i1:
            break
        noisy_mmd.append(ku.mmd(s1, s2, kernel=ku.jaccard_kernel, k=5))

# %% 3a1. plot distribution

fig, ax = plt.subplots(1, 1)

sns.histplot(noisy_mmd, ax=ax)

plt.tight_layout()
plt.show(block=True)

# %% 4a. Lower bounds of generalizability, how tight are they? Compute variance

"""
From Variance-Aware Estimation of Kernel Mean Embedding, https://arxiv.org/abs/2210.06672.
"""
reload(ku)

df = pd.read_parquet(DATA_DIR / "results.parquet")

# get a subset
# encoders = ["OHE", "DTEM5", "DE", "CBE", "TE", "SE"]
# models = ["LR"]
# tunings = ["no"]
# df = df.query("encoder in @encoders and model in @models and tuning in @tunings").reset_index(drop=True)

# resolve ties at random
# df["cv_score"] = df["cv_score"] + np.random.default_rng(125).normal(0, 0.001, len(df["cv_score"]))

# get rankings
rf = ru.get_rankings_from_df(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                             target="cv_score",
                             lower_is_better=False, impute_missing=True)

# fill missing values, all with the same number. !!! THEY ARE TIED.
rf = rf.fillna(rf.max())

encoder_rankings = ru.SampleAM.from_rank_function_dataframe(rf)

t1 = time()

v = ku.var(encoder_rankings, ku.jaccard_kernel, k=1)
mmd_encoders = {
    n: ku.subsample_mmd_distribution(encoder_rankings, subsample_size=n, rep=100, use_njit=True, use_key=False,
                                     kernel=ku.jaccard_kernel_rf, k=1)
    for n in (5, 10, 20, 40, 80, 160)  # , 320, 640, 1000)
}
print(f"DONE in {time() - t1}")

# %% 4b. Lower bounds. How tight is the lower bound?

reload(gu)

colors = [plt.get_cmap('magma')(i / len(mmd_encoders)) for i in range(len(mmd_encoders))]

epss = np.logspace(-2, 0, 1000)
for color, (n, mmde) in zip(colors, mmd_encoders.items()):
    gene = [gu.generalizability(mmde, eps) for eps in epss]
    lbe = [gu.generalizability_lowerbound(eps, n, kbar=1, v=v) for eps in epss]
    lbe2 = [gu.mmd_lowerbound(eps, n, kbar=1) for eps in epss]

    plt.plot(epss, gene, c=color, label=f"n={n}")
    plt.plot(epss, lbe, c=color, ls='--', )
    plt.plot(epss, lbe2, c=color, ls=':', )

plt.title("Generalizability for varying n.")
plt.xlabel("eps")
plt.ylabel("generalizability")
plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.show(block=True)
# %% 5a. Varying m, fix n

"""
What happens if we fix n but shrink the universe?
"""

df = pd.read_parquet(DATA_DIR / "results.parquet")
df["cv_score"] = df["cv_score"] + np.random.default_rng(125).normal(0, 0.001, len(df["cv_score"]))
rf = ru.get_rankings_from_df(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                             target="cv_score",
                             lower_is_better=False, impute_missing=True)
rf = rf.fillna(rf.max())
encoder_rankings = ru.SampleAM.from_rank_function_dataframe(rf)
subsamples = {m: ru.SampleAM(np.random.default_rng(1444).choice(encoder_rankings, m))
              for m in [100, 200, 400, 800, 1600, 2000]}
# for m in [40, 80, 160, 320, 640, 1280, 2000]}

n = 50
rep = 1000

mmd_subsamples = {
    m: ku.subsample_mmd_distribution(subsample, subsample_size=n, rep=rep, kernel=ku.jaccard_kernel_rf, k=2,
                                     use_njit=True)
    for m, subsample in subsamples.items()
}

print("DONE")

# %% 5a1. Varying m, fix n, plots

colors = [plt.get_cmap('magma')(i / len(mmd_subsamples)) for i in range(len(mmd_subsamples))]

epss = np.logspace(-2, 0, 1000)
for color, (m, mmds) in zip(colors, mmd_subsamples.items()):
    gene = [gu.generalizability(mmds, eps) for eps in epss]
    plt.plot(epss, gene, c=color, label=f"m={m}")

plt.title(f"Generalizability for varying m, n={n}, {rep} repetitions.")
plt.xlabel("eps")
plt.ylabel("generalizability")
plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.show(block=True)

# %% 5b. Varying m, n = m/2

"""
What happens if we fix n but shrink the universe?
"""

df = pd.read_parquet(DATA_DIR / "results.parquet")
df["cv_score"] = df["cv_score"] + np.random.default_rng(125).normal(0, 0.001, len(df["cv_score"]))
rf = ru.get_rankings_from_df(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                             target="cv_score",
                             lower_is_better=False, impute_missing=True)
rf = rf.fillna(rf.max())
encoder_rankings = ru.SampleAM.from_rank_function_dataframe(rf)
subsamples = {m: ru.SampleAM(np.random.default_rng(1444).choice(encoder_rankings, m))
              for m in [40, 80, 160, 320, 640]}

rep = 100

mmd_subsamples = {
    m: ku.subsample_mmd_distribution(subsample, subsample_size=int(m / 2), rep=rep, kernel=ku.jaccard_kernel_rf, k=2,
                                     use_njit=True)
    for m, subsample in subsamples.items()
}

print("DONE")

# %% 5b1. Varying m, n=m/2, plots

colors = [plt.get_cmap('magma')(i / len(mmd_subsamples)) for i in range(len(mmd_subsamples))]

epss = np.logspace(-3, 0, 1000)
for color, (m, mmds) in zip(colors, mmd_subsamples.items()):
    gene = [gu.generalizability(mmds, eps) for eps in epss]
    plt.plot(epss, gene, c=color, label=f"m={m}")

plt.title(f"Generalizability for varying m, n=m/2, {rep} repetitions.")
plt.xlabel("eps")
plt.ylabel("generalizability")
plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.show(block=True)

# %% 5c. repetitions for fixed m, to see variance

"""
What happens if we subsample from the universe a lot of times?
"""

df = pd.read_parquet(DATA_DIR / "results.parquet")
df["cv_score"] = df["cv_score"] + np.random.default_rng(125).normal(0, 0.001, len(df["cv_score"]))
rf = ru.get_rankings_from_df(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                             target="cv_score",
                             lower_is_better=False, impute_missing=True)
rf = rf.fillna(rf.max())
encoder_rankings = ru.SampleAM.from_rank_function_dataframe(rf)

mrep = 10
subsamples_ = {m: [ru.SampleAM(np.random.default_rng(1444 + it).choice(encoder_rankings, m)) for it in range(mrep)]
               for m in [100, 1000, 2000]}

n = 20
t1 = time()
rep = 1000
mmd_subsamples_ = {
    m: [ku.subsample_mmd_distribution(subsample, subsample_size=n, rep=rep, kernel=ku.jaccard_kernel_rf, k=2,
                                      use_njit=True) for subsample in subsample_]
    for m, subsample_ in subsamples_.items()
}

print(f"DONE in {time() - t1}")

# %% 5c1. Plots for repetitions with fixed m

from collections import defaultdict

epss = np.logspace(-3, 0, 1000)
gene_ = defaultdict(lambda: [])
for m, mmds_ in mmd_subsamples_.items():
    for mmds in mmds_:
        gene_[m].append([gu.generalizability(mmds, eps) for eps in epss])

cols = [f"sample_{i}" for i in range(mrep)]
df_ = pd.DataFrame.from_dict(gene_, orient="index", columns=cols) \
    .explode(cols).reset_index(names="m").melt(id_vars="m", var_name="sample", value_name="generalizability")
df_["eps"] = list(epss) * 30  # TODO find a better solution
# %% 5c1a. Load dataset. TEMP
mrep = 10
epss = np.logspace(-3, 0, 1000)

df_ = pd.read_csv("TEMPORARY FILE GEN SUBSAMPLES.csv")

colors = [plt.get_cmap('magma')(i / mrep) for i in range(mrep)]

fig, axes = plt.subplots(1, 3, sharex=True)
for m, ax in zip(df_["m"].unique(), axes):
    df__ = df_.query("m == @m")
    std_ = df__.groupby("eps")["generalizability"].std().values

    sns.lineplot(df__, x="eps", y="generalizability", ax=ax, errorbar="sd", color="blue")
    ax.plot(epss, std_, c="red", label="std")

    ax.set_title(f"m={m}\nmax std={std_.max():.03f}")
    ax.set_xlim((df_.query("generalizability == 0")["eps"].max(), df_.query("generalizability == 1")["eps"].min()))
    ax.set_xscale("log")
    ax.set_xlabel("eps")
    ax.set_ylabel("generalizability")

fig.suptitle(f"Generalizability for 10 samples with fixed m, n=20, 1000 repetitions.")
plt.legend()
plt.tight_layout()
plt.show(block=True)

# %% 5d. What if the rankings are smaller?

"""
What happens if we fix n but shrink the universe?
"""

encs = ["OHE", "TE", "DE", "DTEM5"]
df = pd.read_parquet(DATA_DIR / "results.parquet").query("encoder in @encs").reset_index(drop=True)
df["cv_score"] = df["cv_score"] + np.random.default_rng(125).normal(0, 0.001, len(df["cv_score"]))
rf = ru.get_rankings_from_df(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                             target="cv_score",
                             lower_is_better=False, impute_missing=True)
rf = rf.fillna(rf.max())
encoder_rankings = ru.SampleAM.from_rank_function_dataframe(rf)

mrep = 10
subsamples_ = {m: [ru.SampleAM(np.random.default_rng(1444 + it).choice(encoder_rankings, m)) for it in range(mrep)]
               for m in [100, 1000, 2000]}

n = 20
t1 = time()
rep = 1000
mmd_subsamples_ = {
    m: [ku.subsample_mmd_distribution(subsample, subsample_size=n, rep=rep, kernel=ku.jaccard_kernel_rf, k=2,
                                      use_njit=True) for subsample in subsample_]
    for m, subsample_ in subsamples_.items()
}

"""it took twice as long than with the full rankings. weird"""
print(f"DONE in {time() - t1}")

# %% 5d1. Plot smaller rankings

from collections import defaultdict

epss = np.logspace(-3, 0, 1000)
gene_ = defaultdict(lambda: [])
for m, mmds_ in mmd_subsamples_.items():
    for mmds in mmds_:
        gene_[m].append([gu.generalizability(mmds, eps) for eps in epss])

cols = [f"sample_{i}" for i in range(mrep)]
df_ = pd.DataFrame.from_dict(gene_, orient="index", columns=cols) \
    .explode(cols).reset_index(names="m").melt(id_vars="m", var_name="sample", value_name="generalizability")
df_["eps"] = list(epss) * 30  # TODO find a better solution

colors = [plt.get_cmap('magma')(i / mrep) for i in range(mrep)]

fig, axes = plt.subplots(1, 3, sharex=True)
for m, ax in zip(df_["m"].unique(), axes):
    df__ = df_.query("m == @m")
    std_ = df__.groupby("eps")["generalizability"].std().values

    sns.lineplot(df__, x="eps", y="generalizability", ax=ax, errorbar="sd", color="blue")
    ax.plot(epss, std_, c="red", label="std")

    ax.set_title(f"m={m}\nmax std={std_.max():.03f}")
    ax.set_xlim((df_.query("generalizability == 0")["eps"].max(), df_.query("generalizability == 1")["eps"].min()))
    ax.set_xscale("log")
    ax.set_xlabel("eps")
    ax.set_ylabel("generalizability")

fig.suptitle(f"Generalizability of small rankings for 10 samples with fixed m, n=20, 1000 repetitions.")
plt.legend()
plt.tight_layout()
plt.show(block=True)

# %% 6 Compute generalizaiblity for LR and DT

df = pd.read_parquet(DATA_DIR / "results.parquet")
# resolve ties at random
df["cv_score"] = df["cv_score"] + np.random.default_rng(125).normal(0, 0.001, len(df["cv_score"]))
# get a subset
models = ["LR", "DTC"]

mmd_encoders_model = {}
for model in tqdm(models):
    df_ = df.query("model == @model").reset_index(drop=True)

    # get rankings
    rf = ru.get_rankings_from_df(df_, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                                 target="cv_score",
                                 lower_is_better=False, impute_missing=True)

    # fill missing values, all with the same number. !!! THEY ARE TIED.
    rf = rf.fillna(rf.max())
    encoder_rankings = ru.SampleAM.from_rank_function_dataframe(rf)

    mmd_encoders_model[model] = {
        n: ku.subsample_mmd_distribution(encoder_rankings, subsample_size=n, rep=1000, kernel=ku.jaccard_kernel, k=2)
        for n in (5, 10, 20, 40, 80, 160)
    }

# %% 6a Plot for different models

"""
The results: Overall, DTC shows better generalizable results than LR. 
LR is better than DTC for small epsilon. Interpretation: if you are very strict on what "similar results" means 
    (small eps), LR will be better. If instead you fix the generalizability you want to achieve (alpha), DTC is better.
"""

from collections import defaultdict

epss = np.logspace(-3, 0, 1000)

ys = defaultdict(lambda: dict())
for model, mmde_model in mmd_encoders_model.items():
    for n, mmde in mmde_model.items():
        ys[model][n] = [gu.generalizability(mmde, eps) for eps in epss]

dfp = pd.DataFrame(ys).explode(["LR", "DTC"]).reset_index(names="n") \
    .melt(id_vars=["n"], var_name="model", value_name="generalizability")
dfp["eps"] = list(epss) * int(len(dfp) / len(epss))

sns.lineplot(dfp, x="eps", y="generalizability", hue="n", style="model", palette="dark:salmon")

plt.title("Generalizability for varying n.")
plt.xscale("log")
plt.tight_layout()
plt.show(block=True)

# %% 6b Quantiles for different models

"""
Ideally, we would want to observe a straight downwrad slope line in log(n) and log(eps). 
Straight meaning that the "distance" between quantiles is constant, making it easier to crack the pattern. 
What we do observe is that the lines are almost straight but not quite. This could be due to random effects. 
Especially for the 99th percentile (most interesting), deviation from straight line is more evident. 
"""

epss = np.logspace(-3, 0, 1000)
quantiles = [0.01, 0.05, 0.5, 0.95, 0.99]

qs = defaultdict(lambda: dict())
for model, mmde_model in mmd_encoders_model.items():
    for n, mmde in mmde_model.items():
        qs[model][n] = np.quantile(mmde, quantiles)

dfp = pd.DataFrame(qs).explode(["LR", "DTC"]).reset_index(names="n") \
    .melt(id_vars=["n"], var_name="model", value_name="eps")
dfp["quantile"] = quantiles * int(len(dfp) / len(quantiles))

sns.lineplot(dfp, x="n", y="eps", hue="quantile", style="model", palette="dark:salmon")

plt.title("Quantiles of generalizability for varying n.")
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.show(block=True)

# %% 7. Random sample from datasets and not from anything. Generalizability in the sample of datasets.

df = pd.read_parquet(DATA_DIR / "results.parquet")
df = df.query("tuning == 'full' and scoring != 'AUC'").reset_index(drop=True)

rf = ru.get_rankings_from_df(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                             target="cv_score",
                             lower_is_better=False, impute_missing=True)
rf = rf.fillna(rf.max())
encoder_rankings = ru.SampleAM.from_rank_function_dataframe(rf)
encoder_rankings.set_key(rf.columns.get_level_values(0))

s = len(rf.columns)
sd = rf.columns.get_level_values(0).value_counts().max()
sms = (rf.columns.get_level_values(1) + rf.columns.get_level_values(3)).value_counts().max()

t1 = time()
mmd_encoders_multi = {
    "random_dataset": {
        sd * nd: ku.subsample_mmd_distribution(encoder_rankings.set_key(rf.columns.get_level_values(0)),
                                               subsample_size=nd, rep=100, use_njit=True, use_key=True,
                                               kernel=ku.jaccard_kernel_rf, k=2)
        for nd in [3, 6, 15]  # every dataset appears 12 times -> max 15
    },
    "random_modelscoring": {
        sms * nms: ku.subsample_mmd_distribution(encoder_rankings.set_key(rf.columns.get_level_values(1) +
                                                                          rf.columns.get_level_values(3)),
                                                 subsample_size=nms, rep=100, use_njit=True, use_key=True,
                                                 kernel=ku.jaccard_kernel_rf, k=2)
        for nms in [1, 2, 6]  # every combination appears 30 times -> max 6
    },
    "random": {
        n: ku.subsample_mmd_distribution(encoder_rankings, subsample_size=n, rep=100, use_njit=True, use_key=False,
                                         kernel=ku.jaccard_kernel_rf, k=2)
        for n in [30, 36, 60, 72, 180]  # every index appears once, max is 360/2 = 180
    }
}

# %% 7a. PLot

epss = np.logspace(-3, 0, 1000)

ys = defaultdict(lambda: dict())
for factor, mmde_factor in mmd_encoders_multi.items():
    for n, mmde in mmde_factor.items():
        ys[factor][n] = [gu.generalizability(mmde, eps) for eps in epss]

dfp = pd.DataFrame(ys)
dfp1 = dfp[["random_dataset"]].dropna().explode("random_dataset").reset_index(names="n")
dfp2 = dfp[["random_modelscoring"]].dropna().explode("random_modelscoring").reset_index(names="n")
dfp3 = dfp[["random"]].dropna().explode("random").reset_index(names="n")

# put them back together
dfp2.index = dfp2.index + 2000
dfp3["random_dataset"] = [np.nan] * len(dfp3)
dfp3["random_modelscoring"] = [np.nan] * len(dfp3)
dfp3.loc[dfp3["n"].isin(dfp1["n"].unique()), "random_dataset"] = dfp1["random_dataset"]
dfp3.loc[dfp3["n"].isin(dfp2["n"].unique()), "random_modelscoring"] = dfp2["random_modelscoring"]

dfp = dfp3.melt(id_vars=["n"], var_name="sample_from", value_name="generalizability")

dfp["eps"] = list(epss) * int(len(dfp) / len(epss))

fig, ax = plt.subplots()
sns.lineplot(dfp, x="eps", y="generalizability", hue="sample_from", style="n", ax=ax)

ax.set_title("Generalizability for varying n=number of rankings of encoders. Tuning=no, scoring!=AUC.\n"
             f"The number of sampled datasets is n/{sd}, the number of sampled (model+scoring) is n/{sms}.")
ax.set_xscale("log")
plt.tight_layout()
plt.show(block=True)
# %% 8. Cross-validated VS non cross-validated results

dfcv = pd.read_parquet(DATA_DIR / "results.parquet")
rfcv = ru.get_rankings_from_df(dfcv, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                               target="cv_score",
                               lower_is_better=False, impute_missing=False)
rfcv = rfcv.fillna(rfcv.max())
reload(ru)

dfnocv = pd.read_parquet(DATA_DIR / "results_nocv.parquet")
rfnocv = ru.get_rankings_from_df(dfnocv, factors=["dataset", "model", "tuning", "scoring", "fold"],
                                 alternatives="encoder",
                                 target="cv_score", lower_is_better=False, impute_missing=False)
rfnocv = rfnocv.fillna(rfnocv.max())
print("Loaded")

# %% 8a. Compare rankings distributions

ucv = ru.SampleAM.from_rank_function_dataframe(rfcv)
unocv = ru.SampleAM.from_rank_function_dataframe(rfnocv)

# vcv = ku.var(ucv, ku.jaccard_kernel, k=1)
# vnocv = ku.var(unocv, ku.jaccard_kernel, k=1)

# print(vcv, vnocv) 0.8148332754534618 0.8564610452487514

t1 = time()

mmdcv = {
    n: ku.subsample_mmd_distribution(ucv, subsample_size=n, rep=1000, use_njit=True, use_key=False,
                                     kernel=ku.jaccard_kernel_rf, k=1)
    for n in (5, 10, 20, 40, 80, 160)  # , 320, 640, 1000)
}

mmdnocv = {
    n: ku.subsample_mmd_distribution(unocv, subsample_size=n, rep=1000, use_njit=True, use_key=False,
                                     kernel=ku.jaccard_kernel_rf, k=1)
    for n in (5, 10, 20, 40, 80, 160)  # , 320, 640, 1000)
}

print(f"Done in {time() - t1:.02f} seconds.")

# %% 8a1. Plot
epss = np.logspace(-1.5, 0.1, 1000)

ys = defaultdict(lambda: dict())
for cv, mmde_ in [("cv", mmdcv), ("nocv", mmdnocv)]:
    for n, mmde in mmde_.items():
        ys[cv][n] = [gu.generalizability(mmde, eps) for eps in epss]

dfp = pd.DataFrame(ys).explode(["cv", "nocv"]).reset_index(names="n") \
    .melt(id_vars=["n"], var_name="cv?", value_name="generalizability")
dfp["eps"] = list(epss) * int(len(dfp) / len(epss))

fig, ax = plt.subplots()
sns.lineplot(dfp, x="eps", y="generalizability", hue="cv?", style="n", ax=ax)

ax.set_title("Generalizability with and without cross-validation.")
ax.set_xscale("log")
plt.tight_layout()
plt.show(block=True)

# %% 9. Check different sampling methods

reload(ru)
reload(ku)

df = pd.read_parquet(DATA_DIR / "results.parquet")
# df = df.query("tuning == 'full' and scoring != 'AUC'").reset_index(drop=True)
rf = ru.get_rankings_from_df(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                             target="cv_score",
                             lower_is_better=False, impute_missing=True)
rf = rf.fillna(rf.max())
encoder_rankings = ru.SampleAM.from_rank_function_dataframe(rf)
encoder_rankings.set_key(rf.columns.get_level_values(0))
t1 = time()

ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
mmd_sampling = {
    f"disjoint={disjoint}_replace={replace}": {
        n: ku.subsample_mmd_distribution(encoder_rankings, subsample_size=n, rep=500, use_njit=True, use_key=False,
                                         replace=replace, disjoint=disjoint, kernel=ku.jaccard_kernel_rf, k=1)
        for n in tqdm(ns)
    }
    for disjoint, replace in tqdm([(True, True), (True, False), (False, True), (False, False)])
}

print(f"Done in {time() - t1:.1f} seconds.")

# %% 9a. Plot
epss = np.logspace(-2, 0.5, 1000)

ys = defaultdict(lambda: dict())
for factor, mmde_ in mmd_sampling.items():
    for n, mmde in mmde_.items():
        ys[factor][n] = [gu.generalizability(mmde, eps) for eps in epss]

dfp_ = pd.DataFrame(ys)
dfps = {col: dfp_[[col]].dropna().explode(col).reset_index(names="n")[col]
        for col in dfp_.columns}
dfp = pd.concat(dfps.values(), axis=1)
dfp["n"] = dfp_[[dfp_.columns[0]]].dropna().explode(dfp_.columns[0]).reset_index(names="n")["n"]

dfp = dfp.melt(id_vars=["n"], var_name="sampling", value_name="generalizability")

dfp["eps"] = list(epss) * int(len(dfp) / len(epss))

fig, ax = plt.subplots()
sns.lineplot(dfp, x="eps", y="generalizability", hue="sampling", style="n", ax=ax)

ax.set_title("Generalizability for varying MMD sampling method.")
ax.set_xscale("log")
plt.tight_layout()
plt.show(block=True)

# %% 9b. Quantiles

epss = np.logspace(-2, 0.5, 1000)
quantiles = [0.1, 0.5, 0.9]

qs = defaultdict(lambda: dict())
for factor, mmde_ in mmd_sampling.items():
    for n, mmde in mmde_.items():
        qs[factor][n] = np.quantile(mmde, quantiles)

dfp_ = pd.DataFrame(qs)
dfps = {col: dfp_[[col]].dropna().explode(col).reset_index(names="n")[col]
        for col in dfp_.columns}
dfp = pd.concat(dfps.values(), axis=1)
dfp["n"] = dfp_[[dfp_.columns[0]]].dropna().explode(dfp_.columns[0]).reset_index(names="n")["n"]

dfp = dfp.melt(id_vars=["n"], var_name="sampling", value_name="eps")

dfp["quantile"] = quantiles * int(len(dfp) / len(quantiles))

fig, ax = plt.subplots()
sns.lineplot(dfp, x="n", y="eps", hue="sampling", style="quantile", ax=ax)

ax.set_title("Quantiles of generalizability for varying MMD sampling method.")
ax.set_xscale("log")
ax.set_yscale("log")
plt.tight_layout()
plt.show(block=True)

# %% 10. Visualization of probability distributions of rankings

df = pd.read_parquet(DATA_DIR / "results.parquet")
# df = df.query("tuning == 'full' and scoring != 'AUC'").reset_index(drop=True)
rf = ru.get_rankings_from_df(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                             target="cv_score",
                             lower_is_better=False, impute_missing=True)
rf = rf.fillna(rf.max())
encoder_rankings = ru.SampleAM.from_rank_function_dataframe(rf)

# %% 10a. PCA

from sklearn.decomposition import KernelPCA, PCA

rankings = rf.to_numpy().T

t1 = time()
pcas = {
    "linear": PCA(n_components=2).fit_transform(rankings),
    "jaccard_1": KernelPCA(n_components=2, kernel=lambda x, y: ku.jaccard_kernel_rf(x, y, 1)).fit_transform(rankings),
    "jaccard_3": KernelPCA(n_components=2, kernel=lambda x, y: ku.jaccard_kernel_rf(x, y, 3)).fit_transform(rankings),
    "mallows_1": KernelPCA(n_components=2, kernel=ku.mallows_kernel_rf).fit_transform(rankings),
}
print(f"Done in {time() - t1:0.1f} seconds.")

# %% 10a1. PCA Plot

pcas = {
    k: (a - a.min(axis=0)) / (a.max(axis=0) - a.min(axis=0)) for k, a in pcas.items()
}

fig, axes = plt.subplots(1, len(pcas))

for ax, (kind, data) in zip(axes.flatten(), pcas.items()):
    ax.set_title(kind)
    ax.scatter(*data.T, alpha=0.1)

plt.tight_layout()
plt.show(block=True)

# %% 10b. PCA and MMD

from sklearn.decomposition import KernelPCA, PCA

rankings = rf.to_numpy().T

kernels = {
    "jaccard_1": (ku.jaccard_kernel_rf, {"k": 1}),
    "jaccard_2": (ku.jaccard_kernel_rf, {"k": 2}),
    "jaccard_5": (ku.jaccard_kernel_rf, {"k": 5}),
    "mallows_0.01": (ku.mallows_kernel_rf, {"nu": 0.01}),
    "mallows_1": (ku.mallows_kernel_rf, {"nu": 1}),
    "trivial": (ku.trivial_kernel, {}),
    "degenerate": (ku.degenerate_kernel, {}),
}

t1 = time()
pcas = {
    kname: KernelPCA(n_components=2, kernel=k[0], kernel_params=k[1]).fit_transform(rankings)
    for kname, k in kernels.items()
}
print(f"Done in {time() - t1:0.1f} seconds.")

pcas = {
    kname: (a - a.min(axis=0)) / (a.max(axis=0) - a.min(axis=0)) for kname, a in pcas.items()
}

ns = [50]
mmd_kernel = {
    kname: {
        n: ku.subsample_mmd_distribution(encoder_rankings, subsample_size=n, rep=500, use_njit=True, use_key=False,
                                         replace=True, disjoint=False, kernel=k[0], **k[1])
        for n in tqdm(ns)
    }
    for kname, k in tqdm(kernels.items())
}
# %% 10b1. PCA and MMD plot

palette = sns.color_palette("magma")

fig, axes = plt.subplots(3, len(pcas), sharex="row", sharey="row")
for axcol, (kname, data) in zip(axes.T, pcas.items()):
    ax = axcol[0]
    ax.set_title(kname)
    sns.histplot(x=data[:, 0], y=data[:, 1], binwidth=0.03, ax=ax, stat="probability", cbar=True, color="r")

    ax = axcol[1]
    try:
        sns.histplot(mmd_kernel[kname][50], stat="probability", binwidth=0.005, kde=True, ax=ax)
    except ValueError:
        sns.histplot(mmd_kernel[kname][50], stat="probability", bins=[0, 0.005], kde=True, ax=ax)
    ax.set_ylabel("")

    ax = axcol[2]
    sns.ecdfplot(mmd_kernel[kname][50], ax=ax)
    # ax.set_xscale("log")
    ax.set_ylabel("")

# plt.tight_layout()
plt.show(block=True)

# %% 10b2. Better plotting

dfp = pd.concat([pd.DataFrame(kpca, columns=pd.MultiIndex.from_tuples([(kname, "x"), (kname, "y")]))
                 for kname, kpca in pcas.items()],
                axis=1)
dfp = dfp.unstack().unstack(level=1).reset_index(level=1, drop=True).rename_axis("kernel").reset_index()

sns.displot(data=dfp, x="x", y="y", col="kernel", kind="kde", fill=True, height=3, aspect=1.0, cmap="magma", cbar=True)

# %% 10c. graph embedding via node2vec
"""
If used (likely not), cite: ttps://github.com/louisabraham/fastnode2vec
"""

from fastnode2vec import Graph, Node2Vec
from itertools import product

df = pd.read_parquet(DATA_DIR / "results.parquet")
# df = df.query("tuning == 'full' and scoring != 'AUC'").reset_index(drop=True)
rf = ru.get_rankings_from_df(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                             target="cv_score",
                             lower_is_better=False, impute_missing=True)
rf = rf.fillna(rf.max())
encoder_rankings = ru.SampleAM.from_rank_function_dataframe(rf)
rankings = rf.to_numpy()

kernels = {
    "jaccard_1": (ku.jaccard_kernel_rf, {"k": 1}),
    "jaccard_2": (ku.jaccard_kernel_rf, {"k": 2}),
    "jaccard_5": (ku.jaccard_kernel_rf, {"k": 5}),
    "mallows_0.01": (ku.mallows_kernel_rf, {"nu": 0.01}),
    "mallows_1": (ku.mallows_kernel_rf, {"nu": 1}),
    "trivial": (ku.trivial_kernel, {}),
    "degenerate": (ku.degenerate_kernel, {}),
}

gs = {}
n2vs = {}
embeddings = {}
for kname, k in (list(kernels.items())):
    print(kname)
    gram = ku.gram_matrix_rf(rankings, rankings, k[0], **k[1])

    edges = [(i1, i2, gram[i1, i2])
             for (i1, r1), (i2, r2) in product(enumerate(encoder_rankings), repeat=2)
             if i1 < i2]

    gs[kname] = Graph(edges=edges, directed=False, weighted=True)
    n2vs[kname] = Node2Vec(gs[kname], dim=2, walk_length=100, window=10, p=1.0, q=1.0, use_skipgram=True)
    n2vs[kname].train(epochs=100)
    embeddings[kname] = n2vs[kname].wv.vectors

embeddings = {
    kname: (a - a.min(axis=0)) / (a.max(axis=0) - a.min(axis=0)) for kname, a in embeddings.items()
}

ns = [50]
mmd_kernel = {
    kname: {
        n: ku.subsample_mmd_distribution(encoder_rankings, subsample_size=n, rep=500, use_njit=True, use_key=False,
                                         replace=True, disjoint=False, kernel=k[0], **k[1])
        for n in tqdm(ns)
    }
    for kname, k in tqdm(kernels.items())
}

# %% 10c2. Plot graph embeddings VS MMD

fig, axes = plt.subplots(3, len(embeddings), sharex="row", sharey="row")
for axcol, (kname, data) in zip(axes.T, embeddings.items()):
    ax = axcol[0]
    ax.set_title(kname)
    sns.histplot(x=data[:, 0], y=data[:, 1], binwidth=0.03, ax=ax, stat="probability", cbar=True, color="r")

    ax = axcol[1]
    try:
        sns.histplot(mmd_kernel[kname][50], stat="probability", binwidth=0.005, kde=True, ax=ax)
    except ValueError:
        sns.histplot(mmd_kernel[kname][50], stat="probability", bins=[0, 0.005], kde=True, ax=ax)
    ax.set_ylabel("")

    ax = axcol[2]
    sns.ecdfplot(mmd_kernel[kname][50], ax=ax)
    # ax.set_xscale("log")
    ax.set_ylabel("")

# plt.tight_layout()
plt.show(block=True)

# %% 10d. t-SNE

from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE

df = pd.read_parquet(DATA_DIR / "results.parquet")
# df = df.query("tuning == 'full' and scoring != 'AUC'").reset_index(drop=True)
rf = ru.get_rankings_from_df(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                             target="cv_score",
                             lower_is_better=False, impute_missing=True)
rf = rf.fillna(rf.max())
encoder_rankings = ru.SampleAM.from_rank_function_dataframe(rf)
rankings = rf.to_numpy()

kernels = {
    "jaccard_1": (ku.jaccard_kernel_rf, {"k": 1}),
    "jaccard_2": (ku.jaccard_kernel_rf, {"k": 2}),
    "jaccard_5": (ku.jaccard_kernel_rf, {"k": 5}),
    "degenerate": (ku.degenerate_kernel, {}),
    "mallows_0.01": (ku.mallows_kernel_rf, {"nu": 0.01}),
    "mallows_1": (ku.mallows_kernel_rf, {"nu": 1}),
    "trivial": (ku.trivial_kernel, {}),
}

# for t-SNE initialization
t1 = time()
pcas = {
    kname: KernelPCA(n_components=2, kernel=k[0], kernel_params=k[1]).fit_transform(rankings.T)
    for kname, k in kernels.items()
}
print(f"Done in {time() - t1:0.1f} seconds.")
tsnes = {}
for kname, k in tqdm(list(kernels.items())):
    gram = ku.gram_matrix_rf(rankings, rankings, k[0], **k[1])
    dist = np.sqrt(2) * np.sqrt(gram.max() - gram)
    tsnes[kname] = TSNE(n_components=2, metric="precomputed", init=pcas[kname]).fit_transform(dist)

tsnes.update({
    k: (a - a.min(axis=0)) / (a.max(axis=0) - a.min(axis=0)) for k, a in tsnes.items() if a.max() > a.min()
})

ns = [50]
mmd_kernel = {
    kname: {
        n: ku.subsample_mmd_distribution(encoder_rankings, subsample_size=n, rep=500, use_njit=True, use_key=False,
                                         replace=True, disjoint=False, kernel=k[0], **k[1])
        for n in tqdm(ns)
    }
    for kname, k in tqdm(kernels.items())
}

# %% 10d1, Plot t-SNE

fig, axes = plt.subplots(2, len(tsnes), sharex="row", sharey="row")
for ic, (axcol, (kname, data)) in enumerate(zip(axes.T, tsnes.items())):
    ax = axcol[0]
    ax.set_title(kname)
    sns.histplot(x=data[:, 0], y=data[:, 1], binwidth=0.05, ax=ax, stat="probability", cbar=(ic == len(axes.T) - 1),
                 vmin=0, vmax=0.3,
                 cmap="flare")
    # sns.kdeplot(x=data[:, 0], y=data[:, 1], cbar=True, ax=ax, vmin=0, vmax=0.2, warn_singular=False, cmap="flare", fill=True)
    if ic == 0:
        ax.set_ylabel("t-SNE of KernelPCA of rankings")
    ax.set_aspect('equal')

    ax = axcol[1]
    try:
        sns.histplot(mmd_kernel[kname][50], stat="probability", binwidth=0.005, kde=True, ax=ax, color="b")
    except ValueError:
        sns.histplot(mmd_kernel[kname][50], stat="probability", bins=[0, 0.005], kde=True, ax=ax, color="b")
    ax.set_ylim((0, 0.5))
    if ic == 0:
        ax.set_ylabel("MMD")

    # ax = axcol[2]
    # sns.ecdfplot(mmd_kernel[kname][50], ax=ax)
    # # ax.set_xscale("log")
    # if ic == 0:
    #     ax.set_ylabel("Generalizability")

plt.tight_layout()
plt.show(block=True)

# %% 10e. Heatmap of gram matrix

df = pd.read_parquet(DATA_DIR / "results.parquet")
# df = df.query("tuning == 'full' and scoring != 'AUC'").reset_index(drop=True)
rf = ru.get_rankings_from_df(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                             target="cv_score",
                             lower_is_better=False, impute_missing=True)
rf = rf.fillna(rf.max())
encoder_rankings = ru.SampleAM.from_rank_function_dataframe(rf)
rankings = rf.to_numpy()

kernels = {
    "jaccard_1": (ku.jaccard_kernel_rf, {"k": 1}),
    "jaccard_2": (ku.jaccard_kernel_rf, {"k": 2}),
    "jaccard_5": (ku.jaccard_kernel_rf, {"k": 5}),
    "degenerate": (ku.degenerate_kernel, {}),
    "mallows_0.01": (ku.mallows_kernel_rf, {"nu": 0.01}),
    "mallows_1": (ku.mallows_kernel_rf, {"nu": 1}),
    "trivial": (ku.trivial_kernel, {}),
}

grams = {}
for kname, k in list(kernels.items()):
    grams[kname] = ku.gram_matrix_rf(rankings, rankings, k[0], **k[1])

dists = {kname: np.sqrt(2) * np.sqrt(gram.max() - gram) for kname, gram in grams.items()}

ns = [50]
mmd_kernel = {
    kname: {
        n: ku.subsample_mmd_distribution(encoder_rankings, subsample_size=n, rep=500, use_njit=True, use_key=False,
                                         replace=True, disjoint=False, kernel=k[0], **k[1])
        for n in tqdm(ns)
    }
    for kname, k in tqdm(kernels.items())
}
# %% 10e1. Plot heatmap

from sklearn.cluster import OPTICS

fig, axes = plt.subplots(2, len(grams), sharex="row", sharey="row")
for ic, (axcol, (kname, data)) in enumerate(zip(axes.T, grams.items())):

    # compute clusters for heatmap
    cluster = OPTICS(min_samples=10, metric="precomputed").fit_predict(dists[kname])
    key = sorted(range(len(data)), key=lambda i: cluster[i])

    ax = axcol[0]
    ax.set_title(kname)
    ax.imshow(data[key, :][:, key], vmin=0, vmax=1)

    # sns.heatmap(data[key, :][:, key], vmin=0, vmax=1, cmap="flare", cbar=False, ax=ax, square=True, mask=np.triu(data))
    if ic == 0:
        ax.set_ylabel("Gram matrix")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axcol[1]
    try:
        sns.histplot(mmd_kernel[kname][50], stat="probability", binwidth=0.005, kde=True, ax=ax, color="b")
    except ValueError:
        sns.histplot(mmd_kernel[kname][50], stat="probability", bins=[0, 0.005], kde=True, ax=ax, color="b")
    ax.set_ylim((0, 0.5))
    if ic == 0:
        ax.set_ylabel("MMD")

plt.tight_layout()
plt.show(block=True)

# %% 11. Behavior wrt Jaccard_k

df = pd.read_parquet(DATA_DIR / "results.parquet")
# df = df.query("tuning == 'full' and scoring != 'AUC'").reset_index(drop=True)
rf = ru.get_rankings_from_df(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                             target="cv_score",
                             lower_is_better=False, impute_missing=True)
rf = rf.fillna(rf.max())
encoder_rankings = ru.SampleAM.from_rank_function_dataframe(rf)
rankings = rf.to_numpy()

kernels = {f"jaccard_{k:02d}": (ku.jaccard_kernel_rf, {"k": k}) for k in range(1, 33)}

t1 = time()
grams = {}
for kname, k in list(kernels.items()):
    grams[kname] = ku.gram_matrix_rf(rankings, rankings, k[0], **k[1])
dists = {kname: np.sqrt(2) * np.sqrt(gram.max() - gram) for kname, gram in grams.items()}

t2 = time()

ns = [50]
mmd_kernel = {
    kname: {
        n: ku.subsample_mmd_distribution(encoder_rankings, subsample_size=n, rep=500, use_njit=True, use_key=False,
                                         replace=True, disjoint=False, kernel=k[0], **k[1])
        for n in tqdm(ns)
    }
    for kname, k in tqdm(kernels.items())
}

print(f"Done in {t2-t1:.0f} + {time()-t2:.0f} seconds.")

# %% 11a. Get a gif of MMD for Jaccard similarity

import imageio.v2 as iio

plt.ioff()
for kname, mmdk_ in mmd_kernel.items():
    mmdk = mmdk_[ns[0]]

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.suptitle(f"MMD of encoders\n{kname}")

    try:
        sns.histplot(mmdk, stat="probability", binwidth=0.002, kde=True, ax=ax, color="b")
    except ValueError:
        sns.histplot(mmdk, stat="probability", bins=[0, 0.002], kde=True, ax=ax, color="b")

    ax.set_xlim((0, 0.4))
    ax.set_ylim((0, 0.2))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "MMD for Jaccard kernel" / f"{kname}")
plt.ion()
plt.close("all")

images = [iio.imread(image) for image in (FIGURES_DIR / "MMD for Jaccard kernel").iterdir()]
iio.mimwrite(FIGURES_DIR / "MMD_jaccard.gif", images, duration=150, loop=0)  # kwargs: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html

# %% 11b. Get a gif of generalizability for Jaccard similarity

import imageio.v2 as iio
import os

dirname = "Gen for Jaccard kernel"
try:
    os.mkdir(FIGURES_DIR / dirname)
except FileExistsError:
    pass

epss = np.linspace(0, 0.4, 1000)

plt.ioff()
for kname, mmdk_ in mmd_kernel.items():
    mmdk = mmdk_[ns[0]]

    y = [gu.generalizability(mmdk, eps) for eps in epss]

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.suptitle(f"Generalizability of encoders\n{kname}")

    sns.ecdfplot(mmdk, ax=ax, color="b")

    ax.set_xlim((0, 0.4))
    ax.set_ylim((0, 1))
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / dirname / f"{kname}")
    plt.close("all")
plt.ion()


images = [iio.imread(image) for image in (FIGURES_DIR / dirname).iterdir()]
iio.mimwrite(FIGURES_DIR / "generalizability_jaccard.gif", images, duration=150, loop=0)  # kwargs: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html

# %%