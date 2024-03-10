"""
This file produces the plots for the PhD Colloquium.

1. MMD - generalizability - heatmap kernel for different synthetic distributions

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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from time import time
from tqdm.notebook import tqdm
from typing import Iterable

import src.generalizability_utils as gu
import src.kernel_utils1 as ku
import src.probability_distributions as prob
import src.rankings_utils as ru

DATA_DIR = Path("./data/benchmark_encoders")
FIGURES_DIR = Path("./Figures") / "Phd colloquium"
mpl.use("TkAgg")
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r"""\usepackage{helvet}"""
mpl.rcParams["font.family"] = "Serif"
sns.set()
sns.set_style("white")
palette = "flare_r"
sns.set_palette("flare_r")

# %% 1. Sample different distributions

na = 3
N = 40
seed = 1444

U = ru.universe_untied_rankings(na=na)

samples = {
    "uniform": prob.UniformDistribution(U).sample(N, seed),
    # "bidegenerate": prob.MDegenerateDistribution(U).sample(N, seed, 2),
    "center": prob.CenterProbabilityDistribution(U).sample(N, seed, kernel=lambda x, y: ku.mallows_kernel(x, y,
                                                                                                          use_rf=True,
                                                                                                          nu=1 / 3))
}

kernels = {
    "jaccard": (ku.jaccard_kernel, {"k": 1}),
    "mallows": (ku.mallows_kernel, {"nu": "auto"})
}

mmds = defaultdict(lambda: {})
grams = defaultdict(lambda: {})
for dname, distr in samples.items():
    for kname, (kernel, kernelargs) in kernels.items():
        mmds[dname][kname] = ku.subsample_mmd_distribution(distr, subsample_size=10, rep=100,
                                                           use_rf=True, use_key=False, seed=seed,
                                                           disjoint=True, replace=False,
                                                           kernel=kernel, **kernelargs)
        grams[dname][kname] = ku.square_gram_matrix(distr, use_rf=False, kernel=kernel, **kernelargs)

fig, axes = plt.subplots(len(kernels) * len(samples), 2, sharex="col", sharey="col")
for i0, dname in enumerate(samples.keys()):
    for i1, kname in enumerate(kernels.keys()):
        axrow = axes[i0 * len(kernels) + i1, :]

        ax = axrow[0]
        sns.heatmap(grams[dname][kname], ax=ax, square=True, cbar=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(kname)

        ax = axrow[1]
        sns.histplot(mmds[dname][kname], ax=ax, stat="probability")
        sns.ecdfplot(mmds[dname][kname], ax=ax)
        # ax.set_xscale("log")

sns.despine()
plt.show()

# %% 2. Simple generalizability of encoder results

model = "LR"
tuning = "no"
scoring = "ACC"
eps = np.sqrt(2) * np.sqrt(1-0.95)  # at least 0.95 overlapping (on average)
alpha = 0.80  # in 95% of cases, we'll get similar experimental results
lr_confidence = 0.9  # confidence interval for linear prediction
seed = 1444

# ---- Load data
df = pd.read_parquet(DATA_DIR / "results.parquet")
df = df.query("model == @model and tuning == @tuning and scoring == @scoring").reset_index(drop=True)
rf = ru.get_rankings_from_df(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                             target="cv_score",
                             lower_is_better=False, impute_missing=True)
rf = rf.fillna(rf.max())
rankings = ru.SampleAM.from_rank_function_dataframe(rf)

mmds = {
    n: ku.subsample_mmd_distribution(rankings, subsample_size=n, rep=100,
                                     use_rf=True, use_key=False, seed=seed,
                                     disjoint=True, replace=False,
                                     kernel=ku.jaccard_kernel, **{"k": 1})
    for n in [5, 10, 20]
}

logepss = np.linspace(np.log(eps) - 0.5,
                      np.log(max(np.quantile(mmde, alpha) for mmde in mmds.values())) + 0.1,
                      1000)
ys = {}
qs = {}
for n, mmde in mmds.items():
    ys[n] = [gu.generalizability(mmde, np.exp(logeps)) for logeps in logepss]
    qs[n] = np.log(np.quantile(mmde, alpha))

dfy = pd.DataFrame(ys)
dfy["log(eps)"] = logepss
dfy = dfy.melt(var_name="n", value_name="generalizability", id_vars="log(eps)")
dfy["n"] = dfy["n"].astype(int)
dfy["eps"] = np.exp(dfy["log(eps)"])

dfq = pd.DataFrame.from_dict(qs, orient="index").reset_index()
dfq.columns = ["n", "log(eps)"]
dfq["log(n)"] = np.log(dfq["n"])
dfq["eps"] = np.exp(dfq["log(eps)"])

# -- Fit a linear predictor with cross-validated confidence intervals for log(n) as function of log(eps)
X = dfq["log(eps)"].to_numpy().reshape(-1, 1)
y = dfq["log(n)"].to_numpy().reshape(-1, 1)

cv = KFold(len(y))
linear_predictors = []  # fitted predictors
res = np.array([])  # residuals
for tr, te in cv.split(X):
    lr_tmp = LinearRegression()
    res = np.append(res, y[te] - lr_tmp.fit(X[tr], y[tr]).predict(X[te]))
    linear_predictors.append(lr_tmp)
# TODO: move to another ci
ci = np.abs(np.quantile(res, 1 - lr_confidence))  # confidence interval in log(n)

# - Get all predictions
ns_pred_cv = []
for lr_tmp in linear_predictors:
    ns_tmp = np.exp(lr_tmp.predict(logepss.reshape(-1, 1)).reshape(1, -1)[0])
    ns_pred_cv.append(ns_tmp)

# - Refit a linear model on the entire data
lr = LinearRegression()
lr.fit(X, y)
ns_pred = np.exp(lr.predict(logepss.reshape(-1, 1)).reshape(1, -1)[0])

# -- Predict nstar
nstar_cv = []
for lr_tmp, ns_tmp in zip(linear_predictors, ns_pred_cv):
    nstar_tmp = ns_tmp[np.argmax(logepss > np.log(eps))] if lr_tmp.coef_ != 0 else np.nan
    nstar_cv.append(nstar_tmp)

nstar = ns_pred[np.argmax(logepss > np.log(eps))] if lr.coef_ != 0 else np.nan
nstar_lower, nstar_upper = np.quantile(nstar_cv, [0.05, 0.95])

# -- Plot
fig, axes = plt.subplots(2, 1, sharex="all", figsize=(8, 7))
# fig, ax = plt.subplots(figsize=(8, 5))

logepss = np.exp(logepss)  #TODO

# - Generalizability
ax = axes[0]
sns.lineplot(dfy, x="eps", y="generalizability", hue="n", ax=ax, palette=palette)
ax.hlines(alpha, ls="--", xmin=np.min(logepss), xmax=np.max(logepss), color="black")
for n in mmds.keys():
    ax.vlines(np.exp(qs[n]), ymin=0, ymax=alpha, ls=":")
# ax.vlines(eps, ls="--", ymin=0, ymax=1, color="black")
# for n, mmd in mmds.items():
#     ax.hlines(gu.generalizability(mmd, eps), xmin=min(logepss), xmax=max(logepss), ls=":")

ax.set_xlabel(r"$\varepsilon$")
# ax.set_xlim(min(logepss), max(logepss))
# ax.set_yticks(np.append([0, 1],
#                         [gu.generalizability(mmd, eps) for mmd in mmds.values()]))
ax.set_yticks([0, 1, alpha])
ax.set_xscale("log")
# ax.set_xticks([min(logepss), max(logepss), eps])
# ax.set_xticks(np.append([],
#                         dfq["eps"].unique()))

ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter('{x:.02f}')
ax.xaxis.set_major_formatter('{x:.02f}')

sns.despine(ax=ax)

# - Quantiles
ax = axes[1]
ymax = max(ns_pred)
sns.lineplot(dfq, x="eps", y="n", ax=ax, ls="", marker="o", legend=False)
for n in mmds.keys():
    ax.vlines(np.exp(qs[n]), ymin=n, ymax=ymax, ls=":")
ax.vlines(eps, ymin=0.1, ymax=ymax, color="black", ls="--")

# - Linear regression
sns.lineplot(x=logepss, y=ns_pred, color="green", ls="-.", ax=ax)
# for it, ns_tmp in enumerate(ns_pred_cv):
#     if np.max(ns_tmp) > 1000:  # TODO: hard-coded threshold for broken confidence intervals, make it more reliable
#         continue
#     sns.lineplot(x=logepss, y=ns_tmp, color="green", ls="-.", alpha=0.5, ax=ax)

# - N*
ax.hlines(nstar, xmin=np.min(logepss), xmax=eps, ls=":", color="red")
# ax.hlines(nstar_upper, xmin=np.min(logepss), xmax=np.log(eps), ls="-", color="red", alpha=0.3)
# ax.hlines(nstar_lower, xmin=np.min(logepss), xmax=np.log(eps), ls="-", color="red", alpha=0.3)

ax.set_yscale("log")
ax.set_yticks(np.append(ax.get_yticks(),
                        nstar))
ax.set_ylim([min(ns_pred), ymax])
# ax.yaxis.set_major_formatter('{x:.01e}')

sns.despine(ax=ax)

# - Finalize
# fig.suptitle(r"Generalizability for N = {len(df['dataset'].unique()):02d}\n"
#              r"n*(\alpha={alpha:.02f}, \vareps={eps:.02f}) = {np.ceil(nstar)}\n"
#              r"{lr_confidence} confidence interval: [{np.ceil(nstar_lower)}, {np.ceil(nstar_upper)}]")
plt.tight_layout()
# plt.savefig(FIGURES_DIR / "example_generalizability_fixedeps.pdf")
# plt.savefig(FIGURES_DIR / "example_generalizability_fixedalpha.pdf")
# plt.savefig(FIGURES_DIR / "example_generalizability_fixedepsalpha.pdf")
# plt.savefig(FIGURES_DIR / "example_generalizability_withquantiles.pdf")
plt.savefig(FIGURES_DIR / "example_generalizability_withquantiles_withlinreg.pdf")











