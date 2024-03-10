"""
Run PoC 5 on the LLMs data.
In particular, now, all experimental conditions are considered the same (task, subtkas, number of shots, score).
"""

import glob
import imageio.v2 as iio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from collections import defaultdict
from importlib import reload
from itertools import product
from numpy.random import Generator
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from time import time
from tqdm import tqdm

import src.generalizability_utils as gu
import src.kernel_utils1 as ku
import src.probability_distributions as prob
import src.rankings_utils as ru
import src.relation_utils as rlu

DATA_DIR = Path("./data/benchmark_llms")
FIGURES_DIR = Path("./Figures")

mpl.use("TkAgg")
sns.set_style("ticks")
sns.set_context("notebook")
palette = "flare_r"
sns.set_palette("flare_r")


df = pd.read_parquet(DATA_DIR / "benchmark_llms.parquet")

# print(np.max(df.groupby(by=["task_name", "subtask_description", "number_of_shots", "score_key",
#                       "model_family_model_name"]).count()))

# rf = ru.get_rankings_from_df(df,
#                              factors=["task_name", "subtask_description", "number_of_shots", "score_key"],
#                              alternatives="model_family_model_name",
#                              target="score_value",
#                              lower_is_better=False, impute_missing=False, verbose=True)
# rf.to_parquet(DATA_DIR / "rankings_llms.parquet")

# rf = pd.read_parquet(DATA_DIR / "rankings_llms.parquet")

#%% 1. Pre-processing: remove underrepresented models and tasks

rf = pd.read_parquet(DATA_DIR / "rankings_llms.parquet")

tol = 0.2

# rf_ = rf.copy()
# rf_ = rf_.loc[rf_.isna().sum(axis=1) <= rf_.shape[1] * tol, :]
# rf_ = rf_.loc[:, rf_.isna().sum(axis=0) <= rf_.shape[0] * tol]
# print(rf_.shape)

rf_ = rf.copy()
rf_ = rf_.loc[:, rf_.isna().sum(axis=0) <= rf_.shape[0] * tol]
rf_ = rf_.loc[rf_.isna().sum(axis=1) <= rf_.shape[1] * tol, :]
print(rf_.shape)

for ec in tqdm(rf_.columns):
    rf_[ec] = rlu.score2rf(rf_[ec], lower_is_better=True, impute_missing=True)
rf_.to_parquet(DATA_DIR / "rankings_llms_preproc.parquet")

"""
The second method, allowing for 20% of missing values, leaves a decent table.
We compact the rankings and fill in the missing values with the max, although in this case it's not a failed evaluation
    but a genuine missing one. 
"""

#%% 2. Proceed with the generalizability analysis

# ---- Parameters
eps = np.sqrt(2) * np.sqrt(1-0.95)  # max MMD is sqrt2 sqrt(kmax-kmin), we fix kmin=0.95
alpha = 0.95  # in 95% of cases, we'll get similar experimental results
lr_confidence = 0.9  # confidence interval for linear prediction
seed = 1444

# ---- Load data
rf = pd.read_parquet(DATA_DIR / "rankings_llms_preproc.parquet")

# -- Directory
EXP0_DIR = FIGURES_DIR / "Proof of concept 6" / f"LLMs"

kernels = {
    # "mallows_auto": (ku.mallows_kernel, {"nu": "auto"}),
    "jaccard_1": (ku.jaccard_kernel, {"k": 1}),
    # "borda_OHE": (ku.borda_kernel, {"idx": rf.index.get_loc("OHE")}),
    # "borda_DE": (ku.borda_kernel, {"idx": rf.index.get_loc("DE")})
}

# ---- Main loop
for kernelname, (kernel, kernelargs) in tqdm(list(kernels.items())):

    EXP1_DIR = EXP0_DIR / f"{kernelname}"

    for disjoint, replace in product([False, True], repeat=2):  # disjoint -> conservative, replace -> conservative
    # for disjoint, replace in [(True, False)]:
        # ---- Create directories
        EXP2_DIR = EXP1_DIR / f"nstar_N_alpha={alpha}_eps={eps}_ci={lr_confidence}_disjoint={disjoint}_replace={replace}"
        for ED in [EXP0_DIR, EXP1_DIR, EXP2_DIR]:
            try:
                os.mkdir(ED)
            except FileExistsError:
                pass

        # ---- Computation
        ec_pool = rf.columns  # we remove datasets from it to simulate running new experiments
        ecs = np.array([])  # datasets on which we have already run experiments
        out = []
        plt.ioff()
        while len(ec_pool) > 0 and len(ecs) < 50:  # maximum size
            # -- Sample a new dataset
            # initialization with the minimum meaningful n=6 if disjoint is True
            if len(ecs) == 0:
                ecs = np.random.default_rng(seed).choice(ec_pool, 10, replace=False)
            else:
                ecs = np.append(ecs,
                                np.random.default_rng(seed).choice(ec_pool, 10, replace=False))
            ec_pool = np.setdiff1d(ec_pool, ecs)  # remove the sampled ecs rom the pool

            rf_ = rf.loc[:, ecs]
            na, nv = rf_.shape
            rankings = ru.SampleAM.from_rank_function_dataframe(rf_)

            # -- Compute the lower bound
            variance = ku.var(rankings, use_rf=True, kernel=kernel, **kernelargs)
            var_lower_bound = gu.sample_mean_embedding_lowerbound(eps, len(ecs), kbar=1,
                                                                  v=variance)
            # print(f"---- N = {len(datasets)} ----")
            # print(f"Lower bound on prob(||mu(U) - mu(P)|| < {eps}) = {var_lower_bound:.03f}.")

            # -- Compute mmds

            t1 = time()
            mmds = {
                n: ku.subsample_mmd_distribution(rankings, subsample_size=n, rep=100,
                                                 use_rf=True, use_key=False, seed=seed,
                                                 disjoint=disjoint, replace=replace,
                                                 kernel=kernel, **kernelargs)
                for n in range(2, nv // 2 + 1)
                # for n in [6, ]
            }
            # print(f"MMD done in {time() - t1:.1f} seconds.")

            # -- Compute generalizability and quantiles
            logepss = np.linspace(np.log(eps) - 1,
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

            dfq = pd.DataFrame.from_dict(qs, orient="index").reset_index()
            dfq.columns = ["n", "log(eps)"]
            dfq["log(n)"] = np.log(dfq["n"])

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
            fig, axes = plt.subplots(2, 1, sharex="all", figsize=(10, 8))

            # - Generalizability
            ax = axes[0]
            sns.lineplot(dfy, x="log(eps)", y="generalizability", hue="n", ax=ax, palette=palette)
            ax.hlines(alpha, ls="--", xmin=np.min(logepss), xmax=np.max(logepss), color="black")
            for n in mmds.keys():
                ax.vlines(qs[n], ymin=0, ymax=alpha, ls=":")
            sns.despine(ax=ax)

            # - Quantiles
            ax = axes[1]
            ymax = max(ns_pred)
            sns.lineplot(dfq, x="log(eps)", y="n", ax=ax, ls="", marker="o", hue="n", legend=False)
            for n in mmds.keys():
                ax.vlines(qs[n], ymin=n, ymax=ymax, ls=":")
            ax.vlines(np.log(eps), ymin=0.1, ymax=ymax, color="black", ls="--")

            # - Linear regression
            sns.lineplot(x=logepss, y=ns_pred, color="green", ls="-.", ax=ax)
            for it, ns_tmp in enumerate(ns_pred_cv):
                if np.max(ns_tmp) > 1000:  # TODO: hard-coded threshold for broken confidence intervals, make it more reliable
                    continue
                sns.lineplot(x=logepss, y=ns_tmp, color="green", ls="-.", alpha=0.5, ax=ax)

            # - N*
            ax.hlines(nstar, xmin=np.min(logepss), xmax=np.log(eps), ls="-", color="red")
            ax.hlines(nstar_upper, xmin=np.min(logepss), xmax=np.log(eps), ls="-", color="red", alpha=0.3)
            ax.hlines(nstar_lower, xmin=np.min(logepss), xmax=np.log(eps), ls="-", color="red", alpha=0.3)

            ax.set_yscale("log")
            sns.despine(ax=ax)

            # - Finalize
            fig.suptitle(f"Generalizability for N = {len(ecs):02d}\n"
                         f"n*(alpha={alpha}, eps={eps}) = {np.ceil(nstar)}\n"
                         f"{lr_confidence} confidence interval: [{np.ceil(nstar_lower)}, {np.ceil(nstar_upper)}]")
            plt.tight_layout()
            plt.savefig(EXP2_DIR / f"N={len(ecs):02d}.png")
            plt.savefig(EXP2_DIR / f"N={len(ecs):02d}.pdf")
            plt.close("all")

            out.append({
                "kernel": kernelname,
                "alpha": alpha,
                "eps": eps,
                "disjoint": disjoint,
                "replace": replace,
                "N": len(ecs),
                "nstar": nstar,
                "nstar_lower": nstar_lower,
                "nstar_upper": nstar_upper,
                "variance": variance,
                "var_lower_bound": var_lower_bound,
            })
        plt.ion()

        # -- Get a GIF
        images = [iio.imread(image) for image in glob.glob(str(EXP2_DIR / "*.png"))]
        iio.mimwrite(EXP2_DIR / f"nstar.gif", images, duration=750,
                     loop=0)  # kwargs: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html

        # -- Store nstar predictions
        out = pd.DataFrame(out)
        out.to_parquet(EXP2_DIR / "nstar.parquet")

print("Done!")

# %% 2. Behavior of nstar for different sampling strategies
"""
You need to run all strategies before running this section, by changing disjoint and replace.

Strategies:
    - conservative: disjoint=True, replace=True
    - optimistic: disjoint=False, replace=False
    - realistic: disjoint=True, replace=False
    - ? : disjoint=False, replace=True
"""
# model = "LR"
# EXP0_DIR = FIGURES_DIR / "Proof of concept 2" / f"encoders_{model}_{tuning}_{scoring}"
for kernelname, (kernel, kernelargs) in tqdm(list(kernels.items())):
    EXP1_DIR = EXP0_DIR / f"{kernelname}"
    dfstar = []
    for disjoint, replace in product([True, False], repeat=2):
        df_tmp = pd.read_parquet(
            EXP1_DIR / f"nstar_N_alpha={alpha}_eps={eps}_ci={lr_confidence}_disjoint={disjoint}_replace={replace}" / "nstar.parquet")
        match disjoint, replace:
            case (True, True):
                df_tmp["sampling_strategy"] = "pessimistic"
            case (False, False):
                df_tmp["sampling_strategy"] = "optimistic"
            case (True, False):
                df_tmp["sampling_strategy"] = "realistic"
            case (False, True):
                df_tmp["sampling_strategy"] = "?"
        dfstar.append(df_tmp)
    dfstar = pd.concat(dfstar, axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(dfstar, x="N", y="nstar", hue="sampling_strategy", palette="colorblind")
    for sampling_strategy, color in zip(dfstar["sampling_strategy"].unique(),
                                        sns.color_palette("colorblind", len(dfstar["sampling_strategy"].unique()))):
        df_tmp = dfstar.query("sampling_strategy == @sampling_strategy")
        ax.fill_between(df_tmp["N"], df_tmp["nstar_lower"], df_tmp["nstar_upper"], alpha=0.3, color=color)
    sns.lineplot(dfstar, x="N", y="N", color="black", ls="--", label="termination")

    ax.set_ylim(0.1, np.quantile(dfstar["nstar"], 0.95))
    ax.grid()
    sns.despine(ax=ax)

    fig.suptitle("n* as function of N. The termination criterion is n* <= N\n"
                 f"model: {model}, scoring: {scoring}, tuning: {tuning}\n"
                 f"kernel: {kernelname}, alpha:{alpha}, eps: {eps}")
    plt.tight_layout()
    plt.savefig(EXP1_DIR / "nstar_N_sampling_strategies.pdf")
    plt.show()

# %% 3. Visualize MMD
"""
We found that results for Jaccard(k=1) are not generalizable, requiring 70 datasets. However, results with Mallows(nu=1)
    are generalizable, requiring "only" 47 datasets. 
Remember that the prediction of n* depends is made on at most 25 datasets in both cases.

Now, let's see what the distributions look like, so to visualize the discrepancy. 

For the Jaccard kernel, it should be enough to look at how many times each encoder was among the best ones.
For the Mallows kernel, we need to consider the entire rankings.

Let's start from MMD.

Realistic sampling: disjoint=True, replace=False


TODO: why does Mallows_1 give so much generalizability? something is off. as when I checked MMD it only had two possible 
    values. Extremely weird. It does make sense! We are predicting (from n=25) that at N=50 we'll get a decent 
    generalizability. I guess because the spiked MMD converges quite fast. 
    More, mallows_1 is basically the trivial kernel.

"""
reload(ku)
disjoint = True
replace = False

# kernels = {
#     "jaccard_1": (ku.jaccard_kernel, {"k": 1}),
#     "mallows_1": (ku.mallows_kernel, {"nu": 1}),
#     "mallows_auto": (ku.mallows_kernel, {"nu": "auto"}),
#     "borda_OHE": (ku.borda_kernel, {"idx": 24}),
# }

rankings = ru.SampleAM.from_rank_function_dataframe(rf)

mmd_kernels = {kname: {n: ku.subsample_mmd_distribution(rankings, subsample_size=n, rep=100,
                                                        use_rf=True, use_key=False, seed=seed,
                                                        disjoint=disjoint, replace=replace,
                                                        kernel=kernel, **kernelargs)
                       for n in [25]}
               for kname, (kernel, kernelargs) in kernels.items()}

dfmmd = pd.DataFrame(mmd_kernels).explode(list(mmd_kernels.keys())).rename_axis(index="n").reset_index().melt(id_vars="n", var_name=["kernel"], value_name="MMD")

fig, ax = plt.subplots()

sns.histplot(data=dfmmd, x="MMD", hue="kernel", stat="probability", kde=True, multiple="layer", cumulative=False,
             palette="colorblind", ax=ax)
ax.axvline(eps, ls="--", color="black")

plt.tight_layout()
plt.show()

# %% 4. Visualize Gram matrices
reload(ku)

grams = {}
for kname, k in list(kernels.items()):
    grams[kname] = ku.gram_matrix(rankings, rankings, use_rf=True, kernel=k[0], **k[1])

dists = {kname: np.sqrt(2) * np.sqrt(gram.max() - gram) for kname, gram in grams.items()}

from sklearn.cluster import OPTICS

fig, axes = plt.subplots(ncols=len(grams), sharex="row", sharey="row")
for ax, (kname, data) in zip(axes.flatten(), grams.items()):

    # compute clusters for heatmap
    cluster = OPTICS(min_samples=10, metric="precomputed").fit_predict(dists[kname])
    key = sorted(range(len(data)), key=lambda i: cluster[i])

    ax.set_title(kname)
    ax.imshow(data[key, :][:, key], vmin=0, vmax=1)

plt.tight_layout()
plt.show()

# %% 5. Visualize variance

for kernelname, (kernel, kernelargs) in tqdm(list(kernels.items())):
    EXP1_DIR = EXP0_DIR / f"{kernelname}"
    dfstar = []
    for disjoint, replace in product([True, False], repeat=2):
        df_tmp = pd.read_parquet(
            EXP1_DIR / f"nstar_N_alpha={alpha}_eps={eps}_ci={lr_confidence}_disjoint={disjoint}_replace={replace}" / "nstar.parquet")
        match disjoint, replace:
            case (True, True):
                df_tmp["sampling_strategy"] = "pessimistic"
            case (False, False):
                df_tmp["sampling_strategy"] = "optimistic"
            case (True, False):
                df_tmp["sampling_strategy"] = "realistic"
            case (False, True):
                df_tmp["sampling_strategy"] = "?"
        dfstar.append(df_tmp)
    dfstar = pd.concat(dfstar, axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(dfstar, x="N", y="variance", palette="colorblind")
    for sampling_strategy, color in zip(dfstar["sampling_strategy"].unique(),
                                        sns.color_palette("colorblind", len(dfstar["sampling_strategy"].unique()))):
        df_tmp = dfstar.query("sampling_strategy == @sampling_strategy")

    ax.grid()
    sns.despine(ax=ax)

    fig.suptitle("Variance as function of N\n"
                 f"model: {model}, scoring: {scoring}, tuning: {tuning}\n"
                 f"kernel: {kernelname}, alpha:{alpha}, eps: {eps}")
    plt.tight_layout()
    plt.savefig(EXP1_DIR / "variance_N.pdf")
    plt.show()

# %% 5. Visualize prob good

for kernelname, (kernel, kernelargs) in tqdm(list(kernels.items())):
    EXP1_DIR = EXP0_DIR / f"{kernelname}"
    dfstar = []
    for disjoint, replace in product([True, False], repeat=2):
        df_tmp = pd.read_parquet(
            EXP1_DIR / f"nstar_N_alpha={alpha}_eps={eps}_ci={lr_confidence}_disjoint={disjoint}_replace={replace}" / "nstar.parquet")
        match disjoint, replace:
            case (True, True):
                df_tmp["sampling_strategy"] = "pessimistic"
            case (False, False):
                df_tmp["sampling_strategy"] = "optimistic"
            case (True, False):
                df_tmp["sampling_strategy"] = "realistic"
            case (False, True):
                df_tmp["sampling_strategy"] = "?"
        dfstar.append(df_tmp)
    dfstar = pd.concat(dfstar, axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(dfstar, x="N", y="var_lower_bound", palette="colorblind")
    for sampling_strategy, color in zip(dfstar["sampling_strategy"].unique(),
                                        sns.color_palette("colorblind", len(dfstar["sampling_strategy"].unique()))):
        df_tmp = dfstar.query("sampling_strategy == @sampling_strategy")

    ax.grid()
    sns.despine(ax=ax)

    fig.suptitle("Lower bound on ||mu(U) - mu(P)|| as function of N\n"
                 f"model: {model}, scoring: {scoring}, tuning: {tuning}\n"
                 f"kernel: {kernelname}, alpha:{alpha}, eps: {eps}")
    plt.tight_layout()
    plt.savefig(EXP1_DIR / "var_lower_bound_N.pdf")
    plt.show()


