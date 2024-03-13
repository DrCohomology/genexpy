# %% Script
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

import src.lower_bounds as gu
import src.kernel_utils as ku
import src.probability_distributions as prob
import src.rankings_utils as ru
import src.mmd as mmd

DATA_DIR = Path("/home/kn/Code/Generalizability-of-experimental-comparisons/data/benchmark_encoders")
FIGURES_DIR = Path("/home/kn/Code/Generalizability-of-experimental-comparisons/Figures")

mpl.use("TkAgg")
sns.set_style("ticks")
sns.set_context("notebook")
palette = "flare_r"
sns.set_palette("flare_r")

# ---- Parameters
eps = 0.2  # max MMD is sqrt2 sqrt(kbar), i.e., sqrt2 in most applications
alpha = 0.95  # in 95% of cases, we'll get similar experimental results
lr_confidence = 0.9  # confidence interval for linear prediction
seed = 1444

# ---- Load data
#Dataset
DATA_SET = Path(DATA_DIR / "results.parquet")

# load Dataset into df
if DATA_SET.suffix == '.parquet':
    df = pd.read_parquet(DATA_SET)
elif DATA_SET.suffix == '.csv':
    df = pd.read_csv(DATA_SET)
else:
    raise Exception("Please use a Parquet or CSV file as the format of your data")

# Querying parameters
model = "LR"
tuning = "no"
scoring = "ACC"

# Build query
df = df.query("model == @model and tuning == @tuning and scoring == @scoring").reset_index(drop=True)
rf = ru.get_rankings_from_df(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder",
                             target="cv_score",
                             lower_is_better=False, impute_missing=True)
rf = rf.fillna(rf.max())

# -- Directory
EXP0_DIR = FIGURES_DIR / "Proof of concept 5" / f"encoders_{model}_{tuning}_{scoring}"

# Decide what kernels to use

kernels = {
    "mallows_auto": (ku.mallows_kernel, {"nu": "auto"}),
    "jaccard_1": (ku.jaccard_kernel, {"k": 1}),
    "borda_OHE": (ku.borda_kernel, {"idx": rf.index.get_loc("OHE")}),
    "borda_DE": (ku.borda_kernel, {"idx": rf.index.get_loc("DE")})
}

# Decide on the CI for N*
CI_LOWER = 0.05
CI_UPPER = 0.95

# ---- Main loop
for kernelname, (kernel, kernelargs) in tqdm(list(kernels.items())):

    EXP1_DIR = EXP0_DIR / f"{kernelname}"

    # for disjoint, replace in product([False, True], repeat=2):  # disjoint -> conservative, replace -> conservative
    for disjoint, replace in [(True, False)]:
        # ---- Create directories
        EXP2_DIR = EXP1_DIR / f"nstar_N_alpha={alpha}_eps={eps}_ci={lr_confidence}_disjoint={disjoint}_replace={replace}"
        for ED in [EXP0_DIR, EXP1_DIR, EXP2_DIR]:
            try:
                os.mkdir(ED)
            except FileExistsError:
                pass

        # ---- Computation
        dataset_pool = df["dataset"].unique()  # we remove datasets from it to simulate running new experiments
        datasets = np.array([])  # datasets on which we have already run experiments
        out = []
        plt.ioff()
        while len(dataset_pool) > 0:
            # -- Sample a new dataset
            # initialization with the minimum meaningful n, 4 if disjoint is True
            if len(datasets) == 0:
                datasets = np.random.default_rng(seed).choice(dataset_pool, 10, replace=False)
            else:
                datasets = np.append(datasets,
                                     np.random.default_rng(seed).choice(dataset_pool, 10, replace=False))
            dataset_pool = np.setdiff1d(dataset_pool, datasets)  # remove the sampled datasets rom the pool

            rf_ = rf.loc[:, datasets]
            na, nv = rf_.shape
            rankings = ru.SampleAM.from_rank_function_dataframe(rf_)

            # -- Compute the lower bound
            variance = ku.var(rankings, use_rf=True, kernel=kernel, **kernelargs)
            var_lower_bound = gu.sample_mean_embedding_lowerbound(eps, len(datasets), kbar=1,
                                                                  v=variance)

            # -- Compute mmds
            mmds = {
                n: mmd.subsample_mmd_distribution(rankings, subsample_size=n, rep=100,
                                                 use_rf=True, use_key=False, seed=seed,
                                                 disjoint=disjoint, replace=replace,
                                                 kernel=kernel, **kernelargs)
                for n in range(2, nv // 2 + 1)
            }

            # -- Compute generalizability and quantiles
            logepss = np.linspace(np.log(eps) - 0.1,
                                  np.log(max(np.quantile(mmde, alpha) for mmde in mmds.values())) + 0.1,
                                  1000)
            ys = {}
            qs = {}
            for n, mmde in mmds.items():
                ys[n] = [mmd.generalizability(mmde, np.exp(logeps)) for logeps in logepss]
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
            nstar_lower, nstar_upper = np.quantile(nstar_cv, [CI_LOWER, CI_UPPER])

            print("N*: ", nstar)
            print(f"N* CI from {CI_LOWER} to {CI_UPPER}: [{nstar_lower}, {nstar_upper}]")

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
            fig.suptitle(f"Generalizability for N = {len(datasets):02d}\n"
                         f"n*(alpha={alpha}, eps={eps}) = {np.ceil(nstar)}\n"
                         f"{lr_confidence} confidence interval: [{np.ceil(nstar_lower)}, {np.ceil(nstar_upper)}]")
            plt.tight_layout()
            # plt.savefig(EXP2_DIR / f"N={len(datasets):02d}.png")
            plt.savefig(EXP2_DIR / f"N={len(datasets):02d}.pdf")
            plt.close("all")

            out.append({
                "kernel": kernelname,
                "alpha": alpha,
                "eps": eps,
                "disjoint": disjoint,
                "replace": replace,
                "N": len(datasets),
                "nstar": nstar,
                "nstar_lower": nstar_lower,
                "nstar_upper": nstar_upper,
                "variance": variance,
                "var_lower_bound": var_lower_bound,
            })
        plt.ion()
