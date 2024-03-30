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

DATA_DIR = Path("../data/benchmark_encoders")
FIGURES_DIR = Path("../Figures")

mpl.use("TkAgg")
sns.set_style("ticks")
sns.set_context("notebook")
palette = "flare_r"
sns.set_palette("flare_r")

# ---- Parameters

# gif for png & gif, pdf for pdf
FORMAT = "gif"


EPS = 0.2  # max MMD is sqrt2 sqrt(kbar), i.e., sqrt2 in most applications
ALPHA = 0.95  # in 95% of cases, we'll get similar experimental results

# Decide on the CI for N*
LR_CONFIDENCE = 0.9
CI_LOWER = (1 - LR_CONFIDENCE)/2
CI_UPPER = LR_CONFIDENCE + CI_LOWER # confidence interval for linear prediction

seed = 1444


# ---- Load data

# Specify Dataset
DATA_SET = Path(DATA_DIR / "results.parquet")

# load Dataset into df
if DATA_SET.suffix == '.parquet':
    df = pd.read_parquet(DATA_SET)
elif DATA_SET.suffix == '.csv':
    df = pd.read_csv(DATA_SET)
else:
    raise Exception("Please use a Parquet or CSV file as the format of your data")

# Set query params
experimental_conditions ={
    "dataset": None,
    "model": "LR",
    "tuning": "no",
    "scoring": "ACC"
}

target = "cv_score"
alternatives = "encoder"

query_string = " and ".join(f"{key} == '{value}'" if isinstance(value, str) else f"{key} == {value}"
                            for key, value in experimental_conditions.items() if value is not None)

# Check if query params exist in the df
columns_to_check = set(experimental_conditions.keys()).union({target, alternatives})
missing_columns = columns_to_check - set(df.columns)
if missing_columns:
    raise ValueError(f"The following columns are missing from the dataframe: {missing_columns}")

# Build query
df = df.query(query_string).reset_index(drop=True)
rf = ru.get_rankings_from_df(df, factors=list(experimental_conditions.keys()), alternatives=alternatives,
                             target=target,
                             lower_is_better=False, impute_missing=True)
rf = rf.fillna(rf.max())

disjoint = True
replace = False


# -- Directory
EXP0_DIR = FIGURES_DIR / "Proof_of_concept_5" / "encoders_{model}_{tuning}_{scoring}".format(**experimental_conditions)

# Decide what kernels to use

kernels = {
    "mallows_auto": (ku.mallows_kernel, {"nu": "auto"}),
    "jaccard_1": (ku.jaccard_kernel, {"k": 1}),
    "borda_OHE": (ku.borda_kernel, {"idx": rf.index.get_loc("OHE")}),
    "borda_DE": (ku.borda_kernel, {"idx": rf.index.get_loc("DE")})
}

# ---- Main loop
for kernelname, (kernel, kernelargs) in tqdm(list(kernels.items())):

    EXP1_DIR = EXP0_DIR / f"{kernelname}"

    # ---- Create directories
    EXP2_DIR = EXP1_DIR / f"nstar_N_ALPHA={ALPHA}_eps={EPS}_ci={LR_CONFIDENCE}_disjoint={disjoint}_replace={replace}"
    for ED in [EXP0_DIR, EXP1_DIR, EXP2_DIR]:
        ED.mkdir(parents=True, exist_ok=True)

    # ---- Computation
    # TODO: Not dataset, but experimental conditions! change names
    #TODO: initialize the dataset pool with the experimental conditions, initialized earlier
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
        var_lower_bound = gu.sample_mean_embedding_lowerbound(EPS, len(datasets), kbar=1,
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
        logepss = np.linspace(np.log(EPS) - 0.1,
                                np.log(max(np.quantile(mmde, ALPHA) for mmde in mmds.values())) + 0.1,
                                1000)
        ys = {}
        qs = {}
        for n, mmde in mmds.items():
            ys[n] = [mmd.generalizability(mmde, np.exp(logeps)) for logeps in logepss]
            qs[n] = np.log(np.quantile(mmde, ALPHA))

        dfy = pd.DataFrame(ys)
        dfy["log(EPS)"] = logepss
        dfy = dfy.melt(var_name="n", value_name="generalizability", id_vars="log(EPS)")
        dfy["n"] = dfy["n"].astype(int)

        dfq = pd.DataFrame.from_dict(qs, orient="index").reset_index()
        dfq.columns = ["n", "log(EPS)"]
        dfq["log(n)"] = np.log(dfq["n"])

        # -- Fit a linear predictor with cross-validated confidence intervals for log(n) as function of log(EPS)
        X = dfq["log(EPS)"].to_numpy().reshape(-1, 1)
        y = dfq["log(n)"].to_numpy().reshape(-1, 1)

        linear_predictors = [LinearRegression().fit(X[tr], y[tr]) for tr, _ in KFold(len(y)).split(X)]
        
        # - Get all predictions TODO: Make list comprehension, write comment
        ns_pred_cv = []
        for lr_tmp in linear_predictors:
            ns_tmp = np.exp(lr_tmp.predict(logepss.reshape(-1, 1)).reshape(1, -1)[0])
            ns_pred_cv.append(ns_tmp)

        # - Refit a linear model on the entire data
        lr = LinearRegression()
        lr.fit(X, y)
        ns_pred = np.exp(lr.predict(logepss.reshape(-1, 1)).reshape(1, -1)[0])

        # -- Predict nstar TODO: summarize all 3 loops together into 1 loop
        nstar_cv = []
        for lr_tmp, ns_tmp in zip(linear_predictors, ns_pred_cv):
            nstar_tmp = ns_tmp[np.argmax(logepss > np.log(EPS))] if lr_tmp.coef_ != 0 else np.nan
            nstar_cv.append(nstar_tmp)

        nstar = ns_pred[np.argmax(logepss > np.log(EPS))] if lr.coef_ != 0 else np.nan
        nstar_lower, nstar_upper = np.quantile(nstar_cv, [CI_LOWER, CI_UPPER])

        print("N*: ", nstar)
        print(f"N* CI from {CI_LOWER} to {CI_UPPER}: [{nstar_lower}, {nstar_upper}]")

        # -- Plot
        fig, axes = plt.subplots(2, 1, sharex="all", figsize=(10, 8))

        # - Generalizability
        ax = axes[0]
        sns.lineplot(dfy, x="log(EPS)", y="generalizability", hue="n", ax=ax, palette=palette)
        ax.hlines(ALPHA, ls="--", xmin=np.min(logepss), xmax=np.max(logepss), color="black")
        for n in mmds.keys():
            ax.vlines(qs[n], ymin=0, ymax=ALPHA, ls=":")
        sns.despine(ax=ax)

        # - Quantiles
        ax = axes[1]
        ymax = max(ns_pred)
        sns.lineplot(dfq, x="log(EPS)", y="n", ax=ax, ls="", marker="o", hue="n", legend=False)
        for n in mmds.keys():
            ax.vlines(qs[n], ymin=n, ymax=ymax, ls=":")
        ax.vlines(np.log(EPS), ymin=0.1, ymax=ymax, color="black", ls="--")

        # - Linear regression
        sns.lineplot(x=logepss, y=ns_pred, color="green", ls="-.", ax=ax)
        for it, ns_tmp in enumerate(ns_pred_cv):
            if np.max(ns_tmp) > 1000:  # TODO: hard-coded threshold for broken confidence intervals, make it more reliable
                continue
            sns.lineplot(x=logepss, y=ns_tmp, color="green", ls="-.", alpha=0.5, ax=ax)

        # - N*
        ax.hlines(nstar, xmin=np.min(logepss), xmax=np.log(EPS), ls="-", color="red")
        ax.hlines(nstar_upper, xmin=np.min(logepss), xmax=np.log(EPS), ls="-", color="red", alpha=0.3)
        ax.hlines(nstar_lower, xmin=np.min(logepss), xmax=np.log(EPS), ls="-", color="red", alpha=0.3)

        ax.set_yscale("log")
        sns.despine(ax=ax)

        # - Finalize
        fig.suptitle(f"Generalizability for N = {len(datasets):02d}\n"
                        f"n*(alpha={ALPHA}, eps={EPS}) = {np.ceil(nstar)}\n"
                        f"{LR_CONFIDENCE} confidence interval: [{np.ceil(nstar_lower)}, {np.ceil(nstar_upper)}]")
        plt.tight_layout()
        if FORMAT == "pdf":
            plt.savefig(EXP2_DIR / f"N={len(datasets):02d}.pdf")
        else:
            plt.savefig(EXP2_DIR / f"N={len(datasets):02d}.png")
        plt.close("all")

        out.append({
            "kernel": kernelname,
            "alpha": ALPHA,
            "eps": EPS,
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
    if FORMAT == "gif":
        images = [iio.imread(image) for image in glob.glob(str(EXP2_DIR / "*.png"))]
        iio.mimwrite(EXP2_DIR / f"nstar.gif", images, duration=750,
                        loop=0)

    # -- Store nstar predictions
    out = pd.DataFrame(out)
    out.to_parquet(EXP2_DIR / "nstar.parquet")

# %%
