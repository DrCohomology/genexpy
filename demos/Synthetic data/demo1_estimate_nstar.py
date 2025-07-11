import numpy as np
import os
import pandas as pd
import yaml

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from genexpy import lower_bounds as gu
from genexpy import kernels as ku
from genexpy import rankings_utils as ru
from genexpy import mmd as mmd
from genexpy import probability_distributions as du

os.chdir(Path(os.getcwd()) / "demos" / "Synthetic data")

with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = Path(config['paths']['output_dir'])
FIGURES_DIR = Path(config['paths']['figures_dir'])

SEED = config['parameters']['seed']
RNG = np.random.default_rng(SEED)
ALPHA = config['parameters']['alpha']
LR_CONFIDENCE = config['parameters']['lr_confidence']
CI_LOWER = (1 - LR_CONFIDENCE) / 2
CI_UPPER = LR_CONFIDENCE + CI_LOWER

DATASET = Path(config['data']['dataset_path'])
EXPERIMENTAL_FACTORS = config['data']['experimental_factors']
TARGET = config['data']['target']
ALTERNATIVES = config['data']['alternatives']

SAMPLE_SIZE = config['sampling']['sample_size']
DISJOINT = config['sampling']['disjoint']
REPLACE = config['sampling']['replace']

KERNELS = {}
for kernel_config in config['kernels']:
    kernel_func = getattr(ku, kernel_config['kernel'], None)

    if kernel_func:
        delta = kernel_config['delta']  # to get epsilon
        match kernel_config['kernel']:
            case "mallows_kernel":
                eps = np.sqrt(2 * (1 - np.exp(-delta)))  # assumes nu = 1/binom(n, 2)
            case "jaccard_kernel":
                eps = np.sqrt(2 * (1 - (1 - delta)))
            case "borda_kernel":
                eps = np.sqrt(2 * (1 - np.exp(-delta)))  # assumes nu = 1/n
            case _:
                raise ValueError(
                    f"The kernel {kernel_config['kernel']} must be either the Jaccard, Mallows, or Borda kernel.")

        for param_key, param_values in kernel_config['params'].items():
            if isinstance(param_values, list):
                for value in param_values:
                    params = {param_key: value}
                    kernel_name = f"{kernel_config['kernel']}_{param_key}_{value}"
                    KERNELS[kernel_name] = (kernel_func, params, eps, delta)
            else:
                params = {param_key: param_values}
                kernel_name = f"{kernel_config['kernel']}_{param_key}_{param_values}"
                KERNELS[kernel_name] = (kernel_func, params, eps, delta)
    else:
        print(f"Kernel function '{kernel_config['kernel']}' not found in module 'kernels'.")
#%%
def create_quantiles_dataframe(mmds):
    qs = {n: np.log(np.quantile(mmde, ALPHA)) for n, mmde in mmds.items()}
    dfq = pd.DataFrame(list(qs.items()), columns=['n', 'log(eps)'])
    dfq['log(n)'] = np.log(dfq['n'])
    return dfq

def perform_linear_regression_with_cv(dfq):
    # Extracting features and target from DataFrame
    X = dfq[['log(eps)']].values
    y = dfq[['log(n)']].values

    cv = KFold(n_splits=len(y))
    residuals, linear_predictors = [], []

    for train_index, test_index in cv.split(X):
        lr = LinearRegression().fit(X[train_index], y[train_index])
        predicted = lr.predict(X[test_index])
        residuals.extend(y[test_index] - predicted)

        linear_predictors.append(lr)

    return linear_predictors, residuals

def predict_nstar(logepss, linear_predictors, dfq, eps):
    X = dfq[['log(eps)']].values
    y = dfq[['log(n)']].values

    ns_pred_cv = [np.exp(lr.predict(logepss.reshape(-1, 1)).reshape(-1)) for lr in linear_predictors]
    ns_pred = np.exp(LinearRegression().fit(X, y).predict(logepss.reshape(-1, 1)).reshape(-1))
    nstar_cv = [pred[np.argmax(logepss > np.log(eps))] for pred in ns_pred_cv if not np.all(pred == 0)]
    nstar = ns_pred[np.argmax(logepss > np.log(eps))]
    nstar_lower, nstar_upper = np.quantile(nstar_cv, [CI_LOWER, CI_UPPER])
    return ns_pred, ns_pred_cv, nstar, nstar_lower, nstar_upper
#%% Computations

distr = du.UniformDistribution(na=5, seed=42, ties=True)
for repnum in tqdm(range(50), desc="repnum"):
    universe = distr.sample(1000)
    for kernelname, (kernel, kernelargs, epsstar, deltastar) in list(KERNELS.items()):

        # Iteratively sample from ec_pool increasing the sample size at every iteration
        # The sampled experimental conditions are used to approximate the distribution of true results

        exp0_dir = OUTPUT_DIR / f"configuration_{repnum}"
        exp1_dir = exp0_dir / f"{kernelname}"
        exp21_dir = exp1_dir / f"nstar_N_ALPHA={ALPHA}_delta={epsstar}_ci={LR_CONFIDENCE}"
        exp21_dir.mkdir(parents=True, exist_ok=True)
        exp22_dir = exp1_dir / "computed_generalizability"
        exp22_dir.mkdir(parents=True, exist_ok=True)
        exp23_dir = exp1_dir / "computed_quantiles"
        exp23_dir.mkdir(parents=True, exist_ok=True)
        nstar_dir, gen_dir, quant_dir = exp21_dir, exp22_dir, exp23_dir

        out = []
        for N in [1000, 10, 20, 40, 80]:  # size of empirical study, the first N corresponds to an ideal study

            rankings = universe.get_subsample(subsample_size=N, seed=43*repnum)

            # Sample the distribution of MMD for varying sizes
            mmds = {
                n: mmd.mmd_distribution(rankings, subsample_size=n, seed=SEED, rep=100, use_rv=True, use_key=False,
                                        replace=REPLACE, disjoint=DISJOINT, kernel=kernel, **kernelargs)
                for n in range(2, min(30, N // 2 + 1))
            }

            dfmmd = pd.DataFrame(mmds).melt(var_name="n", value_name="eps")
            dfmmd["repnum"] = repnum

            # Compute generalizability and quantiles
            logepss = np.linspace(np.log(epsstar) - 0.1,
                                  # np.log(max(np.quantile(mmde, ALPHA) for mmde in mmds.values())) + 0.1,
                                  0.5 * np.log(2),
                                  1000)

            ys = {n: [mmd.generalizability(mmde, np.exp(logeps)) for logeps in logepss] for n, mmde in mmds.items()}
            dfy = pd.DataFrame(ys, index=logepss).reset_index().melt(id_vars='index', var_name='n',
                                                                     value_name='generalizability')
            dfy.rename(columns={'index': 'log(eps)'}, inplace=True)
            dfy["kernel"] = kernelname
            dfy['n'] = dfy['n'].astype(int)
            dfy["N"] = N
            dfy["repnum"] = repnum

            dfq = create_quantiles_dataframe(mmds)

            # if the log of any quantile is -inf, the predicted nstar is 1
            if np.isinf(dfq["log(eps)"]).any():
                nstar, nstar_lower, nstar_upper = 1, 1, 1
                singular = True
            else:
                # Linear regression with cross-validation
                linear_predictors, residuals = perform_linear_regression_with_cv(dfq)

                # Predictions
                ns_pred, ns_pred_cv, nstar, nstar_lower, nstar_upper = predict_nstar(logepss, linear_predictors, dfq,
                                                                                     epsstar)
                singular = False

            # Store results
            result_dict = {
                "kernel": kernelname,
                "alpha": ALPHA,
                "eps": epsstar,
                "delta": deltastar,
                "disjoint": DISJOINT,
                "replace": REPLACE,
                "N": N,
                "repnum": repnum,
                "nstar": nstar,
                "nstar_lower": nstar_lower,
                "nstar_upper": nstar_upper,
                # "variance": variance,
                # "var_lower_bound": var_lower_bound,
                "singular": singular
            }
            # result_dict.update(factors_dict)
            out.append(result_dict)

            dfy.to_parquet(gen_dir / f"dfy_{N}.parquet")
            # dfq.to_parquet(quant_dir / f"dfq_{len(ecs)}_{ALPHA}.parquet")  # not needed
            dfmmd.to_parquet(quant_dir / f"dfmmd_{N}.parquet")

        # Store nstar predictions
        out = pd.DataFrame(out)
        out.to_parquet(nstar_dir / "nstar.parquet")

#%% fix: add kernel column to dfy (DO NOT RUN)

# for x in tqdm(list(OUTPUT_DIR.glob("**/**/**/dfy_*.parquet")), desc="Updating dataframes"):
#     d = pd.read_parquet(x)
#     if "jaccard" in str(x):
    #     d["kernel"] = "jaccard_kernel_k_1"
    # elif "mallows" in str(x):
    #     d["kernel"] = "mallows_kernel_nu_auto"
    # else:
    #     raise Exception
    # d.to_parquet(x)

#%% True n*

@np.vectorize
def theoretical_nstar(alphastar, epsstar, kbar=1):
    beta1 = -2
    beta0 = 2*np.log(np.sqrt(2*kbar) + np.sqrt(-4*kbar * np.log(1-alphastar)))
    return np.exp(beta0 + beta1*np.log(epsstar))

dfys = pd.concat([pd.read_parquet(x)
                      for x in tqdm(list(OUTPUT_DIR.glob("**/**/**/dfy_*.parquet")),
                                    desc="Loading dataframes")]).reset_index(drop=True)

# usually, epssar is very similar for all kernels
es = [t[2] for t in KERNELS.values()]
assert np.max(es) - np.min(es) < 0.01, "Epsstar are too different. Improve the code."
epsstar = np.mean(es)
nstar_true = dfys.query("N == 1000").loc[(dfys["log(eps)"] <= np.log(epsstar)) & (dfys["generalizability"] >= ALPHA)].groupby(["repnum", "kernel"])["n"].min().rename("nstar_true")

# predicted and theoretical nstars
df_nstar = pd.concat([pd.read_parquet(x)
                      for x in tqdm(list(OUTPUT_DIR.glob("**/**/**/nstar.parquet")),
                                    desc="Loading dataframes")]).reset_index(drop=True)
df_nstar = df_nstar.join(nstar_true, on=["repnum", "kernel"])
df_nstar["nstar_th"] = theoretical_nstar(df_nstar["alpha"], df_nstar["eps"], kbar=1)

#%% Plotting

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

@np.vectorize
def theoretical_nstar(alphastar, epsstar, kbar=1):
    beta1 = -2
    beta0 = 2*np.log(np.sqrt(2*kbar) + np.sqrt(-4*kbar * np.log(1-alphastar)))
    return np.exp(beta0 + beta1*np.log(epsstar))

sns.set(style="ticks", context="paper", font="times new roman")

# mpl.use("TkAgg")
# mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r"""
    \usepackage{mathptmx}
    \usepackage{amsmath}
    \usepackage{nicefrac}
"""
mpl.rc('font', family='Times New Roman')

# pretty names
pc = {"alpha": r"$\alpha^*$", "eps": r"$\varepsilon^*$", "nstar": r"$n^*$", "delta": r"$\delta^*$", "N": r"$N$", "nstar_rel_error": r"$\frac{n^*-\hat{n}^*_N}{n^*}$"}  # columns
pk = {"borda_kernel_idx_OHE": r"$\kappa_b^{\text{OHE}, 1/n}$", "mallows_kernel_nu_auto": r"$\kappa_m^{1/\binom{n}{2}}$", "jaccard_kernel_k_1": r"$\kappa_j^{1}$"}  # kernels
pk.update({"jaccard_kernel_k_1": r"$g_2$","mallows_kernel_nu_auto": r"$g_3$"})

# get prediction error
df_ = df_nstar.copy()
df_["nstar_error"] = df_["nstar"] - df_["nstar_true"]
df_["nstar_relative_error"] = (df_["nstar"] - df_["nstar_true"]) / df_["nstar_true"]
df_["nstar_absolute_error"] = np.abs(df_["nstar"] - df_["nstar_true"])
df_["nstar_absrel_error"] = np.abs(df_["nstar"] - df_["nstar_true"]) / df_["nstar_true"]
df_["nstar_rel_error"] = (df_["nstar"] - df_["nstar_true"]) / df_["nstar_true"]
# df_ = df_.loc[df_["N"] != df_["Nmax"]]

dfplot = df_.copy().query("N < 1000").rename(columns=pc)
dfplot["kernel"] = dfplot["kernel"].map(pk)

y = pc["nstar_rel_error"]

fig, ax = plt.subplots(1, 1, figsize=(5.5/2, 2))

sns.boxplot(dfplot, x=pc["N"], y=y, showfliers=False, fliersize=0.3, hue="kernel", palette="cubehelix", ax=ax,
            legend=True, linewidth=1.2, fill=False, gap=0.25)

ax.grid(color="grey", alpha=.2)
ax.set_yticks([-0.6, -0.3, 0, 0.3, 0.6])
ax.axhline(0, c="black", lw=0.5)


ax.legend(*ax.get_legend_handles_labels()).get_frame().set_edgecolor("w")
sns.despine()
plt.tight_layout(pad=.5)

plt.savefig(FIGURES_DIR / "synthetic_nstar_rel_error.pdf")
plt.show()

