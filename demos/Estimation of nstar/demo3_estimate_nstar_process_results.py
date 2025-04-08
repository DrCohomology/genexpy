"""
vectorized and small n test.
    try if one can get greater accuracy in estimating nstar from just two values of n (very small)/.
    it will require more repetitions, but it might be worth it in terms of runtime
"""


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
from genexpy import kernels_vectorized as kvu
from genexpy import rankings_utils as ru
from genexpy import mmd as mmd
from genexpy import probability_distributions as du

try:
    os.chdir(Path(os.getcwd()) / "demos" / "Estimation of nstar")
except FileNotFoundError:
    pass

with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

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

FIGURES_DIR = Path(config['paths']['figures_dir'])

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

#!! BROKEN BS: CAN ONLY LOAD ONE PARAMETER VALUE AT A TIME
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


#! DIRTY, BUT IT WORKS FOR TESTING ONLY
KERNELS = {k: list(v) for k, v in KERNELS.items()}
KERNELS["borda_kernel_nu_auto"][1] = {"nu": "auto", "idx": 0}
KERNELS = {k: tuple(v) for k, v in KERNELS.items()}
del KERNELS["borda_kernel_alternative_0"]


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

def predict_nstar(epss, linear_predictors, dfq, eps):
    logepss = np.log(epss)
    X = dfq[['log(eps)']].values
    y = dfq[['log(n)']].values

    ns_pred_cv = [np.exp(lr.predict(logepss.reshape(-1, 1)).reshape(-1)) for lr in linear_predictors]
    ns_pred = np.exp(LinearRegression().fit(X, y).predict(logepss.reshape(-1, 1)).reshape(-1))
    nstar_cv = [pred[np.argmax(logepss > np.log(eps))] for pred in ns_pred_cv if not np.all(pred == 0)]
    nstar = ns_pred[np.argmax(logepss > np.log(eps))]
    nstar_lower, nstar_upper = np.quantile(nstar_cv, [CI_LOWER, CI_UPPER])
    return ns_pred, ns_pred_cv, nstar, nstar_lower, nstar_upper

#%% Load results

OUTPUT_DIR = Path("outputs_redone")

@np.vectorize
def theoretical_nstar(alphastar, epsstar, kbar=1):
    beta1 = -2
    beta0 = 2*np.log(np.sqrt(2*kbar) + np.sqrt(-4*kbar * np.log(1-alphastar)))
    return np.exp(beta0 + beta1*np.log(epsstar))

# dfys = []
df_nstar = []
for distr_dir in OUTPUT_DIR.iterdir():
    # dfys_ = []
    nstar_true = []
    for kernel_name, (_, _, eps, delta) in KERNELS.items():
        dfy_tmp = pd.concat([pd.read_parquet(x)
                           for x in tqdm(list(distr_dir.glob(f"**/{kernel_name}/computed_generalizability/dfy_*.parquet")),
                                         desc="Loading dataframes")]).reset_index(drop=True)
        dfy_tmp["kernel"] = kernel_name
        nstar_true_tmp = dfy_tmp.query("N == 1000").loc[(dfy_tmp["log(eps)"] <= np.log(eps)) & (dfy_tmp["generalizability"] >= ALPHA)].groupby("repnum")["n"].min().rename("nstar_true")
        nstar_true_tmp = nstar_true_tmp.reset_index()
        nstar_true_tmp["kernel"] = kernel_name

        # dfys_.append(dfy_tmp)
        nstar_true.append(nstar_true_tmp)

    # dfys_ = pd.concat(dfys_)
    nstar_true = pd.concat(nstar_true)

    df_nstar_ = pd.concat([pd.read_parquet(x)
                           for x in tqdm(list(distr_dir.glob("**/**/**/nstar.parquet")),
                                         desc="Loading dataframes")]).reset_index(drop=True)
    # df_nstar_ = df_nstar_.join(nstar_true, on=["repnum", "kernel"])
    df_nstar_ = pd.merge(df_nstar_, nstar_true, on=["kernel", "repnum"])
    df_nstar_["distr"] = distr_dir.name
    df_nstar_["nstar_th"] = theoretical_nstar(df_nstar_["alpha"], df_nstar_["eps"], kbar=1)

    # dfys.append(dfys_)
    df_nstar.append(df_nstar_)

# dfys = pd.concat(dfys, axis=0).reset_index(drop=True)
df_nstar = pd.concat(df_nstar, axis=0).reset_index(drop=True)

df_nstar.to_parquet(OUTPUT_DIR / "nstar.parquet")






