"""
generalizability-kernel relation.
Generalizability = alpha means that with probability alpha the results are conclusive.
    This means that the average kernel is high with probability alpha (probability in the sample)

take some data, get rankings, consider multiple n's, get generalizability, get kernel distribution, plot
"""

import numpy as np
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


with open("test/config.yaml", 'r') as file:
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

if DATASET.suffix == '.parquet':
    df = pd.read_parquet(DATASET).drop(columns=["time"])
elif DATASET.suffix == '.csv':
    df = pd.read_csv(DATASET).drop(columns=["time"])
else:
    raise Exception("Please use a Parquet or CSV file as the format of your data")

# Check whether exactly one of the experimental factors is None
assert sum(value is None for value in EXPERIMENTAL_FACTORS.values()) == 1, "Exactly one experimental factor must be set to null in config.yaml."

# Check whether the factors listed in the config coincide with the columns of df
columns_to_check = set(EXPERIMENTAL_FACTORS.keys()).union({TARGET, ALTERNATIVES})
if not_in_df := columns_to_check - set(df.columns):
    raise ValueError(f"The following columns are missing from the dataframe: {not_in_df}")
if not_in_config:= set(df.columns) - columns_to_check:
    raise ValueError(f"The following columns in the dataframe are not required: {not_in_config}")

# query fot the held-constant factors
try:
    query_string = " and ".join(f"{factor} == '{lvl}'" if isinstance(lvl, str) else f"{factor} == {lvl}"
                                for factor, lvl in EXPERIMENTAL_FACTORS.items()
                                if lvl not in [None, "_all"])
    df = df.query(query_string)
except ValueError:
    pass

# get the combinations of levels of allowed-to-vary-factors
try:
    groups = df.groupby([factor for factor, lvl in EXPERIMENTAL_FACTORS.items() if lvl == "_all"]).groups
except ValueError:
    groups = {"None": df.index}


# Rank the alternatives
idf = df.reset_index(drop=True)
rank_matrix = ru.get_rankings_from_df(idf, factors=list(EXPERIMENTAL_FACTORS.keys()),
                                      alternatives=ALTERNATIVES,
                                      target=TARGET,
                                      lower_is_better=False, impute_missing=True)
# Impute the missing values
rank_matrix = rank_matrix.fillna(rank_matrix.max())

def create_generalizability_dataframe(mmds, logepss):
    ys = {n: [mmd.generalizability(mmde, np.exp(logeps)) for logeps in logepss] for n, mmde in mmds.items()}
    dfy = pd.DataFrame(ys, index=logepss).reset_index().melt(id_vars='index', var_name='n', value_name='generalizability')
    dfy.rename(columns={'index': 'log(eps)'}, inplace=True)
    dfy['n'] = dfy['n'].astype(int)
    return dfy

def create_quantiles_dataframe(mmds):
    qs = {n: np.log(np.quantile(mmde, ALPHA)) for n, mmde in mmds.items()}
    dfq = pd.DataFrame(list(qs.items()), columns=['n', 'log(eps)'])
    dfq['log(n)'] = np.log(dfq['n'])
    return dfq

#%% Computation

rankings = ru.SampleAM.from_rank_function_dataframe(rank_matrix)
_, nv = rank_matrix.shape
for kernelname, (kernel, kernelargs, epsstar, deltastar) in KERNELS.items():
    mmds = {
        n: mmd.subsample_mmd_distribution(
            rankings, subsample_size=n, rep=1000, use_rv=True, use_key=False,
            seed=SEED, disjoint=DISJOINT, replace=REPLACE, kernel=kernel, **kernelargs
        )
        for n in range(2, nv // 2 + 1)
    }

    # Sample the distribution of MMD for varying sizes
    dfmmd = pd.DataFrame(mmds).melt(var_name="n", value_name="eps")

    # Compute generalizability and quantiles
    logepss = np.linspace(np.log(epsstar) - 0.1, np.log(max(np.quantile(mmde, ALPHA) for mmde in mmds.values())) + 0.1,
                          1000)  # the bounds are chosen to optimize
    dfy = create_generalizability_dataframe(mmds, logepss)
    dfq = create_quantiles_dataframe(mmds)

    dfmmd.to_parquet(OUTPUT_DIR / f"{kernelname}_mmd.parquet")
    dfy.to_parquet(OUTPUT_DIR / f"{kernelname}_generalizability.parquet")
    dfq.to_parquet(OUTPUT_DIR / f"{kernelname}_quantiles.parquet")

#%% Resample kernel distributions

"""
Get the distribution of the mean kernel for different samples.
Try something else: percentage of pairs of rankings with kernel greater than some threshold delta.
"""

from importlib import reload
reload(ru)

delta = 0.2
eps = np.sqrt(2 * (1 - np.exp(-delta)))

rankings = ru.SampleAM.from_rank_function_dataframe(rank_matrix)
_, nv = rank_matrix.shape
rep = 100
kdist = []
for kernelname, (kernel, kernelargs, epsstar, deltastar) in KERNELS.items():
    for n in range(2, nv // 1):
        for i in range(rep):
            subsample = rankings.get_subsample(subsample_size=n, seed=SEED + i, use_rv=True, use_key=False,
                                               replace=True)  # replace=True ensures an iid sample
            gram = ku.square_gram_matrix(subsample, use_rv=True, kernel=kernel, **kernelargs)
            kdist.append({
                "kernel": kernelname,
                "n": n,
                "mean": gram.mean(),  # with main diagonal
                "std": gram.std(),  # with main diagonal
                "gtdp": (gram >= np.exp(-delta)).mean(),  # greater than delta'
            })
    dfk = pd.DataFrame(kdist)
    dfk.to_parquet(OUTPUT_DIR / f"{kernelname}_mean_kernel_distribution.parquet")




#%% Plots

import matplotlib.pyplot as plt
import seaborn as sns

dfy_= dfy.loc[dfy["log(eps)"] >= np.log(eps)]
dfy_ = dfy_.loc[dfy_["log(eps)"] == dfy_["log(eps)"].min()]

fig, ax = plt.subplots(1, 1)
sns.boxplot(data=dfk, x="n", y="mean", ax=ax, fill=False, showfliers=False, color="blue")
sns.boxplot(data=dfk, x="n", y="gtdp", ax=ax, fill=False, showfliers=False, color="green")
# sns.boxplot(data=dfk, x="n", y="std", hue="kernel", ax=ax, fill=False, showfliers=False, legend=False)

sns.lineplot(data=dfy_, x="n", y="generalizability", ax=ax, markers=".", legend=False, color="red")

plt.tight_layout()
plt.show()

#%% Try something different: instead of average kernel, percentage of pairs of rankings with

"""
In the light of the new findings, repeat the exercise but this time compute the average of the kernel between the two 
    samples.
    
1. get two subsamples
2. compute MMD
3. compute average kernel for both and between subsamples
4. does it make sense? 
    expectations: kernel within samples decreases (as seen from previous experiment), but kernel between samples decreases
        much faster
"""













