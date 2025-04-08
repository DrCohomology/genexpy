"""
Like test, but now we are using the same samples for kernel and MMD

generalizability-kernel relation.
Generalizability = alpha means that with probability alpha the results are conclusive.
    This means that the average kernel is high with probability alpha (probability in the sample)

take some data, get rankings, consider multiple n's, get generalizability, get kernel distribution, plot
"""

import numpy as np
import pandas as pd
import yaml

from importlib import reload
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

# #%% Computation
#
# rankings = ru.SampleAM.from_rank_function_dataframe(rank_matrix)
# _, nv = rank_matrix.shape
# for kernelname, (kernel, kernelargs, epsstar, deltastar) in KERNELS.items():
#     mmds = {
#         n: mmd.subsample_mmd_distribution(
#             rankings, subsample_size=n, rep=1000, use_rv=True, use_key=False,
#             seed=SEED, disjoint=DISJOINT, replace=REPLACE, kernel=kernel, **kernelargs
#         )
#         for n in range(2, nv // 2 + 1)
#     }
#
#     # Sample the distribution of MMD for varying sizes
#     dfmmd = pd.DataFrame(mmds).melt(var_name="n", value_name="eps")
#
#     # Compute generalizability and quantiles
#     logepss = np.linspace(np.log(epsstar) - 0.1, np.log(max(np.quantile(mmde, ALPHA) for mmde in mmds.values())) + 0.1,
#                           1000)  # the bounds are chosen to optimize
#     dfy = create_generalizability_dataframe(mmds, logepss)
#     dfq = create_quantiles_dataframe(mmds)
#
#     dfmmd.to_parquet(OUTPUT_DIR / f"{kernelname}_mmd.parquet")
#     dfy.to_parquet(OUTPUT_DIR / f"{kernelname}_generalizability.parquet")
#     dfq.to_parquet(OUTPUT_DIR / f"{kernelname}_quantiles.parquet")

#%% Resample kernel distributions

"""
Get the distribution of the mean kernel for different samples.
Try something else: percentage of pairs of rankings with kernel greater than some threshold delta.
"""

from importlib import reload
reload(ru)

SEED = 1444

delta = 0.05
# eps = np.sqrt(2 * (1 - np.exp(-delta)))
ksup = 1
kinf = np.exp(-1)

rankings = ru.SampleAM.from_rank_function_dataframe(rank_matrix)
_, nv = rank_matrix.shape
rep = 100
kdist = []
kgen = []
epss = np.linspace(0, 2*(ksup-kinf), 100)

for kernelname, (kernel, kernelargs, epsstar, deltastar) in KERNELS.items():
    # estimate the expected MMD
    sub3 = rankings.get_subsample(subsample_size=25, seed=SEED-1, use_rv=True, use_key=False, replace=True)
    gram33 = ku.square_gram_matrix(sub3, use_rv=True, kernel=kernel, **kernelargs)
    gram33 = gram33 - np.diag(gram33.diagonal())

    for n in tqdm(range(2, nv // 2 + 1)):
        mmds = np.array([], dtype=float)
        for i in range(rep):
            subsamples = rankings.get_subsamples_pair(subsample_size=n, seed=SEED + i, use_rv=True, use_key=False,
                                                      replace=False)
            sub1 = ru.SampleAM.from_rank_function_matrix(subsamples[0])
            sub2 = ru.SampleAM.from_rank_function_matrix(subsamples[1])

            gram11 = ku.square_gram_matrix(sub1, use_rv=True, kernel=kernel, **kernelargs)
            gram22 = ku.square_gram_matrix(sub2, use_rv=True, kernel=kernel, **kernelargs)
            gram12 = ku.gram_matrix(sub1, sub2, use_rv=True, kernel=kernel, **kernelargs)

            mmd12 = mmd.mmdb(sub1, sub2, use_rv=True, kernel=kernel, **kernelargs)
            mmdu12 = mmd.mmdu_squared(sub1, sub2, use_rv=True, kernel=kernel, **kernelargs)

            # to estimate the probability of sample average of kernel >= kup-n/2eps (lb on gen)
            sub4 = rankings.get_subsample(subsample_size=n, seed=SEED + 2*i, use_rv=True, use_key=False, replace=True)
            gram44 = ku.square_gram_matrix(sub4, use_rv=True, kernel=kernel, **kernelargs)
            gram44 = gram44 - np.diag(gram44.diagonal())

            kdist.append({
                "kernel": kernelname,
                "n": n,
                # "delta": delta,
                # "eps": eps,
                "k11": gram11.mean(),
                "k11_std": gram11.std(),
                "k22": gram22.mean(),
                "k22_std": gram22.std(),
                "k12": gram12.mean(),
                "k12_std": gram12.std(),
                "mmd12": mmd12,
                "mmdb12_squared": mmd12 ** 2,
                "mmdu12_squared": mmdu12,
                "expected_MMD2": 2/n * (ksup - gram33.mean()),  # Proposition A.7 in the updated paper, replacing E[x(X, Y] with its sample estimator
                "mean_k": gram12.mean()
            })
            mmds = np.append(mmds, mmd12)

        kgen.extend([{
            "kernel": kernelname,
            "n": n,
            # "delta": delta,
            "eps": eps,
            "gen": mmd.generalizability(mmds, eps=eps),
            "gen2": mmd.generalizability(mmds**2, eps=eps)
        } for eps in epss])

    dfk = pd.DataFrame(kdist)
    dfk.to_parquet(OUTPUT_DIR / f"{kernelname}_kernel_mmd.parquet")

    dfg = pd.DataFrame(kgen)
    dfg.to_parquet(OUTPUT_DIR / f"{kernelname}_gen_lame.parquet")

#%% 2. Plot MMD versus kernels

import matplotlib.pyplot as plt
import seaborn as sns

# df_ = pd.merge(dfk, dfg, on=["kernel", "n", "delta", "eps"])
df_ = dfk.copy()
df_ = df_.melt(id_vars="n", value_vars=["k11", "k22", "k12", "mmd12", "expected_MMD2"], var_name="stat")

fig, ax = plt.subplots()

sns.lineplot(df_, x="n", y="value", hue="stat", ax=ax, errorbar="sd")
ax.hlines(eps, xmin=0, xmax=25, ls=":", color="grey", label="eps")
ax.hlines(delta, xmin=0, xmax=25, ls=":", color="black", label="delta")

sns.despine()
plt.tight_layout()
plt.show()

#%% 2a. Plot expected MMD vs actual MMD

import matplotlib.pyplot as plt
import seaborn as sns

df_ = dfk.copy()
df_["expected_MMD2"] = (df_["expected_MMD2"])
df_["mmd12"] = df_["mmd12"]**2
df_ = df_.melt(id_vars="n", value_vars=["mmd12", "expected_MMD2"], var_name="stat")


fig, ax = plt.subplots()
sns.lineplot(df_, x="n", y="value", hue="stat", ax=ax, errorbar="sd")

sns.despine()
plt.tight_layout()
plt.show()

#%% 3. Check

# lower bound on mmd**2
lb = []
for eps in dfg["eps"].unique():
    for n in dfg["n"].unique():
        dfk_ = dfk.query("n == @n")
        kdist = dfk_["mean_k"].values
        klb = np.mean(kdist >= ksup - n/2 * eps)
        mmdestdist = (n-1)/n * dfk_["mmdu12_squared"] + 2/n * ksup - 2/n * dfk_["mean_k"]
        mmdestdist = mmdestdist.values
        mmdest = np.mean(mmdestdist <= eps)
        lb.append({"eps": eps, "n": n, "klb": klb, "mmdest": mmdest})
dflb = pd.DataFrame(lb)


# %% 3a. Check (pt2)

n = 10

df1_ = dfg.query("n == @n")
df2_ = dflb.query("n == @n")

x = "eps"
hue = "n"
palette = "flare_r"

fig, ax = plt.subplots()
sns.lineplot(df1_, x="eps", y="gen2", ax=ax, errorbar="sd", color="r", label="gen")
sns.lineplot(df2_, x="eps", y="klb", ax=ax, errorbar="sd", color="b", label="lb", linestyle=":")
sns.lineplot(df2_, x="eps", y="mmdest", ax=ax, errorbar="sd", color="green", label="genest", linestyle="--")

# sns.lineplot(dfg, x=x, y="gen2", ax=ax, errorbar="sd", hue=hue, palette=palette, legend=False)
# sns.lineplot(dflb, x=x, y="klb", ax=ax, errorbar="sd", hue=hue,  linestyle=":", palette=palette, legend=False)

# ax.set_xlim(0, 0.8)
ax.set_xscale("log")

sns.despine()
plt.tight_layout()
plt.show()


#%% 3b. Check







