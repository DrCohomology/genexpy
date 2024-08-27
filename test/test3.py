"""
Extend test2 to probabilitz distributions. Test an exact relation between MMDb and K.

generalizability-kernel relation.
Generalizability = alpha means that with probability alpha the results are conclusive.
    This means that the average kernel is high with probability alpha (probability in the sample)

take some data, get rankings, consider multiple n's, get generalizability, get kernel distribution, plot
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

            n = len(sub1)
            m = len(sub2)

            gram11 = ku.square_gram_matrix(sub1, use_rv=True, kernel=kernel, **kernelargs)
            gram22 = ku.square_gram_matrix(sub2, use_rv=True, kernel=kernel, **kernelargs)
            gram12 = ku.gram_matrix(sub1, sub2, use_rv=True, kernel=kernel, **kernelargs)

            ugram11 = gram11 - np.diag(gram11.diagonal())
            ugram22 = gram22 - np.diag(gram22.diagonal())

            mmdb2 = gram11.mean() + gram22.mean() - 2*gram12.mean()
            mmdu2 = np.sum(ugram11)/(n*(n-1)) + np.sum(ugram22)/(m*(m-1)) - 2 * np.mean(gram12)

            # to estimate the probability of sample average of kernel >= kup-n/2eps (lb on gen)
            # sub4 = rankings.get_subsample(subsample_size=n, seed=SEED + 2*i, use_rv=True, use_key=False, replace=True)
            # gram44 = ku.square_gram_matrix(sub4, use_rv=True, kernel=kernel, **kernelargs)
            # gram44 = gram44 - np.diag(gram44.diagonal())

            kdist.append({
                "kernel": kernelname,
                "n": n,
                "k11": gram11.mean(),
                "uk11": np.sum(gram11)/(n*(n-1)),
                "k22": gram22.mean(),
                "uk22": np.sum(gram22) / (n * (n - 1)),
                "k12": gram12.mean(),
                "mmdb2": mmdb2,
                "mmdu2": mmdu2,
                # "mean_k": gram44.mean()
            })
            mmds = np.append(mmds, np.sqrt(mmdb2))

        kgen.extend([{
            "kernel": kernelname,
            "n": n,
            "eps": eps,
            "gen": mmd.generalizability(mmds, eps=eps),
            "gen2": mmd.generalizability(mmds**2, eps=eps)
        } for eps in epss])

    dfk = pd.DataFrame(kdist)
    dfk.to_parquet(OUTPUT_DIR / f"{kernelname}_kernel_mmd.parquet")

    dfg = pd.DataFrame(kgen)
    dfg.to_parquet(OUTPUT_DIR / f"{kernelname}_gen_lame.parquet")

#%% 2. get probability mass of kernel in window with independence assumption (not working)

ksup = 1
kinf = np.exp(-1)
deltas = np.linspace(0, 2*(ksup-kinf), 100)

est = []
# lower bound on mmd**2
for n in tqdm(dfg["n"].unique()):     # subsample size
    dfk_ = dfk.query("n == @n")
    x = 2 * dfk_["mean_k"]
    for eps in dfg["eps"].unique():  # quantile
        for delta in deltas:        # integration variable
            cond2 = -n * (eps - delta) + 2 * ksup
            cond1 = n * delta + (1 - n) * eps + 2 * (n - 1) / n * ksup
            window_prob = np.mean((x >= cond2) & (x <= cond1))
            est.append({"n": n,
                        "eps": eps,
                        "delta": delta,
                        "window_lower": cond2,
                        "window_upper": cond1,
                        "window_prob": window_prob})
dfest = pd.DataFrame(est)

dfprob = dfest.groupby(["n", "eps"])["window_prob"].sum().reset_index()
dfprob["prob"] = dfprob["window_prob"] / len(deltas)

# check the non-empty interval condition
dfest["nonempty_condition"] = dfest["eps"] - 2/dfest["n"]*ksup
dfest["nonempty"] = (dfest["nonempty_condition"] > 0)
dfest["check_condition"] = dfest["window_upper"] - dfest["window_lower"]        # check passed! interval is correct

# %% 2a. plot

n = 2

df1_ = dfg.query("n == @n")
df2_ = dfprob.query("n == @n")

x = "eps"
hue = "n"
palette = "flare_r"

fig, ax = plt.subplots()
sns.lineplot(df1_, x=x, y="gen2", ax=ax, errorbar="sd", color="r", label="gen2")
sns.lineplot(df2_, x=x, y="prob", ax=ax, errorbar="sd", color="b", label="estprob", linestyle=":")

sns.despine()
plt.tight_layout()
plt.show()

# %% 3. get probability mass of kernel in window without independence assumption

ksup = 1
kinf = np.exp(-1)

delta_min = 0
delta_max = 2*(ksup-kinf)
deltas = np.linspace(delta_min, delta_max, 100)

est = []
# lower bound on mmd**2
for n in tqdm(dfg["n"].unique()):     # subsample size
    dfk_ = dfk.query("n == @n")
    uk11 = dfk_["uk11"].values
    uk22 = dfk_["uk22"].values
    k12 = dfk_["k12"].values
    for eps in dfg["eps"].unique():  # quantile
        for delta in deltas:        # integration variable
            cond2 = -n / 2 * (eps - delta) + ksup
            cond1 = n * delta + (1 - n) * eps + 2 * (n - 1) / n * ksup
            window_prob = np.mean((k12 >= cond2) & (uk11 + uk22 <= cond1))
            est.append({"n": n,
                        "eps": eps,
                        "delta": delta,
                        "cond2": cond2,
                        "cond1": cond1,
                        "window_prob": window_prob})
dfest = pd.DataFrame(est)

dfprob = dfest.groupby(["n", "eps"])["window_prob"].sum().reset_index()
dfprob["prob"] = dfprob["window_prob"] * (delta_max-delta_min) / len(deltas)

# check the non-empty interval condition
dfest["nonempty_condition"] = dfest["eps"] - 2/dfest["n"]*ksup
dfest["nonempty"] = (dfest["nonempty_condition"] > 0)
dfest["check_condition"] = dfest["cond1"] - 2*dfest["cond2"]        # check passed! interval is correct

print("problem if true: ", (dfest["nonempty"] & (dfprob["window_prob"] > 0)).any())

# %% 3a. Plot

n = 25

df1_ = dfg.query("n == @n")
df2_ = dfprob.query("n == @n")

x = "eps"
hue = "n"
palette = "flare_r"

fig, ax = plt.subplots()
sns.lineplot(df1_, x=x, y="gen2", ax=ax, errorbar="sd", color="r", label="gen2")
sns.lineplot(df2_, x=x, y="prob", ax=ax, errorbar="sd", color="b", label="estprob", linestyle=":")

sns.despine()
plt.tight_layout()
plt.show()

# %% 4. Condition without changes

ksup = 1
kinf = np.exp(-1)
delta_min = dfk["mmdu2"].min()
# delta_max = 2*(ksup-kinf)
delta_max = dfk["mmdu2"].max()
deltas = np.linspace(delta_min, delta_max, 100)

est = []
# lower bound on mmd**2
for n in tqdm(dfg["n"].unique()):     # subsample size
    dfk_ = dfk.query("n == @n")
    mmdu2 = dfk_["mmdu2"].values
    mmdb2 = dfk_["mmdb2"].values
    k12 = dfk_["k12"].values
    for eps in dfg["eps"].unique():  # quantile
        gen2 = mmd.generalizability(mmdb2, eps)
        for delta in deltas:        # integration variable
            cond1 = (n-1)/n * mmdu2 <= delta
            cond2 = 2/n * (ksup-k12) <= eps - delta
            window_prob = np.mean(cond1 & cond2)
            window_prob1 = np.mean((n-1)/n * mmdu2 + 2/n * (ksup-k12) <= eps)
            est.append({"n": n,
                        "eps": eps,
                        "delta": delta,
                        "window_prob": window_prob,
                        "window_prob1": window_prob1,
                        "gen2": gen2})
dfest = pd.DataFrame(est)
#%%
dfprob = dfest.groupby(["n", "eps"])[["window_prob1", "gen2"]].mean().reset_index()
dfprob["prob"] = dfprob["window_prob1"] * (delta_max-delta_min) * len(deltas)

# %% 4a. Plot

n = 25

df1_ = dfg.query("n == @n")
df2_ = dfprob.query("n == @n")

x = "eps"
hue = "n"
palette = "flare_r"

fig, ax = plt.subplots()
sns.lineplot(df1_, x=x, y="gen2", ax=ax, errorbar="sd", color="r", label="gen2")
sns.lineplot(df2_, x=x, y="window_prob1", ax=ax, errorbar="sd", color="g", label="estprob", linestyle="--")
# sns.lineplot(df2_, x=x, y="window_prob1", ax=ax, errorbar="sd", color="b", label="estprob", linestyle=":")

# ax.set_yscale("log")

sns.despine()
plt.tight_layout()
plt.show()

#%% 4a1. plot on eps instead of n

eps = 0.613

dfprob["eps"] = dfprob["eps"].round(3)

df1_ = dfprob.query("eps == @eps")

fig, ax = plt.subplots()
sns.lineplot(df1_, x="n", y="gen2", ax=ax, errorbar="sd", color="r", label="gen2")
sns.lineplot(df1_, x="n", y="prob", ax=ax, errorbar="sd", color="b", label="estprob", linestyle=":")

ax.set_yscale("log")

sns.despine()
plt.tight_layout()
plt.show()


# %% test

dfk["estmmdb2"] = (dfk["n"]-1)/dfk["n"] * dfk["mmdu2"] + 2/dfk["n"] * ksup - 2/dfk["n"] * dfk["k12"]

fig, ax = plt.subplots()
sns.lineplot(dfk, x="n", y="mmdb2", ax=ax, errorbar="sd", color="r", label="mmdb2")
sns.lineplot(dfk, x="n", y="estmmdb2", ax=ax, errorbar="sd", color="b", label="estmmdb2", linestyle=":")

ax.set_xscale("log")
ax.set_yscale("log")

ax.legend()
sns.despine()
plt.tight_layout()
plt.show()
