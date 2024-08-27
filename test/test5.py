"""
Like test4, but approximate MMDu2 with the distribution of average kernels.


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

#%% 2. Get the estimate for the limit distribution of MMDu2, based on Gretton (2012), mainly eq. (9) p737

reload(ru)
rankings = ru.SampleAM.from_rank_function_dataframe(rank_matrix)

n = 25  # for MMDu2
rep = 500  # number of samples for MMDu2
rep_kernel = 500  # number of samples for each kernel
neig = 100  # sample size to compute average kernel
nnormal = 500 # number of points sampled from normal distributions in (10)

kernelname = "mallows_kernel_nu_auto"
kernel, kernelargs, _, _ = KERNELS[kernelname]

# --- Empirical MMDu2
kdist = []
for i in tqdm(list(range(rep))):
    sub1, sub2 = rankings.get_subsamples_pair(subsample_size=n, seed=191919 + i, use_key=False, replace=False,
                                              disjoint=True)

    # TODO: double check. replace=False, disjoint=True gives MMDu2 for same distr, while replace=True, disjoint=False does not
    # But it should!

    n1 = len(sub1)
    n2 = len(sub2)

    gram11 = ku.square_gram_matrix(sub1, use_rv=True, kernel=kernel, **kernelargs)
    gram22 = ku.square_gram_matrix(sub2, use_rv=True, kernel=kernel, **kernelargs)
    gram12 = ku.gram_matrix(sub1, sub2, use_rv=True, kernel=kernel, **kernelargs)

    ugram11 = gram11 - np.diag(gram11.diagonal())
    ugram22 = gram22 - np.diag(gram22.diagonal())

    mmdb2 = gram11.mean() + gram22.mean() - 2 * gram12.mean()
    mmdu2 = np.sum(ugram11) / (n1 * (n1 - 1)) + np.sum(ugram22) / (n2 * (n2 - 1)) - 2 * np.mean(gram12)

    kdist.append({
        "kernel": kernelname,
        "n": n,
        "k11": gram11.mean(),
        "uk11": np.sum(gram11) / (n * (n - 1)),
        "k22": gram22.mean(),
        "uk22": np.sum(gram22) / (n * (n - 1)),
        "k12": gram12.mean(),
        "mmdb2": mmdb2,
        "mmdu2": mmdu2,
        # "mean_k": gram44.mean()
    })

# approximate MMDu2
mmdu2_approx = np.array([])
for i in tqdm(list(range(rep_kernel))):
    subs = [rankings.get_subsample(subsample_size=neig, seed=141414 + 3*i + j, use_key=False, replace=True)
            for j in range(3)]
    grams = [ku.square_gram_matrix(sub, use_rv=True, kernel=kernel, **kernelargs) for sub in subs]
    kms = [gram.mean() for gram in grams]
    mmdu2_approx = np.append(mmdu2_approx, kms[0] + kms[1] - 2*kms[2])

#%% 2a. Plot
# Compare
df_ = pd.DataFrame(kdist)

fig, axes = plt.subplots(2, 2)
fig.suptitle(f"Number of eigenvalues: {neig}. MMDu2 sample size: {n}")

axes = axes.flatten()

ax = axes[0]
sns.histplot(data=df_, x="mmdu2", ax=ax, stat="probability", bins=100)

ax = axes[1]
sns.histplot(mmdu2_approx, ax=ax, stat="probability", bins=100)
ax.set_xlabel("mmdu2_approx")

"""
This distribution looks like the one in Gretton (2012) portraying th distribution of MMDu2 under the ALTERNATIVE 
    hypothesis. something is wrong here. 
"""

"""
Moreover, the distribution of the aproximation seems to have the tail in the wrong direction (towards the negatives). 
"""

ax = axes[2]


ax = axes[3]
sns.ecdfplot(data=df_, x="mmdu2", ax=ax, stat="proportion", label="mmdu2")
sns.ecdfplot(mmdu2_approx, ax=ax, stat="proportion", label="mmdu2_approx")
ax.set_xlabel("")
ax.set_ylabel("ecdf")
ax.legend()

sns.despine()
plt.tight_layout()
plt.show()

