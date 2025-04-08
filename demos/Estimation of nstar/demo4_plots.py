"""
vectorized and small n test.
    try if one can get greater accuracy in estimating nstar from just two values of n (very small)/.
    it will require more repetitions, but it might be worth it in terms of runtime
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import yaml

from pathlib import Path

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
FIGURES_DIR.mkdir(exist_ok=True)

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

OUTPUT_DIR = Path("outputs_redone")
df_nstar = pd.read_parquet(OUTPUT_DIR / "nstar.parquet")

plt.close("all")

sns.set(style="ticks", context="paper", font="times new roman")

# mpl.use("TkAgg")
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r"""
    \usepackage{mathptmx}
    \usepackage{amsmath}
"""
mpl.rc('font', family='Times New Roman')
ylim = 1


# pretty names
pretty_columns = {"alpha": r"$\alpha^*$", "eps": r"$\varepsilon^*$", "nstar": r"$n^*$", "delta": r"$\delta^*$", "N": r"$N$",
      "nstar_rel_error": "$(n^*_N-n^*) / n^*$", "nstar_ratio_error": "$n^*_N / n^*$"}  # columns
pretty_kernels = {"borda_kernel_nu_auto": r"$\kappa_b^{0, 1/n}$", "mallows_kernel_nu_auto": r"$\kappa_m^{1/\binom{n}{2}}$",
      "jaccard_kernel_k_1": r"$\kappa_j^{1}$"}  # kernels
pretty_distrs = {
    'Uniform(na=2, ties=True)': r"Uniform$_{n_a=2}$",
    'Uniform(na=4, ties=True)': r"Uniform$_{n_a=4}$",
    'Uniform(na=6, ties=True)': r"Uniform$_{n_a=6}$",
    'Uniform(na=8, ties=True)': r"Uniform$_{n_a=8}$",
    'Uniform(na=10, ties=True)': r"Uniform$_{n_a=10}$",
    'Uniform(na=15, ties=True)': r"Uniform$_{n_a=15}$",
    '1Degenerate(na=4, ties=True)': r"1-Degenerate$_{n_a=4}$",
    '9Degenerate(na=4, ties=True)': r"9-Degenerate$_{n_a=4}$",
    '18Degenerate(na=4, ties=True)': r"18-Degenerate$_{n_a=4}$",
    '37Degenerate(na=4, ties=True)': r"37-Degenerate$_{n_a=4}$",
    'Spike(na=4, ties=True)': "Spike$_{n_a=4}$",
}
# to sort the distributions
pretty_distrs_sort = {
    'Uniform(na=2, ties=True)': r"Uniform$_{n_a=02}$",
    'Uniform(na=4, ties=True)': r"Uniform$_{n_a=04}$",
    'Uniform(na=6, ties=True)': r"Uniform$_{n_a=06}$",
    'Uniform(na=8, ties=True)': r"Uniform$_{n_a=08}$",
    'Uniform(na=10, ties=True)': r"Uniform$_{n_a=10}$",
    'Uniform(na=15, ties=True)': r"Uniform$_{n_a=15}$",
    '1Degenerate(na=4, ties=True)': r"01-Degenerate$_{n_a=4}$",
    '9Degenerate(na=4, ties=True)': r"09-Degenerate$_{n_a=4}$",
    '18Degenerate(na=4, ties=True)': r"18-Degenerate$_{n_a=4}$",
    '37Degenerate(na=4, ties=True)': r"37-Degenerate$_{n_a=4}$",
    'Spike(na=4, ties=True)': "Spike$_{n_a=4}$",
}


#%% 1a. Plotting conditional on distribution

# get prediction error
df_ = df_nstar.copy()
df_["nstar_error"] = df_["nstar"] - df_["nstar_true"]
df_["nstar_relative_error"] = (df_["nstar"] - df_["nstar_true"]) / df_["nstar_true"]
df_["nstar_absolute_error"] = np.abs(df_["nstar"] - df_["nstar_true"])
df_["nstar_absrel_error"] = np.abs(df_["nstar"] - df_["nstar_true"]) / df_["nstar_true"]
df_["nstar_rel_error"] = (df_["nstar"] - df_["nstar_true"]) / df_["nstar_true"]
df_["nstar_ratio_error"] = df_["nstar"] / df_["nstar_true"]
# df_ = df_.loc[df_["N"] != df_["Nmax"]]

dfplot = df_.copy().query("N < 1000").query("distr != 'Spike(na=4, ties=True)'").rename(columns=pretty_columns)
dfplot["kernel"] = dfplot["kernel"].map(pretty_kernels)
dfplot["distr"] = dfplot["distr"].map(pretty_distrs_sort)
dfplot = dfplot.sort_values("distr")
dfplot["distr"] = dfplot["distr"].map({v: k for k, v in pretty_distrs_sort.items()}).map(pretty_distrs)

error_toplot = "nstar_rel_error"
y = pretty_columns[error_toplot]

hue_order = dfplot["kernel"].sort_values().unique()

fig, axes = plt.subplots(5, 2, figsize=(5.5, 8), sharex=False, sharey="all")
for ax, distr in zip(axes.flatten(), dfplot["distr"].unique()):
    ax.set_title(f"{distr}")
    dfplot_ = dfplot.query("distr == @distr")

    ax = sns.boxplot(dfplot_, x=pretty_columns["N"], y=y, showfliers=False, fliersize=0.3, hue="kernel", palette="cubehelix",
                ax=ax, legend=True, linewidth=1.2, fill=False, gap=0.5, hue_order=hue_order)

    ax.set_ylim(-ylim, ylim)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.axhline(0, color="black", zorder=-1, alpha=0.3)
    ax.grid(axis="y", zorder=-1, alpha=0.3)


    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()

    # ax.legend(*ax.get_legend_handles_labels()).get_frame().set_edgecolor("w")
    sns.despine()

plt.tight_layout(pad=1)
plt.subplots_adjust(top=0.9)
plt.figlegend(handles, labels, bbox_to_anchor=(0, 0.9 + 0.02, 1, 0.2),
              loc="lower center", mode=None, borderaxespad=1, ncol=3, frameon=False)
plt.savefig(FIGURES_DIR / f"nstar_{error_toplot}_synthetic.pdf")

#%% 1b. Plotting everything

# main text: kenrrels are goals
pretty_kernels.update({"borda_kernel_nu_auto": "$g_1$", "mallows_kernel_nu_auto": "$g_3$", "jaccard_kernel_k_1": "$g_2$"})

df_ = df_nstar.copy()
df_["nstar_error"] = df_["nstar"] - df_["nstar_true"]
df_["nstar_relative_error"] = (df_["nstar"] - df_["nstar_true"]) / df_["nstar_true"]
df_["nstar_absolute_error"] = np.abs(df_["nstar"] - df_["nstar_true"])
df_["nstar_absrel_error"] = np.abs(df_["nstar"] - df_["nstar_true"]) / df_["nstar_true"]
df_["nstar_rel_error"] = (df_["nstar"] - df_["nstar_true"]) / df_["nstar_true"]
df_["nstar_ratio_error"] = df_["nstar"] / df_["nstar_true"]
# df_ = df_.loc[df_["N"] != df_["Nmax"]]

dfplot = df_.copy().query("N < 1000").query("distr != 'Spike(na=4, ties=True)'").rename(columns=pretty_columns)
dfplot["kernel"] = dfplot["kernel"].map(pretty_kernels)
dfplot["distr"] = dfplot["distr"].map(pretty_distrs_sort)
dfplot = dfplot.sort_values("distr")
dfplot["distr"] = dfplot["distr"].map({v: k for k, v in pretty_distrs_sort.items()}).map(pretty_distrs)

error_toplot = "nstar_rel_error"
y = pretty_columns[error_toplot]

hue_order = dfplot["kernel"].sort_values().unique()

fig, ax = plt.subplots(figsize=(2.5, 5.5 / 2.5))

sns.boxplot(dfplot, x=pretty_columns["N"], y=y, showfliers=False, fliersize=0.3, hue="kernel",
            palette="cubehelix", ax=ax, legend=True, linewidth=1.2, fill=False, gap=0.5, hue_order=hue_order)

ax.set_ylim(-ylim, ylim)
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.axhline(0, color="black", zorder=-1, alpha=0.3)
ax.grid(axis="y", zorder=-1, alpha=0.3)

ax.legend(*ax.get_legend_handles_labels()).get_frame().set_edgecolor("w")
sns.despine()
plt.tight_layout(pad=0.5)
plt.savefig(FIGURES_DIR / f"nstar_{error_toplot}_complete.pdf")


#%% 2. Get percentage of closeness

thr = 0.2

df__ = df_.copy().query("N < 1000").query("distr != 'Spike(na=4, ties=True)'")
goodN = df__.query("-@thr < nstar_rel_error < @thr").groupby("N")["nstar_rel_error"].count()
allN = df__.groupby("N")["nstar_rel_error"].count()

df_good = (goodN/allN).rename(f"fraction_good_{thr}").reset_index()

