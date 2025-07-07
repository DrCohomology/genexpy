"""
Run the generalizability analysis as specified in config.yaml.
"""

import numpy as np
import os
import pandas as pd
import yaml

from functools import reduce
from itertools import product
from pathlib import Path
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

from genexpy import kernels_classes as kcu
from genexpy import rankings_utils as ru
from genexpy import mmd


try:
    os.chdir("demos/Template")
except FileNotFoundError:
    pass

def load_class(path):
    _, cls = path.rsplit(".", maxsplit=1)
    return getattr(kcu, cls)


def create_experiment_directory(kernel_name, factors, delta):
    exp0_dir = OUTPUT_DIR / "_".join([f"{key}={value}" for key, value in factors.items() if value is not None])
    exp1_dir = exp0_dir / f"{kernel_name}"
    exp21_dir = exp1_dir / f"nstar_N_ALPHA={ALPHA}_delta={delta}"
    exp21_dir.mkdir(parents=True, exist_ok=True)
    exp22_dir = exp1_dir / "computed_generalizability"
    exp22_dir.mkdir(parents=True, exist_ok=True)
    exp23_dir = exp1_dir / "computed_quantiles"
    exp23_dir.mkdir(parents=True, exist_ok=True)
    return exp21_dir, exp22_dir, exp23_dir


# ---- 1: load config


with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = Path(config['paths']['output_dir'])
FIGURES_DIR = Path(config['paths']['figures_dir'])

SEED = config['parameters']['seed']
RNG = np.random.default_rng(SEED)
ALPHA = config['parameters']['alpha']
DELTA = config['parameters']['delta']
REP = config["parameters"]["rep"]
# LR_CONFIDENCE = config['parameters']['lr_confidence']
# CI_LOWER = (1 - LR_CONFIDENCE) / 2
# CI_UPPER = LR_CONFIDENCE + CI_LOWER

DATASET = Path(config['data']['dataset_path'])
EXPERIMENTAL_FACTORS = config['data']['experimental_factors']
TARGET = config['data']['target']
ALTERNATIVES = config['data']['alternatives']

SAMPLE_SIZE = config['sampling']['sample_size']
DISJOINT = config['sampling']['disjoint']
REPLACE = config['sampling']['replace']

# ---- 2: load dataset, check conditions

df = pd.read_parquet(DATASET)

assert sum(value is None for value in EXPERIMENTAL_FACTORS.values()) == 1, \
    "Exactly one factor must be set to null in config.yaml."

df = df[list(EXPERIMENTAL_FACTORS.keys()) + [ALTERNATIVES, TARGET]]

df = df.query(" and ".join(f"{factor} == '{lvl}'" if isinstance(lvl, str) else f"{factor} == {lvl}"
                           for factor, lvl in EXPERIMENTAL_FACTORS.items()
                           if lvl not in [None, "_all"])).reset_index(drop=True)

try:
    design_factor_lvls = df.groupby([factor for factor, lvl in EXPERIMENTAL_FACTORS.items() if lvl == "_all"]).groups
except ValueError:
    design_factor_lvls = {"None": df.index}

# ---- 1.2: load kernels

KERNELS = {}
for kernel_dict in config["kernels"]:
    kernel_name_base = kernel_dict["kernel"]
    kernel_params = kernel_dict["params"]
    kernel_cls = load_class(kernel_dict["class"])

    param_combinations = product(*[[(param, val) for val in vals] for param, vals in kernel_params.items()])
    for pc in param_combinations:

        pc = dict(pc)

        # find the index of the selcted alternative
        if kernel_name_base == "borda_kernel":
            pc["idx"] = df[ALTERNATIVES].unique().tolist().index(pc["alternative"])
            del pc["alternative"]

        kernel_name_inst = kernel_name_base + reduce(lambda x, y: x+y, [f"__{p[0]}={p[1]}" for p in pc.items()])
        kernel_obj = kernel_cls(**pc)
        epsstar = kernel_obj.get_eps(DELTA)
        KERNELS[kernel_name_inst] = (kernel_name_base, kernel_obj, epsstar)


# ---- 3. Main loop


np.seterr(divide='ignore')

for fixed_levels, idxs in tqdm(list(design_factor_lvls.items()), position=0, desc="Configurations", leave=True):

    # Query the results for the fixed-levels
    idf = df.loc[idxs].reset_index(drop=True)
    if idf.empty:
        continue

    # Current levels of design and held-constant factor
    factors_dict = {factor: lvl
                    for factor, lvl in EXPERIMENTAL_FACTORS.items()
                    if lvl not in [None, "_all"]}
    factors_dict.update({factor: idf[factor].unique()[0] for factor, lvl in EXPERIMENTAL_FACTORS.items()
                         if lvl == "_all"})

    # Rank the alternatives
    rank_matrix = ru.get_rankings_from_df(idf, factors=list(EXPERIMENTAL_FACTORS.keys()),
                                          alternatives=ALTERNATIVES,
                                          target=TARGET,
                                          lower_is_better=False, impute_missing=True)
    # Impute the missing values
    rank_matrix = rank_matrix.fillna(rank_matrix.max())

    # Global sample of rankings available
    rankings_all = ru.SampleAM.from_rank_vector_matrix(rank_matrix.to_numpy())

    # Loop over the kernels
    for kernel_name_inst, (kernel_name_base, kernel_obj, epsstar) in KERNELS.items():

        # Create experiment directories
        nstar_dir, gen_dir, quant_dir = create_experiment_directory(kernel_name_inst, factors_dict, delta)

        out = []
        for M in range(1, len(rankings_all) // SAMPLE_SIZE):
            if M * SAMPLE_SIZE > len(rankings_all):
                break

            # Sample N random rankings
            N = M * SAMPLE_SIZE
            sample = rankings_all.get_subsample(N, seed=M)

            # We do not need to compute dfy and dfq again if we have already computed them for another (alphastar, deltastar)
            if (f"dfy_{N}" in [x.stem for x in gen_dir.glob("*.parquet")] and
                f"dfmmd_{N}" in [x.stem for x in quant_dir.glob("*.parquet")]):
                try:
                    dfy = pd.read_parquet(gen_dir / f"dfy_{N}.parquet")
                    dfmmd = pd.read_parquet(quant_dir / f"dfmmd_{N}.parquet")

                    dfq = pd.DataFrame(dfmmd.groupby("n")["eps"].quantile(ALPHA)).reset_index()
                    dfq["log(eps)"] = np.log(dfq["eps"])
                    dfq["log(n)"] = np.log(dfq["n"])

                    logepss = dfy["log(eps)"].unique()
                except Exception as e:
                    print("Exception thrown for experimental condition: ", factors_dict)
                    raise e
            else:

                mmds = {n: mmd.mmd_distribution_vectorized_class(sample=sample, n=n, rep=REP, kernel_obj=kernel_obj)
                        for n in [2, N // 2, 2]}

                dfmmd = pd.DataFrame(mmds).melt(var_name="n", value_name="eps")

                logepss = np.log(
                    np.linspace(0.001, max(np.quantile(mmde, ALPHA) for mmde in mmds.values()) + 0.1, 1000))

                ys = {n: [mmd.generalizability(mmde, np.exp(logeps)) for logeps in logepss] for n, mmde in mmds.items()}
                dfy = pd.DataFrame(ys, index=logepss).reset_index().melt(id_vars='index', var_name='n',
                                                                         value_name='generalizability')
                dfy.rename(columns={'index': 'log(eps)'}, inplace=True)
                dfy["n"] = dfy["n"].astype(int)
                dfy["N"] = N
                dfy["disjoint"] = DISJOINT
                dfy["replace"] = REPLACE

                qs = {n: np.log(np.quantile(mmde, ALPHA)) for n, mmde in mmds.items()}
                dfq = pd.DataFrame(list(qs.items()), columns=['n', 'log(eps)'])
                dfq['log(n)'] = np.log(dfq['n'])

            # if the log of any quantile is -inf, the predicted nstar is 1
            if np.isinf(dfq["log(eps)"]).any():
                nstar = 1
                singular = True
            else:
                # Fit linear regression as in equation (4) of Matteucci (2024)
                lr = LinearRegression()
                lr.fit(dfq[['log(eps)']].values, dfq[['log(n)']].values)
                ns_pred = lr.predict(logepss.reshape(-1, 1)).reshape(-1)
                nstar = ns_pred[np.argmax(logepss > np.log(epsstar))]
                singular = False

            # Store results
            result_dict = {
                "kernel": kernel_name_base,
                "alpha": ALPHA,
                "eps": epsstar,
                "delta": DELTA,
                "disjoint": DISJOINT,
                "replace": REPLACE,
                "N": N,
                "nstar": nstar,
                "singular": singular
            }
            result_dict.update(factors_dict)
            out.append(result_dict)

            dfy.to_parquet(gen_dir / f"dfy_{N}.parquet")
            dfmmd.to_parquet(quant_dir / f"dfmmd_{N}.parquet")

        # Store nstar predictions
        out = pd.DataFrame(out)
        out.to_parquet(nstar_dir / "nstar.parquet")
import random


