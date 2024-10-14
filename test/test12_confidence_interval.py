"""
1. What confidence interval can we place on our estimate of generalizability?
    1.1. Estimate generalizability of a single sample
    1.2. Estimate gneralizability of a distribution (can I get the theoretical value for simple distributions?)

2. what can we say on the generalizability for 2n if we know that for n?
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns

from collections import Counter, defaultdict
from importlib import reload
from pathlib import Path
from scipy.special import binom
from scipy.stats import binomtest, friedmanchisquare, spearmanr, wilcoxon
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
from typing import Union

from genexpy import lower_bounds as gu
from genexpy import kernels as ku
from genexpy import rankings_utils as ru
from genexpy import mmd as mmd
from genexpy import probability_distributions as du



#%% 1.1. Estimate generalizability of a single sample

kernel = ku.mallows_kernel
# universe = ru.SampleAM.from_rank_vector_matrix(np.array([[0, 1], [1, 0]]).T)
# pmf = np.array([5, 5])
# distr = du.PMFDistribution(universe=universe, pmf=pmf, ties=False)

distr = du.UniformDistribution(na=10, ties=True)

OUTPUT_DIR = Path(os.getcwd()) / "test" / "outputs" / f"test12_{kernel.__name__}_{distr}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

repnum = 100
N = 1000
sample = distr.sample(N)

mmds = []
for rep in tqdm(list(range(repnum))):
    mmd_ = mmd.subsample_mmd_distribution(sample, subsample_size=5, rep=100, seed=rep, kernel=kernel)
    mmds.append(mmd_)

#%% 1.1.a Plot

eps = np.linspace(0, np.sqrt(2), 100)
gens = np.array([mmd.generalizability(mmd_, eps) for mmd_ in mmds]).T

dfg = pd.DataFrame(gens)
dfg["eps"] = eps
dfg = dfg.melt(id_vars="eps", var_name="rep", value_name="gen")





fig, axes = plt.subplots(1, 2)

ax = axes[0]
# for gen_ in gens.T:
#     sns.lineplot(x=eps, y=gen_, ax=ax, color="black")
# sns.boxplot(dfg, x="eps", y="gen", ax=ax, native_scale=True)

sns.lineplot(dfg, x="eps", y="gen", errorbar="sd", ax=ax)

ax = axes[1]
# sns.lineplot(gens.mean(axis=1), ax=ax, c="red")
sns.lineplot(x=eps, y=gens.var(axis=1), ax=ax, c="blue")

plt.tight_layout()

plt.show()



#%% 1.2 Theoretical derivation

def get_number_samples(n: int, k: int) -> int:
    """
    Number of samples of size k from a pool of choices of size n (with replacement).
    """
    return int(binom(n + k -1, k))




# number of alternatives, corresponding Fubini number, and number of samples
na = 2
nfa = du.get_unique_ranks_distribution(na, normalized=False).sum().astype(int)
nv = 2
ns = get_number_samples(nfa, nv)


unique_symbols = du.get_unique_ranks_distribution(na, normalized=False)


kernel = ku.jaccard_kernel



