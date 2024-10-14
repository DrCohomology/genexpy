"""
To test:
    - samples frpm the same class of distributions have similar persistence landscape

If we do this properly, we can then visualize distributions with thei persistence landscapes.

"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
import torch

from collections import Counter, defaultdict
from functools import reduce
from importlib import reload
from itertools import product
from matplotlib.patches import Arc, Polygon
from pathlib import Path
from persim import plot_diagrams
from ripser import Rips
from scipy.stats import binomtest, friedmanchisquare
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from genexpy import lower_bounds as gu
from genexpy import kernels as ku
from genexpy import rankings_utils as ru
from genexpy import mmd as mmd
from genexpy import probability_distributions as du

FIGURES_DIR = Path("test") / "figures_persistent_homology"
FIGURES_DIR.mkdir(parents=False, exist_ok=True)

seed = 1444
rng = np.random.default_rng(seed)
kernel = ku.jaccard_kernel
na = 5
nv = 500
reps = 3  # number of repetitions per distribution
maxdim = 2  # dimensions to compute homology

rips = Rips(verbose=False, maxdim=maxdim)

distrs = {
    "uniform": du.UniformDistribution(seed=seed+1, na=na, ties=True),
    "spike": du.SpikeDistribution(seed=seed+2, na=na, ties=True, kernel=ku.mallows_kernel),
    "degenerate5": du.MDegenerateDistribution(m=5, seed=seed+3, na=na, ties=True),
    "degenerate50": du.MDegenerateDistribution(m=50, seed=seed+3, na=na, ties=True)
}

samples = {}
for name, distr in distrs.items():
    samples.update({f"{name}_{rep}": distr.sample(nv) for rep in range(reps)})

# get (pseudo) distance matrix
Ds = {}  # distances
for name, sample in tqdm(list(samples.items()), desc="Compute distances"):
    G = ku.square_gram_matrix(sample=sample, use_rv=True, kernel=kernel)
    D = np.sqrt(2 * (1-G))  # d12^2 = k11 + k22 - 2*k12
    Ds[name] = D

# get homologies
Hs = {}
for name, D in tqdm(list(Ds.items()), desc="Get homology"):
    Hs[name] = rips.fit_transform(D, distance_matrix=True)

fig, axes = plt.subplots(len(distrs), reps, figsize=(10, 10), sharex=True, sharey=True)

fig.suptitle(f"kernel: {kernel.__name__}, na: {na}")

for ax, (name, H) in zip(axes.flatten(), Hs.items()):
    plot_diagrams(H, ax=ax, legend=False, size=20)

plt.tight_layout()
fig.savefig(FIGURES_DIR / f"persistent_homology_{kernel.__name__}_na={na}.pdf")
plt.show()




# # --- 4. Compare persistence diagrams
# plt.close("all")
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), sharex="all", sharey="all")
#
# # fig.suptitle(f"Two random benchmarks.\n{na} methods, {nv} datasets.")
# fig.suptitle(f"Two similar benchmarks. \n{na} methods, {nv} datasets.")
#
# plot_diagrams(dgm1, ax=ax1, show=False)
# plot_diagrams(dgm2, ax=ax2, show=False)
#
# plt.show()

# for d in range(maxdim+1):
#     print(f"H{d} features: {len(dgm1[d])} vs {len(dgm2[d])}")


