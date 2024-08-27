"""
Synthetic distributions!
Test the new sampling functions
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
from genexpy import probability_distributions as du

# na = 20, 10000 samples -> 50 seconds
na = 20

distributions = dict(
    uniform = du.UniformDistribution(na=na),
    spike = du.SpikeDistribution(na=na),
    spike_center = du.SpikeDistribution(na=na,
                                        center=ru.AdjacencyMatrix.from_rank_function(np.zeros(na)).tohashable(),
                                        kernel=ku.mallows_kernel, kernelargs=dict(nu="auto")),
    spike_border = du.SpikeDistribution(na=na,
                                        center=ru.AdjacencyMatrix.from_rank_function(np.arange(na)).tohashable(),
                                        kernel=ku.mallows_kernel, kernelargs=dict(nu="auto")),
    mdeg = du.MDegenerateDistribution(na=na, m=2)
)

samples = {name: distr.sample(1000) for name, distr in tqdm(list(distributions.items()))}
times = {name: distr.sample_time for name, distr in distributions.items()}

#%%
n = 50
sample_mmds = {name: mmd.subsample_mmd_distribution(sample, n, seed=1444, rep=500, kernel=ku.mallows_kernel)
        for name, sample in tqdm(list(samples.items()))}

#%%

fig, axes = plt.subplots(len(distributions), 1, sharex="all")
axes = axes.flatten()

for ax, (name, sample_mmd) in zip(axes, sample_mmds.items()):
    sns.histplot(sample_mmd, ax=ax, binwidth=0.005)
    ax.set_title(name)
    # ax.hist(sample_mmd)

plt.tight_layout()
plt.show()

#%%

sample_grams = {name: ku.square_gram_matrix(sample, kernel=ku.mallows_kernel, use_rv=True) for name, sample in tqdm(list(samples.items()))}


