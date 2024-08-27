"""
Test c.i. for n^* and other quantities involved. The c.i. is on the resampling to get distribution of MMD.

With probability 0.9, if I take m samples of MMD, each computed on samples of rankings of size n, MMD is within
    a certain interval.
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

na = 20  # number of alternatives
n = 20  # size of sample of rankings for MMD
m = 10  # size of sample of MMD (number of times we compute it)
rep = 50  #  90% of these m resamples should belong to the confidence interval

uniform = du.UniformDistribution(na=na, seed=0)
sample = uniform.sample(1000)

# compute MMD m times
sample_mmds = []
for ir in tqdm(list(range(rep))):
    tmp = mmd.subsample_mmd_distribution(sample, n, seed=1444+ir, use_rv=True, kernel=ku.mallows_kernel, rep=m,
                                         disjoint=True, replace=False)
    sample_mmds.append(tmp)

#%% plot the generalizability curves

epsstar = 0.2
alphastar = 0.95

logepss = np.linspace(np.log(epsstar)-0.1, np.log(max(np.quantile(mmde, alphastar) for mmde in sample_mmds)) + 0.1, 1000)
gens = np.array([[mmd.generalizability(sample_mmd, eps=np.exp(leps)) for leps in logepss] for sample_mmd in sample_mmds])

df_plot = pd.DataFrame(gens).T
df_plot["logeps"] = logepss
df_plot = df_plot.melt(id_vars="logeps", var_name="MMD_sample", value_name="gen")

fig, ax = plt.subplots()

sns.lineplot(df_plot, x="logeps", y="gen", hue="MMD_sample", ax=ax, palette=sns.color_palette("cubehelix"), errorbar=None)
ax.set_title(f"na: {na}, n: {n}, m: {m}")

sns.despine()
plt.tight_layout()
plt.show()

#%%
"""
What I care about for the confidence interval is the distribution of MMD_n, meaning:
    I can do somthing like prob(dist(MMD_1, MMD_2) < eps) > alpha
        where 
            dist is a distance between distributions (without kernel this time)
            prob is the probability computed with the different resamples

This way, I get the distribution of distances between MMDs and I can tell with a certain confidence that MMD lies
    within a certain ball around the true MMD. This ball can then be visualized with an interval around the true MMD, 
    i.e., with a band sorrounding it 
    The true MMD is MMD computed on an infinite sample -> limit distribution?

Idea: use the test statistic Dn,n = sup(abs(F1-F2)), as for the Kolmogorov-Smirnov test
    This test has rejection region at level alpha:
        Dn,n > sqrt(-log(alpha/2) 1/n)
"""

ks_alpha = 0.95

ks_stat = np.zeros((rep, rep))
for i1, g1 in enumerate(gens):
    for i2, g2 in enumerate(gens):
        ks_stat[i1, i2] = np.max(np.abs(gens[i1] - gens[i2]))

rej_thr = np.sqrt(-np.log(ks_alpha/2) / m)

# --- plot
fig, ax = plt.subplots()

sns.ecdfplot(ks_stat.flatten())
ax.vlines(rej_thr, ymin=0, ymax=1, color="grey")

ax.set_title(f"Distribution of KS test statistic.\n na: {na}, n: {n}, m: {m}")

sns.despine()
plt.tight_layout()
plt.show()



#%%




mmds = {
    n: mmd.subsample_mmd_distribution(
        rankings, subsample_size=n, rep=100, use_rv=True, use_key=False,
        seed=SEED, disjoint=DISJOINT, replace=REPLACE, kernel=kernel, **kernelargs
    )
    for n in range(2, nv // 2 + 1)
}

n = 50
sample_mmds = n : {name: mmd.subsample_mmd_distribution(sample, n, seed=1444, rep=500, kernel=ku.mallows_kernel)
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


