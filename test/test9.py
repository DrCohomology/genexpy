"""
Generalizability and statistical significance.
    1. can we have generalizability without significance? According to VA, yes
    2. we get some results, there is no encoder significantly better than the rest.
        Results are generalizable -> no reason to go any further, your results will not gte more significant
        Results are not gen. -> go further, it might become easier to achieve significance
    3.                      generalizability
                            yes             no
        significance    yes ?               ?
                        no  ?               ?

    4. What if we move to statistical tests for continuous distributions, rather than just Bernoulli "is better than"?
    5. test if the distributions of results are the same or not with some MMD test
        what is the relation between this kind of test and generalizability?
        can we rephrase generalizability as some test?

Generate a certain number of rankings of 2 alternatives, uniformly.
Run a 95% test for "A is better than B" and compute 0.95 generalizability
Plot behavior on n

-----------------------------------------------------------
Take-aways
1. for a uniform distribution, the pvalues of nemenyi and conover tests wildly oscillates and do
    not converge to 1.
    This is explained by: a test with pvalue alpha ALWAYS has a probability alpha of falsely rejecting the
        null hypothesis (null which is indeed true for a uniform). Just think about the definition of rejection region.
    For increasing n, this effect is explained by the fact that the deviations within the sample are getting smaller
        while the test is becoming more and more sensitive to these deviations. The two effects seem to balance out
        quite well, at least in this case.
    Thus: generalizability does increase with n, but the p value does not.
        Eventually, we will get generalizable results, but the test is significant at 0.95 with probability 0.95.
        And this probability is independent from n!
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns

from collections import Counter, defaultdict
from importlib import reload
from pathlib import Path
from scipy.stats import binomtest, friedmanchisquare
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from genexpy import lower_bounds as gu
from genexpy import kernels as ku
from genexpy import rankings_utils as ru
from genexpy import mmd as mmd
from genexpy import probability_distributions as du

reload(du)

na = 2          # number of alternatives
maxN = 500      # size of sample of rankings for MMD and test
step = 25
m = 150         # size of sample of MMD (number of times we compute it)
# rep = 50        # 90% of these m resamples should belong to the confidence interval

"""
bernoulli:      H0: (A better than B) ~ B(2/3)  # H0 is not compatible with the generalizability question
bernoulli2:     H0: (B better than A) ~ B(2/3)  # same as above
conover:        H0: A and B have different ranks
nemenyi:        H0: A and B have different ranks
"""

alpha = 0.95
# eps = 0.01
# epss = [0.001, 0.01, 0.1, 1]
# epss = [0, np.sqrt(2*(1-np.exp(-1)))]
epss = [0]  # for the comparison plot with the test
logepss = np.log(np.linspace(0.0001, np.sqrt(2*(1-np.exp(-1))) / 2, 1000))  # for gen plot

rng = np.random.default_rng(1463290)

distr = du.SpikeDistribution(na=5, seed=19, ties=False)
# distr = du.DegenerateDistribution(na=na, seed=10)

# universe = ru.SampleAM.from_rank_function_matrix(np.array([[0, 1], [1, 0]]).T)
# pmf = np.array([7, 3])
# distr = du.PMFDistribution(universe=universe, pmf=pmf, ties=False)

out = []
outg = []
# for N in tqdm(list(range(10, maxN+1, step))):
for N in tqdm([10, 20, 40, 80, 160, 320]):
    n = N // 2
    sample = distr.sample(N)
    # large_sample = distr.sample(1000)
    sample_n = ru.SampleAM(sample[:n])
    rv = sample_n.to_rank_function_matrix()

    # nab, nba = N - sample_n.to_rank_function_matrix().sum(axis=1)  # A better than B, B better than A
    # df_out["bernoulli1"].append(1 - binomtest(nab, N, p=2 / 3, alternative="two-sided").pvalue)
    # df_out["bernoulli2"].append(1 - binomtest(nba, N, p=2 / 3, alternative="two-sided").pvalue)

    mmd_distr = mmd.subsample_mmd_distribution(sample, n, seed=1444+N, use_rv=True, kernel=ku.mallows_kernel, rep=m,
                                               disjoint=True, replace=False)
    # mmd_distr_ls = mmd.subsample_mmd_distribution(large_sample, n, seed=1333+N, use_rv=True, kernel=ku.mallows_kernel,
    #                                               rep=m, disjoint=False, replace=True)

    tmp = {
        "n":                n,
        "conover":          sp.posthoc_conover_friedman(rv.T).iloc[0, 1],
        "nemenyi":          sp.posthoc_nemenyi_friedman(rv.T).iloc[0, 1],
        f"q_{alpha}":       np.quantile(mmd_distr, alpha),
        f"q_ub_{alpha}":    np.sqrt((4 / np.sqrt(n) * np.sqrt(np.log(1 / alpha))))
    }
    tmp.update({
        f"gen_{eps}": mmd.generalizability(mmd_distr, eps=eps) for eps in epss
    })
    out.append(tmp)

    tmpg = {"n": n}
    tmpg.update({np.exp(eps): mmd.generalizability(mmd_distr, eps=np.exp(eps)) for eps in logepss})
    outg.append(tmpg)

df_out = pd.DataFrame(out)
df_outg = pd.DataFrame(outg)

#%%

toplot = ["n", "conover"] + [ f"gen_{eps}" for eps in epss]
dfplot = df_out[toplot].melt(id_vars="n", var_name="stat", value_name="val")
dfplotg = df_outg.melt(id_vars="n", var_name="eps", value_name="gen")#.query("eps < 0.3")

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
axes = axes.flatten()

ax = axes[0]
ax.set_title(distr)
sns.lineplot(data=dfplot, x="n", y="val", hue="stat", ax=ax)
ax.set_xscale("log")

"""
print(np.mean(df_out["nemenyi"] < 0.05))
this value is one minus the power of the test: probability of falsely rejecting the null when it is true
    this happens in this case (with a uniform distribution) 
"""


ax = axes[2]
ax.set_title("Generalizability")
sns.lineplot(dfplotg, x="n", y="gen", hue="eps", ax=ax, legend=True)
# ax.set_xlim([0, 0.1])
ax.set_xscale("log")

ax = axes[3]
ax.set_title("Generalizability")
sns.lineplot(dfplotg, x="eps", y="gen", hue="n", ax=ax, legend=True)
# ax.set_xlim([0, 0.1])
# ax.set_xscale("log")

sns.despine()
plt.tight_layout()
fig.show()

#%% 3d generalizability plot

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

dfplotly = df_outg.melt(id_vars="n", var_name="eps", value_name="gen").pivot(index="eps", columns="n", values="gen").iloc[::25]

# fig = go.Figure(data=go.Surface(z=z, x=x, y=y))
# fig = px.scatter_3d(dfplotg, x="eps", y="n", z="gen", color=z, log_y=True)
# fig.show()

fig = go.Figure(data=[go.Surface(z=dfplotly.values, x=dfplotly.index, y=dfplotly.columns,
                                 contours={
                                     "x": {"show": False},
                                     "z": {"show": False},
                                 },
                                 opacity=0.6)],
                layout=go.Layout(template="plotly_white"))
fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
fig.show()







