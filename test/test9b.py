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
import os
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns

from collections import Counter, defaultdict
from importlib import reload
from pathlib import Path
from scipy.stats import binomtest, friedmanchisquare, spearmanr, wilcoxon
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from genexpy import lower_bounds as gu
from genexpy import kernels as ku
from genexpy import rankings_utils as ru
from genexpy import mmd as mmd
from genexpy import probability_distributions as du

"""
bernoulli:      H0: (A better than B) ~ B(2/3)  # H0 is not compatible with the generalizability question
bernoulli2:     H0: (B better than A) ~ B(2/3)  # same as above
conover:        H0: A and B have different ranks
nemenyi:        H0: A and B have different ranks
"""

logepss = np.log(np.linspace(1e-10, np.sqrt(2), 1000))
repnum = 100

kernel = ku.jaccard_kernel

rng = np.random.default_rng(1463290)

# distr = du.UniformDistribution(na=2, seed=19, ties=True)
# distr = du.SpikeDistribution(na=5, seed=19, ties=False)
# distr = du.DegenerateDistribution(na=na, seed=10)

universe = ru.SampleAM.from_rank_vector_matrix(np.array([[0, 1], [1, 0]]).T)
pmf = np.array([7, 3])
distr = du.PMFDistribution(universe=universe, pmf=pmf, ties=False)

OUTPUT_DIR = Path(os.getcwd()) / "test" / "outputs" / f"test9b_{kernel.__name__}" / f"test9b_pmf{pmf[0]}-{pmf[1]}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

out = []
outg = []
# preliminary experiments
for N in tqdm([10, 20, 40, 80]):
    for rep in range(repnum):
        # get a sample of preliminary results
        sample = distr.sample(N)
        rv = sample.to_rank_vector_matrix()

        mmd_distr = mmd.subsample_mmd_distribution(sample, N // 2, seed=1444*repnum+N, use_rv=True, kernel=kernel,
                                                   rep=100, disjoint=True, replace=False)

        gens = mmd.generalizability(mmd_distr, np.exp(logepss))

        tmp = {
            "N": N,
            "rep": rep,
            "kernel": kernel.__name__,
            # "friedman": scipy.stats.friedmanchisquare(rv.T),
            "conover": sp.posthoc_conover_friedman(rv.T).iloc[0, 1],
            "nemenyi": sp.posthoc_nemenyi_friedman(rv.T).iloc[0, 1],
            "wilcoxon": wilcoxon(*rv)
        }
        tmpg = {
            (N // 2, rep): gens
        }

        out.append(tmp)
        outg.append(tmpg)

df_out = pd.DataFrame(out)
df_outg = pd.concat([pd.DataFrame.from_dict(tg).T for tg in outg], axis=0)
df_outg.index.rename(("N", "rep"), inplace=True)
df_outg.columns = np.exp(logepss)
df_outg.columns.rename("eps", inplace=True)
df_outg = df_outg.melt(ignore_index=False, value_name="gen").reset_index()

df_out.to_parquet(OUTPUT_DIR / "test9b_tests.parquet")
df_outg.to_parquet(OUTPUT_DIR / "test9b_gen.parquet")


#%% Plotting

a = pmf[0]
b = pmf[1]
kernelname = kernel.__name__

a = 7
b = 3
kernelname = "jaccard_kernel"
testname = "wilcoxon"

gth = 0.9  # generalizability threshold
pth = 0.1  # p-values threshold

OUTPUT_DIR = Path(os.getcwd()) / "test" / "outputs" / f"test9b_{kernelname}" / f"test9b_pmf{a}-{b}"

# load data
df_out = pd.read_parquet(OUTPUT_DIR / "test9b_tests.parquet")
df_outg = pd.read_parquet(OUTPUT_DIR / "test9b_gen.parquet")

# specify epsilon
delta = 0.05
match kernel.__name__:
    case "mallows_kernel":
        eps = np.sqrt(2 * (1 - np.exp(-delta)))  # assumes nu = 1/binom(n, 2)
    case "jaccard_kernel":
        eps = np.sqrt(2 * (1 - (1 - delta)))
    case "borda_kernel":
        eps = np.sqrt(2 * (1 - np.exp(-delta)))  # assumes nu = 1/n
    case _:
        eps = None

# find closest value of epsilon in the dataframe
eps = df_outg["eps"].unique()[np.argmin(np.abs(df_outg["eps"].unique() - eps))]

# join the dataframes
df_out["Ngen"] = df_out["N"] // 2
df_ = pd.merge(df_out, df_outg, left_on=["Ngen", "rep"], right_on=["N", "rep"])
df_.rename(columns={"N_x": "N"}, inplace=True)

# plot
fig, axes = plt.subplots(1, 5, sharey=False, figsize=(14, 4))
# fig.suptitle(f"{pmf[0]}:{pmf[1]} distribution of {len(pmf)} alternatives.")
fig.suptitle(f"{a}:{b} distribution of {len(pmf)} alternatives.")

ax = axes[0]
ax.set_title(f"p-value of {testname}")
sns.boxplot(data=df_out, x="N", y=testname, ax=ax)
ax.axhline(pth, ls="--", c="grey")

ax = axes[1]
ax.set_title(f"Generalizability, eps = {eps:.2f}")
sns.boxplot(data=df_outg.query(f"eps=={eps}"), x="N", y="gen", ax=ax)
ax.axhline(gth, ls="--", c="grey")
ax.set_xlabel("N // 2")

ax = axes[2]
df2_ = df_.query(f"eps=={eps}")
# mi = mutual_info_regression((df2_["gen"].to_numpy()).reshape(-1, 1), df2_[testname].to_numpy())[0]
mi = mutual_info_classif((df2_["gen"].to_numpy()).reshape(-1, 1), df2_[testname].to_numpy(dtype="str"))[0]
rho = spearmanr(df2_["gen"], df2_[testname])[0]
# mi = normalized_mutual_info_score(df2_["gen"], df2_[testname])  # for discrete variables
ax.set_title(f"p-value of {testname} VS gen, eps = {eps:.2f} \n MI: {mi:.2f}; rho: {rho:.2f}")
sns.scatterplot(data=df2_, x="gen", y=testname, hue="N", ax=ax)

ax.axhline(pth, ls="--", c="grey")
ax.axvline(gth, ls="--", c="grey")

ax = axes[3]
ax.set_title(f"Fraction of gen. results\np: {pth}; alpha: {gth}")
dfs = df2_.query(f"{testname} <= @pth").groupby("N")["rep"].count().rename("n_sign")
dfg = df2_.query("gen >= @gth").groupby("N")["rep"].count().rename("n_gen")
dfgs = df2_.query(f"{testname} <= @pth and gen >= @gth").groupby("N")["rep"].count().rename("n_gs")
df3_ = pd.concat([dfs, dfg, dfgs], axis=1)
df3_["p(gen|sign)"] = df3_["n_gs"] / df3_["n_sign"]
sns.lineplot(df3_, x="N", y="p(gen|sign)", ax=ax)

ax = axes[4]
ax.set_title(f"Fraction of sign. results\np: {pth}; alpha: {gth}")
df3_["p(sign|gen)"] = df3_["n_gs"] / df3_["n_gen"]
sns.lineplot(df3_, x="N", y="p(sign|gen)", ax=ax)

plt.tight_layout()

plt.savefig(OUTPUT_DIR.parent / f"pmf{a}-{b}_CI-gen_eps={eps:.2f}.pdf")

plt.show()