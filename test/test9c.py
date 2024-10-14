"""
Similar to test9b, but to investigate whether generalizability helps in practice or not.

Set significance and generalizability thresholds
    pth: 0.1
    gth: 0.9

Run 100 studies (samples of N=10 rankings)
    Record their generalizability (estimated at N // 2)
        technically, we should predict n^* and compare with N
        if we instead ignore this step, we will be underestimating generalizability
    Record their significance

    Repeat until N > Nmax:
        For all not-generalizable studies:
            Run N=10 more experiments
            Compute their generalizability and significance



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
from typing import Union

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

class Study(object):
    __slots__ = ("id", "results", "N", "rv", "test_pvalues", "mmd_distr", "epss", "gens")

    def __init__(self, id: int, results: ru.SampleAM):
        self.id = id
        self.results = results
        self.N = len(self.results)
        self.rv = self.results.to_rank_vector_matrix()

        self.test_pvalues: dict[int: dict] = {}  # N: test: pvalue
        self.mmd_distr: dict[int: np.ndarray[float]] = {}  # N: mmd_distr
        self.epss : np.ndarray[float] = None
        self.gens: dict[int: np.ndarray[float]] = {}  # N: generalizability

    def append_results(self, new_results):
        self.results = self.results.append(new_results)
        self.N = len(self.results)
        self.rv = self.results.to_rank_vector_matrix()

    def run_tests(self):
        self.test_pvalues[self.N] = {
            "conover": sp.posthoc_conover_friedman(self.rv.T).iloc[0, 1],
            "nemenyi": sp.posthoc_nemenyi_friedman(self.rv.T).iloc[0, 1],
            "wilcoxon": wilcoxon(*self.rv)[1],
        }

    def run_mmd(self, kernel, n: Union[str, int], epss=np.linspace(1e-10, np.sqrt(2), 1000)):
        n = self.N // 2 if n == "auto" else n
        self.mmd_distr[self.N] = mmd.subsample_mmd_distribution(self.results, n, seed=1444, use_rv=True,
                                                                kernel=kernel, rep=1000, disjoint=True,
                                                                replace=False)
        self.epss = epss
        self.gens[self.N] = mmd.generalizability(self.mmd_distr[self.N], epss)

    def is_significant(self, pth):
        return {k: (v <= pth) for k, v in self.test_pvalues[self.N].items()}

    def is_generalizable(self, eps, gth):
        """
        Take the maximum epsilon in self.epss which is below eps, then take the generalizability at the corresponding
            index in self.gens, finally compare this generalizability with the threshold.
        """
        return self.gens[self.N][self.epss[self.epss <= eps].argmax()] >= gth

    def story_significant(self, pth):
        return {N: {k: (v <= pth) for k, v in Npvals.items()} for N, Npvals in self.test_pvalues.items()}

rng = 275843895
kernel = ku.jaccard_kernel
universe = ru.SampleAM.from_rank_vector_matrix(np.array([[0, 1], [1, 0]]).T)
pmf = np.array([5, 5])
distr = du.PMFDistribution(universe=universe, pmf=pmf, ties=False)
pth = 0.1
gth = 0.9
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

repnum = 100
N = 10
itermax = 8
studies = []
for rep in tqdm(list(range(repnum))):
    i = 0
    g = False
    while i < itermax and not g:
        if i == 0:
            study = Study(rep, distr.sample(N))
        else:
            study.append_results(distr.sample(N))
        study.run_tests()
        study.run_mmd(kernel=kernel, n="auto")
        s = study.is_significant(pth)
        g = study.is_generalizable(eps, gth)
        i += 1

    studies.append(study)
#%% save

dfs = pd.concat({study.id: pd.DataFrame(study.story_significant(pth)).T for study in studies})
dfN = pd.DataFrame([{"study_id": study.id, "N": study.N} for study in studies])

testname = "wilcoxon"
fp = pd.DataFrame(dfs[testname]).reset_index(names=["study_id", "N"]).groupby(["N", testname]).count()

OUTPUT_DIR = Path(os.getcwd()) / "test" / "outputs" / f"test9c_{kernel.__name__}" / f"test9c_pmf{pmf[0]}-{pmf[1]}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dfs.to_parquet(OUTPUT_DIR / "significance.parquet")
dfN.to_parquet(OUTPUT_DIR / "dfN.parquet")


#%% load

OUTPUT_DIR = Path(os.getcwd()) / "test" / "outputs" / f"test9c_{kernel.__name__}" / f"test9c_pmf{pmf[0]}-{pmf[1]}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dfs = pd.read_parquet(OUTPUT_DIR / "significance.parquet")
dfN = pd.read_parquet(OUTPUT_DIR / "dfN.parquet")

testname = "wilcoxon"
fp = pd.DataFrame(dfs[testname]).reset_index(names=["study_id", "N"]).groupby(["N", testname]).count()

#%%

import seaborn.objects as so

df_ = fp.reset_index().rename(columns={"study_id": "count", "wilcoxon": "significant"})


fig, ax = plt.subplots()

sns.lineplot(data=df_, x="N", y="count", hue="significant")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "more_gen-more_sign.pdf")
plt.show()

#%% significance|generalizability for different N

OUTPUT_DIR = Path(os.getcwd()) / "test" / "outputs" / f"test9c_{kernel.__name__}" / f"test9c_pmf{pmf[0]}-{pmf[1]}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tetsname = "wilcoxon"

dfs = pd.read_parquet(OUTPUT_DIR / "significance.parquet")
dfN = pd.read_parquet(OUTPUT_DIR / "dfN.parquet")

dfs = dfs.reset_index(names=["study_id", "N"])

df_ = pd.merge(dfs, dfN, on=["study_id", "N"], how="left")
df_ = pd.merge(df_, df_.groupby("study_id")["N"].max(), on="study_id")
df_ = df_.rename(columns={"N_x": "N", "N_y": "Nmax"})


out = []
for N in df_["N"].unique():
    df_n = df_[df_["N"] == N]

    df_ng = df_n[df_n["Nmax"] > N]
    df_ngs = df_ng[df_ng[testname]]
    df_ngns = df_ng[~df_ng[testname]]


    df_g = df_n[df_n["Nmax"] <= N]
    df_gs = df_g[df_g[testname]]
    df_gns = df_g[~df_g[testname]]

    out.append({
        "N": N,
        "gen": len(df_g),
        "sign|gen": len(df_gs),
        "sign/gen": np.divide(len(df_gs), len(df_g)).round(2),
        # "notsign|gen": len(df_gns),
        "notgen": len(df_ng),
        "sign|notgen": len(df_ngs),
        "sign/notgen": np.divide(len(df_ngs), len(df_ng)).round(2),
        # "notsign|notgen": len(df_ngns),
    })

out = pd.DataFrame(out)

# out["gen"] = np.cumsum(out["gen"])
# out["sign|gen"] = np.cumsum(out["sign|gen"])