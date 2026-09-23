"""
Is there a difference between reliability obtained from
    1. randomly sampling N new results or
    2. adding Nstep results to the existing N
?

Run from the Categorical Encoders demo folder (where config.yaml is).
n*_N is estimated exactly as in ProjectManager.estimate_nstar(method="embedding");
only the way the sample of size N is obtained changes.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from genexpy import random as du
from genexpy.managers import ProjectManager
from genexpy.utils import rankings as ru

CONFIGURATION = {"model": "LR", "tuning": "no", "scoring": "AUC"}
KERNEL = "MallowsKernel"
ALPHA, DELTA = 0.95, 0.05
REPLICATES = 50
SEED = 37

XLABEL = r"$N$"
YLABEL = r"$\hat n^*_N$"
FONTSIZE = 20


class GivenSampleProjectManager(ProjectManager):
    """ProjectManager whose MMD estimation uses the sample as given, instead of drawing a new one of size N."""

    def _estimate_mmd_from_experiments_rankings(self, sample, configuration, kernel_obj, N, method):
        dfmmd = kernel_obj.mmd_distribution_many_n(sample=sample, nmin=2, nmax=N // 2, step=2,
                                                   rep=self.config_params["rep"],
                                                   disjoint=self.config_sampling["disjoint"],
                                                   replace=self.config_sampling["replace"],
                                                   method=method, N=N, use_cached_support_matrix=True)
        for factor, lvl in configuration.items():
            dfmmd.loc[:, factor] = lvl
        return dfmmd


# --- data, as in ProjectManager.reliability_analysis
pm = GivenSampleProjectManager("config.yaml", demo_dir=os.getcwd(), is_project_manager=False)
pm.dfmmd, pm.dump_results = None, False  # no precomputed MMD, nothing written
pm.config_params["alpha"], pm.config_params["delta"] = [ALPHA], [DELTA]

rankings = ru.get_matrix_from_df(pm.results, factors=list(pm.all_factors),
                                 alternatives=pm.config_data["alternatives_col_name"],
                                 target=pm.config_data["target_col_name"], get_rankings=True,
                                 lower_is_better=pm.config_data["target_is_error"], impute_missing=True,
                                 tol_missing_indices=pm.config_params["tol_missing_alternatives"],
                                 tol_missing_columns=pm.config_params["tol_missing_conditions"], as_numpy=False)
mask = pd.Series(True, index=rankings.columns)
for factor, lvl in CONFIGURATION.items():
    mask &= (rankings.columns.get_level_values(factor) == lvl)
sample_rankings = ru.SampleAM.from_rank_vector_matrix(rankings.loc[:, mask.values].values)

kernel_obj = next(k for k in pm.kernels if k.__class__.__name__ == KERNEL)
kernel_obj.set_support(sample_rankings.get_support_pmf()[0])

Nstep = pm.config_sampling["sample_size"]
Ns = range(Nstep, int(np.nanmin((len(sample_rankings), pm.config_params["Nmax"]))), Nstep)

# --- simulated studies
out = []
for r in tqdm(range(REPLICATES)):
    P = du.PMFDistribution.from_sample(sample_rankings, seed=SEED + r)
    sample_nested = ru.SampleAM(np.empty(0, dtype=object))
    for N in Ns:
        sample_fresh = P.sample(N)                                   # 1. N new results
        sample_nested = sample_nested.append(P.sample(Nstep))        # 2. Nstep results added to the previous ones
        for scheme, sample in [("fresh", sample_fresh), ("nested", sample_nested)]:
            rows = pm.estimate_nstar(sample=sample, configuration=CONFIGURATION, kernel_obj=kernel_obj,
                                     method="embedding", N=N)
            out.extend(dict(row, scheme=scheme, replicate=r) for row in rows)
df = pd.DataFrame(out)

# --- summary
wide = {s: df.query("scheme == @s").pivot(index="replicate", columns="N", values="nstar") for s in ["fresh", "nested"]}
summary = pd.DataFrame({f"{stat}_{s}": getattr(w, stat)() for s, w in wide.items() for stat in ["mean", "std"]})
for s, w in wide.items():
    summary[f"corr_prev_N_{s}"] = [np.nan] + [np.corrcoef(w[a], w[b])[0, 1] for a, b in zip(w.columns[:-1], w.columns[1:])]
print(summary.round(2).to_string())

# --- figure
plt.rcParams.update({"font.size": FONTSIZE})
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, layout="constrained")
for ax, (scheme, w) in zip(axes, wide.items()):
    ax.plot(list(Ns), w.T.values, color="grey", alpha=0.5, lw=1, marker="o", ms=3)
    ax.plot(list(Ns), w.mean().values, color="maroon", lw=3, marker="o")
    ax.set_title(scheme)
    ax.set_xlabel(XLABEL)
axes[0].set_ylabel(YLABEL)
fig.savefig(f"figures/nested_vs_fresh__{kernel_obj}.pdf")
plt.show()