"""
Timing comparison of MallowsKernel.mmd_distribution methods.

Compares, as a function of the number of repetitions, the subsample size, and
the support size:
    vectorized  mean(Kxx) + mean(Kyy) - 2 mean(Kxy) on each subsample
    embedding   the quadratic form alpha.T @ K @ alpha over the support

Writes mmd_method_timings.csv and mmd_method_timings.pdf.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from genexpy.utils import rankings as ru
from genexpy.kernels.rankings import MallowsKernel

# --- figure settings ---------------------------------------------------------
LABELS = {
    "time": "runtime [s]",
    "rep": "repetitions",
    "n": "subsample size",
    "m": "support size",
}
FONTSIZE = 20
FIGSIZE = (19, 6)
OUTFILE = "mmd_method_timings"
LOG_Y = False  # the curves span two decades; set True to read them all at once

plt.rcParams.update({
    "font.size": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE - 4,
})

# --- benchmark settings ------------------------------------------------------
NA = 8           # number of alternatives
N_DEFAULT = 400  # number of experimental conditions
N_SUB = 20       # subsample size
REP = 2000       # repetitions
TIMING_REPEATS = 2
SEED = 1

METHODS = ["vectorized", "embedding"]
MARKERS = ["o", "^"]


def make_sample(na: int, N: int, seed: int = 0) -> ru.SampleAM:
    """A sample of N rankings over na alternatives, with ties."""
    rng = np.random.default_rng(seed)
    rv = np.array([rng.integers(0, 3, size=na) for _ in range(N)]).T
    return ru.SampleAM.from_rank_vector_matrix(rv)


def time_call(fn, repeats: int = TIMING_REPEATS) -> float:
    """Best-of wall time in seconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def benchmark(sample: ru.SampleAM, n: int, rep: int) -> dict:
    """Time both methods on one configuration. The Gram matrix is never pre-cached."""
    support = sample.get_support_pmf()[0]
    out = {"m": len(support), "n": n, "rep": rep}

    for method in METHODS:
        kernel = MallowsKernel(nu="auto", na=NA)
        kernel.set_support(support)
        out[method] = time_call(
            lambda: kernel.mmd_distribution(sample, n=n, rep=rep, seed=SEED, method=method)
        )

    return out


# --- sweeps ------------------------------------------------------------------
records = []

sample = make_sample(NA, N_DEFAULT)
for rep in [500, 1000, 2000, 3000, 4000, 5000]:
    records.append({"sweep": "rep", **benchmark(sample, n=N_SUB, rep=rep)})
    print(records[-1])

for n in [5, 10, 20, 40, 60, 80, 120, 160]:
    records.append({"sweep": "n", **benchmark(sample, n=n, rep=REP)})
    print(records[-1])

for N in [50, 100, 200, 400, 800, 1500, 2500, 4000, 6000, 9000]:
    records.append({"sweep": "m", **benchmark(make_sample(NA, N), n=N_SUB, rep=REP)})
    print(records[-1])

df = pd.DataFrame(records)
df.to_csv(f"{OUTFILE}.csv", index=False)

# --- figure ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
for ax, sweep in zip(axes, ["rep", "n", "m"]):
    sub = df[df["sweep"] == sweep].sort_values(sweep)
    for method, marker in zip(METHODS, MARKERS):
        ax.plot(sub[sweep], sub[method], marker=marker, markersize=9, linewidth=2, label=method)
    ax.set_xlabel(LABELS[sweep])
    ax.set_ylabel(LABELS["time"])
    if LOG_Y:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)

axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(f"{OUTFILE}.pdf", bbox_inches="tight")
print(f"\nwrote {OUTFILE}.csv and {OUTFILE}.pdf")