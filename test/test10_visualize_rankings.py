"""
Assign to each ranking an equally-sized portion of disc.
The rankings are arranged in concentric circular crowns, where the most internal is for the 0-ranking,
    the next is for rankings with 2 unique ranks, and th n-th crown hosts the rankings with n unique ranks.

To calculate the radii of the crowns, let
    - n be the number of unique ranks
    - rn be the radius of the n-th crown
    - Tn be the number of rankings with n unique ranks
        terms n(n-1)/2 + 1 to n(n+1)/2 in T(n, k) in https://oeis.org/A019538

Then, it holds true that
    rn^2 = Tn r1^2 + r(n-1)^2
and thus that
    rn^2 = sum(Tm, m=1...n) r1^2

Rankings need to be arranged in a particular way to have similar rankings close together.
    Still unsolved.
"""

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
from scipy.stats import binomtest, friedmanchisquare
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from genexpy import lower_bounds as gu
from genexpy import kernels as ku
from genexpy import rankings_utils as ru
from genexpy import mmd as mmd
from genexpy import probability_distributions as du


#%%
reload(du)

na = 5  # alternatives
nv = 100  # voters
r1 = 1  # radius of inner circle

Ts = du.get_unique_ranks_distribution(n=na, exact=True, normalized=False)
rs = np.append([0], np.sqrt(np.cumsum(Ts)) * r1)

# ranking
sample = du.UniformDistribution(seed=123, na=na, ties=True).sample(nv)
# sample = ru.SampleAM.from_rank_function_matrix(np.array(list(product([0, 1, 2], repeat=na))).T)
distr = du.PMFDistribution.from_sample(sample)

rankings = distr.universe.to_rank_function_matrix().T
pmf = distr.pmf / distr.pmf.sum()
pmf_alphas = 0.5 + 0.5*(pmf - pmf.min()) / (pmf.max() - pmf.min())
# pmf_alphas = pmf * 100

# plot
fig, ax = plt.subplots()

ax.set_title(f"Distribution of rankings with {na} alternatives. T{na} = {Ts[-1]}")

# setup the disk
disk_alphas = np.linspace(0.3, 0.2, len(rs))
for n, (rn, alpha) in enumerate(zip(rs, disk_alphas)):
    disk = plt.Circle((0, 0), rn, color='black', fill=True, alpha=alpha, ec="black")
    ax.add_patch(disk)

for ith, (v, alpha) in enumerate(zip(rankings, pmf_alphas)):
    # get the number of unique ranks
    nur = len(np.unique(v))

    # get radius
    rv = rs[nur-1]
    r_inner = rs[nur-1]   # Inner circle radius
    r_outer = rs[nur]   # Outer circle radius

    Tn = Ts[nur-1]

    # angles
    th0 = ith * 2*np.pi / Tn
    th1 = ((ith+1)) * 2*np.pi / Tn

    if th0 == th1:
        th1 = 2 * np.pi

    # Create the vertices for the filled sector
    theta = np.linspace(th0, th1, 100)
    ps = np.column_stack((np.cos(theta), np.sin(theta)))
    outer_points = r_outer * ps
    inner_points = r_inner * ps
    points = np.concatenate((outer_points, inner_points[::-1]), axis=0)

    sector = Polygon(points, closed=True, facecolor="red", alpha=alpha, edgecolor="red")
    ax.add_patch(sector)

    offset = np.max(rs) * 0.05
    ax.set_xlim(-np.max(rs) - offset, np.max(rs) + offset)
    ax.set_ylim(-np.max(rs) - offset, np.max(rs) + offset)
    ax.set_aspect('equal', 'box')

fig.show()



# %% 2. SOE

import torch

from genexpy.src import soe

rng = np.random.default_rng(1444)
kernel = ku.mallows_kernel

soeargs = {
    "epochs": 80,
    "batch_size": 10,
    "learning_rate": 1e-2,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

def embed(sample: ru.SampleAM, kernel: ku.Kernel, **soeargs):
    universe, _ = sample.get_universe_pmf()
    nu = len(universe)

    # need to return (xi, xj, xk) st k(xi, xj) >= k(xi, xk)
    candidate_triplets = rng.integers(0, nu, size=(1000, 3))
    triplets = np.empty_like(candidate_triplets)
    for it, t in enumerate(candidate_triplets):
        trs = ru.SampleAM(universe[t])  # get the corresponding rankings
        # kernel (= -distance) between rankings
        gram = ku.square_gram_matrix(trs, use_rv=True, kernel=kernel)
        k01 = gram[0, 1]
        k02 = gram[0, 2]
        k12 = gram[1, 2]

        # find most similar pair
        idx_max = np.argmax([k01, k02, k12])
        match idx_max:
            case 0: triplets[it] = t[[0, 1, 2]]
            case 1: triplets[it] = t[[0, 2, 1]]
            case 2: triplets[it] = t[[1, 2, 0]]

    X, loss_history, triplet_error_history, time_taken, time_history = soe.soe_adam(triplets, n=len(sample), dim=2, **soeargs)

    return X, loss_history

sample_partition = sample.partition_with_ntiers()

Xtiers = {}
losstiers = {}
for ntier, sampletier in tqdm(list(sample_partition.items())):
    Xtiers[ntier], losstiers[ntier] = embed(sampletier, kernel=kernel, **soeargs)

# entire sample
Xtiers[0], losstiers[0] = embed(sample, kernel=kernel, **soeargs)

fig, axes = plt.subplots(1, len(Xtiers), figsize=(10, 5), sharex=True, sharey=True)

for ax, (ntier, Xtier) in zip(axes, Xtiers.items()):
    ax.set_title(ntier)
    sns.scatterplot(x = Xtier[:, 0], y = Xtier[:, 1], ax=ax)

ax = axes[-1]

plt.tight_layout()
fig.savefig("soe_embedding.pdf")
fig.show()

# %% 2.1 SOE different distributions

seed = 1444
rng = np.random.default_rng(seed)
kernel = ku.mallows_kernel
na = 10
nv = 100
reps = 3  # number of repetitions per distribution

soeargs = {
    "epochs": 80,
    "batch_size": 10,
    "learning_rate": 1e-2,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

distrs = {
    "uniform": du.UniformDistribution(seed=seed+1, na=na, ties=True),
    "spike": du.SpikeDistribution(seed=seed+2, na=na, ties=True, kernel=ku.mallows_kernel),
    "degenerate5": du.MDegenerateDistribution(m=5, seed=seed+3, na=na, ties=True)
}

samples = {}
for name, distr in distrs.items():
    samples.update({f"{name}_{rep}": distr.sample(nv) for rep in range(reps)})

Xs = {}
losses = {}
for name, sample in tqdm(list(samples.items())):
    Xs[name], losses[name] = embed(sample, kernel=kernel, **soeargs)

# %% 2.1a plot
fig, axes = plt.subplots(len(distrs), reps, figsize=(10, 5), sharex=True, sharey=True)

for ax, (name, X) in zip(axes.flatten(), Xs.items()):
    ax.set_title(name)
    sns.scatterplot(x = X[:, 0], y = X[:, 1], ax=ax)

plt.tight_layout()
fig.savefig("soe_embedding.pdf")
fig.show()

# %% 2.1b plot losses

fig, axes = plt.subplots(len(distrs), reps, figsize=(10, 5), sharex=True, sharey=True)

for ax, (name, loss) in zip(axes.flatten(), losses.items()):
    ax.set_title(name)
    ax.plot(loss)

plt.tight_layout()
fig.savefig("soe_loss.pdf")
fig.show()

# %% 2.2 SOE different distributions same universe

reload(ru)
reload(du)

seed = 1444
rng = np.random.default_rng(seed)
kernel = ku.jaccard_kernel
na = 8
nv = 1000
reps = 3  # number of repetitions per distribution

def embed(sample: ru.SampleAM, kernel: ku.Kernel, universe=None, **soeargs):
    """
    Based on Terada and Luxburg (2014), implementation adapted from https://github.com/tml-tuebingen/evaluate-OE/tree/main
        (Vankadara, 2023)
    """
    if universe is None:
        universe, _ = sample.get_universe_pmf()
    nu = len(universe)

    # need to return (xi, xj, xk) st k(xi, xj) >= k(xi, xk)
    candidate_triplets = rng.integers(0, nu, size=(100000, 3))
    triplets = np.empty_like(candidate_triplets)
    for it, t in enumerate(candidate_triplets):
        trs = ru.SampleAM(universe[t])  # get the corresponding rankings
        # kernel (= -distance) between rankings
        gram = ku.square_gram_matrix(trs, use_rv=True, kernel=kernel)
        k01 = gram[0, 1]
        k02 = gram[0, 2]
        k12 = gram[1, 2]

        # find most similar pair
        idx_max = np.argmax([k01, k02, k12])
        match idx_max:
            case 0: triplets[it] = t[[0, 1, 2]]
            case 1: triplets[it] = t[[0, 2, 1]]
            case 2: triplets[it] = t[[1, 2, 0]]

    X, loss_history, triplet_error_history, time_taken, time_history = soe.soe_adam(triplets, n=len(sample), dim=2, **soeargs)

    return X, loss_history

soeargs = {
    "epochs": 10,
    "batch_size": 100,
    "learning_rate": 1e-1,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

distrs = {
    "uniform": du.UniformDistribution(seed=seed+1, na=na, ties=True),
    "spike": du.SpikeDistribution(seed=seed+2, na=na, ties=True, kernel=kernel, uniform_size_sample=1000),
    "degenerate5": du.MDegenerateDistribution(m=5, seed=seed+3, na=na, ties=True)
}

# get samples and update the universe
universe = ru.UniverseAM(np.array([]))
samples = {}
sample_universes_pmf = {}
centers = {}
for name, distr in tqdm(list(distrs.items())):
    for rep in range(reps):
        sample = distr.sample(nv)
        samples.update({f"{name}_{rep}": sample})
        sample_universes_pmf.update({f"{name}_{rep}": sample.get_universe_pmf()})
        universe = universe.merge(sample)
        if "Spike" in distr.name:
            centers[rep] = distr.centertmp

# get embedding of the rankings
X, loss = embed(sample=universe, kernel=kernel, universe=universe, **soeargs)
embedding = {rb: x for rb, x in zip(universe, X)}

# embed each sample
Xs = {}
for name, sample in samples.items():
    Xs[name] = np.array([embedding[rb] for rb in sample])

# %% 2.2a plot

# get dataframe
xs = {name: x[:, 0] for name, x in Xs.items()}
ys = {name: x[:, 1] for name, x in Xs.items()}

dfplot = pd.DataFrame(xs).melt(value_name="x", var_name="distribution")
dfplot["y"] = pd.DataFrame(ys).melt(value_name="y", var_name="distribution")["y"]

zero = ru.AdjacencyMatrix.zero(na)

fig, axes = plt.subplots(len(distrs), reps, figsize=(10, 10), sharex=True, sharey=True)

# i = 0
for ax, (name, _) in zip(axes.flatten(), Xs.items()):
    ax.set_title(name)
    ax.set(xlabel=None, ylabel=None, xticks=[], yticks=[])
    sns.scatterplot(data=dfplot.query("distribution == @name"), x="x", y="y", ax=ax, alpha=0.25, color="blue")
    sns.kdeplot(data=dfplot.query("distribution == @name"), x="x", y="y", ax=ax, alpha=0.5, color="blue")

    ax.scatter(zero[0], zero[1], color="red")


    # if "spike" in name:
    #     center_emb = embedding[centers[i]]
    #     ax.scatter(center_emb[0], center_emb[1], color="red", marker="x")
    #     i += 1

plt.tight_layout()
fig.savefig("soe_embedding.pdf")
fig.show()






# fig, ax = plt.subplots()
#
# sns.kdeplot(data=dfplot, x="x", y="y", hue="distribution", levels=3, ax=ax)
#
# sns.despine(bottom=True, left=True)
#
# plt.tight_layout()
# fig.savefig("soe_embedding.pdf")
# fig.show()


# %% 2.4 hyperperameters of SOE. Seems to be little to no effect. Might be that the optimization is just impossible and it never learns.

from itertools import product

seed = 1444
rng = np.random.default_rng(seed)
kernel = ku.jaccard_kernel
na = 8
nv = 1000
reps = 3  # number of repetitions per distribution

def embed(sample: ru.SampleAM, kernel: ku.Kernel, universe=None, **soeargs):
    """
    Based on Terada and Luxburg (2014), implementation adapted from https://github.com/tml-tuebingen/evaluate-OE/tree/main
        (Vankadara, 2023)
    """
    if universe is None:
        universe, _ = sample.get_universe_pmf()
    nu = len(universe)

    # need to return (xi, xj, xk) st k(xi, xj) >= k(xi, xk)
    candidate_triplets = rng.integers(0, nu, size=(100000, 3))
    triplets = np.empty_like(candidate_triplets)
    for it, t in enumerate(candidate_triplets):
        trs = ru.SampleAM(universe[t])  # get the corresponding rankings
        # kernel (= -distance) between rankings
        gram = ku.square_gram_matrix(trs, use_rv=True, kernel=kernel)
        k01 = gram[0, 1]
        k02 = gram[0, 2]
        k12 = gram[1, 2]

        # find most similar pair
        idx_max = np.argmax([k01, k02, k12])
        match idx_max:
            case 0: triplets[it] = t[[0, 1, 2]]
            case 1: triplets[it] = t[[0, 2, 1]]
            case 2: triplets[it] = t[[1, 2, 0]]

    X, loss_history, triplet_error_history, time_taken, time_history = soe.soe_adam(triplets, n=len(sample), dim=2, **soeargs)

    return X, loss_history, triplet_error_history, time_taken, time_history

distrs = {
    "uniform": du.UniformDistribution(seed=seed+1, na=na, ties=True),
    "spike": du.SpikeDistribution(seed=seed+2, na=na, ties=True, kernel=kernel, uniform_size_sample=1000),
    "degenerate5": du.MDegenerateDistribution(m=5, seed=seed+3, na=na, ties=True)
}

# get samples and update the universe
universe = ru.UniverseAM(np.array([]))
samples = {}
sample_universes_pmf = {}
centers = {}
for name, distr in tqdm(list(distrs.items())):
    for rep in range(reps):
        sample = distr.sample(nv)
        samples.update({f"{name}_{rep}": sample})
        sample_universes_pmf.update({f"{name}_{rep}": sample.get_universe_pmf()})
        universe = universe.merge(sample)
        if "Spike" in distr.name:
            centers[rep] = distr.centertmp

hpars_grid = [dict(epochs=epochs, batch_size=100, learning_rate=lr,
                   device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
              for epochs, lr in product(range(10, 51, 10), np.logspace(-4, 0, 5))]


out = []  # will become a dataframe
other_out = {}  # will not become a dataframe
for soeargs in tqdm(hpars_grid):

    # get embedding of the rankings
    X, loss_history, triplet_error_history, time_taken, time_history = embed(sample=universe, kernel=kernel, universe=universe, **soeargs)
    embedding = {rb: x for rb, x in zip(universe, X)}

    # embed each sample
    Xs = {}
    for name, sample in samples.items():
        Xs[name] = np.array([embedding[rb] for rb in sample])

    exp_id = {
        "epochs": soeargs["epochs"],
        "learning_rate": soeargs["learning_rate"],
    }

    tmp = exp_id.copy()
    tmp.update({
        "final_loss": loss[-1],
        "final_triplet_error": triplet_error_history[-1],
        "time_taken": time_taken
    })
    out.append(tmp)

    other_out[(exp_id["epochs"], exp_id["learning_rate"])] = {
        "embedding": embedding,
        "loss_history": loss_history,
        "triplet_error_history": triplet_error_history,
        "time_history": time_history
    }


#%%

dfout = pd.DataFrame(out)

y = "final_triplet_error"

fig, axes = plt.subplots(1, 2)

ax = axes[0]
sns.lineplot(data=dfout, x="epochs", y=y, hue="learning_rate", ax=ax)

ax = axes[1]
sns.lineplot(data=dfout, x="learning_rate", y=y, hue="epochs", ax=ax)
ax.set(xscale="log")

plt.tight_layout()
plt.show()

#%%

losses = [{"epochs": epochs, "lr": lr, "step": i, "loss": loss} for i, ((epochs, lr), loss) in enumerate(other_out.items())]


