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

from collections import Counter, defaultdict
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

# fig.savefig("rankings_distribution.pdf")


#%% Algorithm for visualization
"""
Based on Terada and Luxburg (2014)
"""







