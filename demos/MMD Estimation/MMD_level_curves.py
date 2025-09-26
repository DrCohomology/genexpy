"""
Visualize the distribution of samples in a 2-simplex, each point corresponding to a probability distribution .

"""

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd
import seaborn as sns

from collections import defaultdict
from importlib import reload
from itertools import product
from pathlib import Path
from time import time
from tqdm.auto import tqdm

from genexpy import kernels as ku
from genexpy import kernels_vectorized as kvu
from genexpy import kernels_classes as kcu
from genexpy import mmd
from genexpy import probability_distributions as du
from genexpy import rankings_utils as ru

reload(du)
reload(ku)
reload(kvu)
reload(kcu)
reload(ru)

def get_pmfs_df_from_multisample(ms: ru.MultiSampleAM, pmf_df_base = pd.Series(name="_"), with_base_pmf=False) -> pd.DataFrame:
    """
    Create a dataframe. Index: rankings. Columns: samples in 'ms'.
    The index can be given with pmf_df_base (optional). It is useful if you have a universe and want to keep the pmfs in
        line with it.

    Parameters
    ----------
    ms : ru.MultiSample

    Returns
    -------
    a pd.DataFrame

    """
    tmps = [pmf_df_base]
    for i, s in enumerate(ms):
        universe, pmf = ru.SampleAM(s).get_universe_pmf()
        tmps.append(pd.Series(pmf, index=universe))

    out = pd.concat(tmps, axis=1, ignore_index=False).fillna(0.0)
    return out if with_base_pmf else out.drop(columns=pmf_df_base.name)


FIGURES_DIR = Path("./demos/MMD Estimation/figures")
FIGURES_DIR.mkdir(exist_ok=True)

na = 2
N = 100
ties = True
seed = 1999
^
n = 10          # size of sample
ns = 1000        # number of samples
npoints = 100   # size of simplex grid
center = np.array([1, 1, 10])
center  = center / center.sum()
# center  = np.array([1, 1, 1])/3

# distr = du.UniformDistribution(na=na, ties=ties, seed=seed+1)
universe = ru.SampleAM.from_rank_vector_matrix(np.array([[0, 1, 0],
                                                         [1, 0, 0]]))
distr = du.PMFDistribution(pmf=center, universe=universe)

pmf_base = pd.Series(center, index=universe, name="center")

multisample = distr.multisample(n=n, nm=ns)
p_ms = get_pmfs_df_from_multisample(multisample, pmf_df_base=pmf_base)

kernel_obj = kcu.MallowsKernel(nu="auto")
# kernel_obj = kcu.JaccardKernel(k=1)
match kernel_obj.vectorized_input_format:
    case "adjmat":
        x = universe.to_adjmat_array((na, na))
    case "vector":
        x = universe.to_rank_vector_matrix()
    case _:
        raise ValueError(f"Unsupported input format for vectorized kernel: {kernel_obj.vectorized_input_format}")
K = kernel_obj.gram_matrix(x, x)
#
# K = np.array(([1, 0.8, 0.7],
#               [0.8, 1, 0.4],
#               [0.7, 0.4, 1]))
# K = np.eye(3)

eigvals_K, eigvecs_K = np.linalg.eig(K)

m = K.shape[0]

def generate_simplex_grid(n=100):
    # Generate grid of all (i, j) pairs such that i + j <= n
    i, j = np.meshgrid(np.arange(n+1), np.arange(n+1))
    mask = i + j <= n
    i = i[mask]
    j = j[mask]
    k = n - i - j

    # Normalize to lie in the 2-simplex
    x = i / n
    y = j / n
    z = k / n
    return np.stack([x, y, z], axis=1).T

simplex = generate_simplex_grid(n=npoints)
diff = simplex - center.reshape(-1, 1)
distances = np.sqrt(np.diag(diff.T @ K @ diff))

# --- with kernel matrix for the simplex and rotation mapping simplex to 2d
K2 = K[:m-1, :m-1] - K[:m-1, m-1].reshape(-1, 1) - K[m-1, :m-1] + K[m-1, m-1]
# TODO: for some centers, the eigenvectors of K2 are (1, 1) and (1, -1)
# TODO: it's not the center, it's the projection chosen: removing 00, now everything is "degenerate"
# TODO: all columns of K2 have the same sum, hence (1, 1) is an eingevector
# TODO: the difficulty index cannot depend on the ordering (and which one is number n)


# sanity check for distances using K2
s = (simplex - center.reshape(-1, 1))[:m-1, :]
# print("K2 approx K?", np.diag(s.T @ K2 @ s - diff.T @ K @ diff).max())
# print("K2 sym?", np.max(K2 - K2.T))

eigvals_K2, eigvecs_K2 = np.linalg.eig(K2)

p_ms_2d = p_ms.values[:m-1, :]

# --- How hard is it to estimate the distribution? Variance in a direction
p_ms_2d_c = p_ms_2d - p_ms_2d.mean(axis=1).reshape(-1, 1)
cov = p_ms_2d_c @ p_ms_2d_c.T / (ns - 1)
var_eig = np.diag(eigvecs_K2.T @ cov @ eigvecs_K2)
difficulty = np.sum(var_eig * eigvals_K2) * n  # arbitrary difficulty of generalizability of distribution

# TODO: difficulty index should NOT depend on n. consider some sort of difficulty curve (with n)
# TODO: ideally, this difficulty index would be related to the MMD more strictly (but isn't it already?) or to the limit

colors = ["cyan", "red"]
# for i in range(m-1):
#     print(f"e{i+1} ({colors[i]}): {eigvals_K2[i]:.2f}")
#     print(f"variance of samples: {var_eig[i]:.3f}")
# print(f"Difficulty index: {difficulty:3f}")

# --- (doubled) Eigenvalues of Tk
C = np.eye(m) - np.ones((m, m)) / m
H = C @ K @ C
Tk = 2 * H @ np.diag(center)
eigvals_Tk = np.maximum(np.linalg.eigvalsh(Tk), 0)

C2 = np.eye(m-1) - np.ones((m-1, m-1)) / (m-1)
H2 = C2 @ K2 @ C2
Tk2 = 2 * H2 @ np.diag(center[:m-1])
eigvals_Tk2 = np.maximum(np.linalg.eigvalsh(Tk2), 0)

# TODO: Tk and Tk2 give different chi-square approximations for the MMD. why?

# --- Multinomial approximation of samples
rng = np.random.default_rng(seed + 3)
p_multinomial = 1 /n * rng.multinomial(n=n, pvals=center, size=ns)
p_nm_2d = p_multinomial[:, :2].T

c2d = center[:m-1].reshape(-1, 1)  # 2d center
cov_nm_2d = (np.diag(c2d.flatten()) - c2d @ c2d.T) / n
cov_nm_3d = (np.diag(center) - center.reshape(-1, 1) @ center.reshape(-1, 1).T) / n

eigvals_C2, eigvecs_C2 = np.linalg.eig(cov_nm_2d)

difficulty_2d = np.sum(np.diag(eigvecs_K2.T @ cov_nm_2d @ eigvecs_K2) * eigvals_K2) * n
difficulty_3d = np.sum(np.diag(eigvecs_K.T @ cov_nm_3d @ eigvecs_K) * eigvals_K) * n

# print("Covariance differences:", np.linalg.norm(cov - cov_nm_2d))
# print("Theoretical difficulty: ", difficulty_3d)

# print("Spectrum of K:   ", tuple(round(x, 2) for x in eigvals_K))
# print("Spectrum of K2:  ", tuple(round(x, 2) for x in eigvals_K2))
# print("Spectrum of Tk:  ", tuple(round(x, 2) for x in eigvals_Tk))
print("Eigvals of K")
print(eigvals_K)
print("2D Covariance")
print(cov_nm_2d)
print("eigvecs K2")
print(eigvecs_K2)
print("Directional variance")
print(eigvecs_K2.T @ cov_nm_2d @ eigvecs_K2)
print("2D difficulty")
print(difficulty_2d)


#%%

# --- Plots
x, y, _ = simplex
center_2d = center[:m-1]
eigvecs_2d = (eigvals_K2 * eigvecs_K2 + center[:m-1].reshape(-1, 1))
cov_eigvecs_2d = (eigvals_C2 * eigvecs_C2 * 100 + center[:m-1].reshape(-1, 1))

# plt.close("all")
fig, ax = plt.subplots(figsize=(10, 10))

ax.set_title(f"n: {n}\n"
             # f"sample 2d-index: {difficulty:.05f}\n"
             f"2D-index: {difficulty_2d:.05f}\n"
             f"3D-index: {difficulty_3d:.05f}\n"
             f"2D-eigsum: {np.sum(eigvals_K2)}\n"
             f"3D-eigsum: {np.sum(eigvals_K)}"
             # f"2 * Eigvals of Tk: {eigvals_Tk[0]:.2f}, {eigvals_Tk[1]:.2f}, {eigvals_Tk[2]:.2f}\n"
             # f"2 * Eigvals of Tk2: {eigvals_Tk2[0]:.2f}, {eigvals_Tk2[1]:.2f}"
             )
ax.set_aspect('equal', adjustable='box')
triang = tri.Triangulation(x, y)
contours = ax.tricontourf(triang, distances, levels=np.linspace(0, np.sqrt(2*(np.max(K)-np.min(K))), 20),
                          cmap=plt.cm.bone, alpha=0.4)

# samples and multinomial sample
ax.scatter(*p_ms_2d, c="blue", alpha=0.1)
# ax.scatter(*p_nm_2d, c="red", marker="x", alpha=0.1)


# for i, color in enumerate(colors):
    # ax.annotate("", xytext=center[:m-1], xy=(eigvecs_2d[0, i], eigvecs_2d[1, i]),
    #             arrowprops=dict(arrowstyle="->", color=color, linewidth=2))
for i, color in enumerate(colors):
    ax.annotate("", xytext=center[:m-1], xy=(cov_eigvecs_2d[0, i], cov_eigvecs_2d[1, i]),
                arrowprops=dict(arrowstyle="->", color=color, linewidth=2, ls=":"))

ax.scatter(*center_2d, c="white", marker="*", s=200)

rv_plot = universe.to_rank_vector_matrix().T
ax.text(s=tuple(rv_plot[0]), x=1, y=0)   # universe[0]
ax.text(s=tuple(rv_plot[1]), x=0, y=1)   # universe[1]
ax.text(s=tuple(rv_plot[2]), x=0, y=0)   # universe[2]

sns.despine(top=True, bottom=True, left=True, right=True)
ax.set_xticks([])
ax.set_yticks([])

cbar = fig.colorbar(contours)
cbar.ax.set_ylabel("MMD from center")

wm = plt.get_current_fig_manager()
# wm.window.state('zoomed')
# plt.tight_layout()
plt.show()

# TODO: weird behavior when swapping axes.

#%% 2. Test relation between eigenvectors of K and K2

eigvals_K, eigvecs_K = np.linalg.eig(K)

v = eigvecs_K[:, 0].copy().reshape(-1, 1)
v /= v.sum()   # v should belong to the simplex
# v -= np.eye(3)[:, 2].reshape(-1, 1)

# one needs v eigenvector of K and in the n-1-simplex0! Otherwise there's no way this works.
# such an eigenvector exists iff the sums of columns of K are the same

l = eigvals_K[0]
v2 = v[:2]

t = v.T @ K @ v
t_ = l * v.T @ v

t2 = v2.T @ K2 @ v2
t2_ = l * v2.T @ (np.eye(2) + np.ones((2, 2))) @ v2

print(t, t_, t2, t2_)

#%% 2a. decompose vectors in 2-simplex - 1 = 2-simplex0 (2-simplex translated to the origin)

v = np.array([1, 2, 3]).reshape(-1, 1)
v = v / v.sum()
v = v - np.eye(3)[:, 2].reshape(-1, 1)
t = v.T @ K @ v

v_ = eigvecs_K @ v  # change of basis
v_ = np.sqrt(eigvals_K).reshape(-1, 1) * v_
t_ = v_.T @ v

print(t, t_)

#%% 2b. rewrite K2 in coordinates given by the eigenvectors of K

# v in D0 (2-simplex0)
v = np.array([1, 2, 3]).reshape(-1, 1)
v = v / v.sum()
v = v - np.eye(3)[:, 2].reshape(-1, 1)

# v2 is v without the n-th component BUT in canonical coordinates
v2 = v.copy()[:2].reshape(-1, 1)

# change of basis for v
v = eigvecs_K @ v

t = v.T @ K @ v

# it's in a somewhat easy shape!
K2_ = np.diag(eigvals_K[:2]) + eigvals_K[2] * np.ones((2, 2))
t_ = v2.T @ K2_ @ v2
print(t, t_)


eigvals_K2_, eigvecs_K2_ = np.linalg.eig(K2_)
# TODO: investigate when lambda_n of K is 0. My guess is that all but one eigenvectors lie on D0, the other one being ones(n)

#%% 2c. rewrite in coordinates given by the eigenvectors of K2

# v in D0 (2-simplex0)
v = np.array([1, 2, 3]).reshape(-1, 1)
v = v / v.sum()
v = v - np.eye(3)[:, 2].reshape(-1, 1)

# v2 is v without the n-th component BUT in canonical coordinates
v2 = v.copy()[:2].reshape(-1, 1)

# change of basis for v2
v2 = eigvecs_K2 @ v2

# %% 3a. Level curves of difficulty wrt p

# assumes K is already computed somewhere

K = np.array(([1, 0.1, 0.1],
              [0.1, 1, 0.5],
              [0.1, 0.5, 1]))

K2 = K[:m-1, :m-1] - K[:m-1, m-1].reshape(-1, 1) - K[m-1, :m-1] + K[m-1, m-1]
eigvals_K2, eigvecs_K2 = np.linalg.eig(K2)

ns = 100 + 1
simplex_grid = generate_simplex_grid(n=ns)[:2]

diff2d = []
for c2d in simplex_grid.T.copy():
    c2d = np.array(c2d).reshape(-1, 1)  # where to calculate the difficulty
    cov2d = (np.diag(c2d.flatten()) - c2d @ c2d.T)  # covariance matrix not rescaled with n
    diff2d.append(np.sum(np.diag(eigvecs_K2.T @ cov2d @ eigvecs_K2) * eigvals_K2))

x, y = simplex_grid

fig, ax = plt.subplots(figsize=(10, 10))

ax.set_title(f"Difficulty as a function of the center\n"
             f"K = {K}", ha='right')
ax.set_aspect('equal', adjustable='box')
triang = tri.Triangulation(x, y)
contours = ax.tricontourf(triang, diff2d, levels=ns, cmap="coolwarm")

rv_plot = universe.to_rank_vector_matrix().T
ax.text(s=tuple(rv_plot[0]), x=1, y=0)   # universe[0]
ax.text(s=tuple(rv_plot[1]), x=0, y=1)   # universe[1]
ax.text(s=tuple(rv_plot[2]), x=0, y=0)   # universe[2]

sns.despine(top=True, bottom=True, left=True, right=True)
ax.set_xticks([])
ax.set_yticks([])

cbar = fig.colorbar(contours)
cbar.ax.set_ylabel("Difficulty of center")

wm = plt.get_current_fig_manager()
# wm.window.state('zoomed')
# plt.tight_layout()
plt.show()

#%% 4. Test: can we replicate the 2D-3D equality for difficulty wqith arbiotrary covariance metrices? NO: Id does not work (it's the sum of the eigenvalues)
# the structure of C2 and C3 is probably the key to understanding why it works so well

x = np.array([1, 10, 2]).reshape(-1, 1)
x = x / x.sum()


C3 = np.diag(x.flatten()) - x @ x.T
# C3 = cov_nm_3d * n
# C3 = np.eye(3)

# C3 = np.array([[1, 2, -3],
#                [2, 2, -4],
#                [-3, -4, 7]])

C2 = C3[:2, :2]

eigvals_C2, eigvecs_C2 = np.linalg.eig(C2)
eigvals_C3, eigvecs_C3 = np.linalg.eig(C3)
# print(np.diag(eigvecs_C2.T @ K2 @ eigvecs_C2 * eigvals_C2).sum())
# print(np.diag(eigvecs_C3.T @ K @ eigvecs_C3 * eigvals_C3).sum())

print("2D:\t", np.sum(np.diag(eigvecs_K2.T @ C2 @ eigvecs_K2) * eigvals_K2))
print("3D:\t", np.sum(np.diag(eigvecs_K.T @ C3 @ eigvecs_K) * eigvals_K))

"""
Why does the 3D-2D thing work?
it looks like the important part are: 
    1. the structure of the covariance matrix 
        symmetric stochastic matrix
    2. the fact that the center sums to 1
        which guarantees the first part    
    
The covariance has rank 2 and is symmetric. is it enough?
"""

pass

