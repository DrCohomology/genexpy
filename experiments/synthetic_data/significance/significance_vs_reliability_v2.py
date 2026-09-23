"""
Appendix A -- significance vs. reliability of an experimental study.

Simulation for the redesigned toy experiment of Appendix A.

Model of experimental results (Thurstone-type)
----------------------------------------------
Each experimental condition c yields a performance for every alternative a,
    Y[c, a] = mu[a] + sigma[a] * Z[c, a],     Z ~ N(0, 1) iid,
and the result of the experiment is the induced ranking (rank 0 = best).
The distribution of results P is controlled by the gap Delta = mu[0] - mu[1],
with mu = (Delta, *MU_REST). Rankings are permutations (no ties a.s.).

Research question: "Is the same alternative consistently the best?"
-> Jaccard kernel with t = 1 (top tier), as in Section 4.1.

For every cell (Delta, N) of the grid we compute
  * Pr(significant): probability that a study with N conditions finds a
    significantly best alternative (Friedman omnibus + post-hoc of the sample
    winner against every other alternative, Holm-corrected by default);
  * the probability that an independent replication of the same size is also
    significant *with the same winner*;
  * the reliability R^k(P, N, eps*), i.e. the probability that two independent
    studies of size N yield MMD_k < eps* (Monte Carlo on P);
and, for a few selected cells, the per-study plug-in estimate of reliability
computed from the study's own sample, split by the outcome of the test.

Outputs (in --out):
  grid.csv, per_study.csv, table_grid.tex (booktabs),
  fig_power_reliability.pdf, fig_estimated_reliability.pdf

Usage
-----
  python significance_vs_reliability.py --out figures           # full run
  python significance_vs_reliability.py --out figures --quick   # smoke test

Dependencies: numpy, scipy, pandas, matplotlib
              (scikit-posthocs only for the Conover / Nemenyi post-hoc options).
"""
from __future__ import annotations

import argparse
import itertools
import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Configuration
# =============================================================================
@dataclass(frozen=True)
class Config:
    # --- distribution of results P ---
    na: int = 5                                   # number of alternatives
    deltas: tuple = (0.0, 0.25, 0.5, 1.0, 2.0)    # gap mu[0] - mu[1]
    mu_rest: tuple = (0.0, 0.0, 0.0, 0.0)         # mu[1:], length na - 1
    sigma: tuple = (1.0, 1.0, 1.0, 1.0, 1.0)      # per-alternative noise

    # --- studies ---
    Ns: tuple = (5, 10, 20, 30, 40, 60, 80)       # study sizes (number of conditions)
    n_studies: int = 1000                         # simulated studies per cell

    # --- significance ---
    alpha_test: float = 0.05
    omnibus: bool = True                          # require Friedman rejection first
    posthoc: str = "wilcoxon-holm"                # "wilcoxon-holm" | "conover-holm" | "conover" | "nemenyi"
    wilcoxon_method: str = "auto"                 # passed to scipy.stats.wilcoxon

    # --- reliability ---
    kernel: str = "jaccard"                       # "jaccard" | "borda" | "mallows"
    jaccard_t: int = 1
    borda_astar: int = 0
    alpha_star: float = 0.95
    delta_star: float = 0.05
    n_mc_true: int = 20000                        # MC pairs for the true reliability

    # --- per-study estimates of reliability ---
    per_study_cells: tuple = ((0.5, 20), (0.0, 80))   # (Delta, N) pairs
    est_scheme: str = "with_replacement"          # "with_replacement" | "without_replacement"
    est_n_fraction: float = 1.0                   # n = round(fraction * N); must be <= 0.5 without replacement
    n_resamples: int = 500

    seed: int = 20260916


# Axis / legend text: edit here to rename anything in the figures.
LABELS = {
    "N": r"Study size $N$",
    "power": r"Pr(study is significant)",
    "reliability": r"Reliability $R^{k}(P, N, \varepsilon^{*})$",
    "rhat": r"Estimated reliability $\hat{R}^{k}$",
    "delta_legend": r"$\Delta$",
    "alpha_star": r"$\alpha^{*}$",
    "alpha_test": r"test level",
    "true_R": r"true $R^{k}$",
    "groups": ("not significant", r"significant, winner $a_0$", r"significant, winner $\neq a_0$"),
    "panel_title": r"$\Delta = {delta:g}$, $N = {N}$",
}

FONT_SIZE = 20                     # all figure text (labels, ticks, legend)
FIG_FORMAT = "pdf"

# Colours (validated reference palette): ordinal blue ramp for Delta,
# categorical slots for the three outcome groups.
DELTA_COLORS = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")
GROUP_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# =============================================================================
# Distribution of results
# =============================================================================
def mu_vector(cfg: Config, delta: float) -> np.ndarray:
    mu = np.array((delta, *cfg.mu_rest), dtype=float)
    assert mu.size == cfg.na, "mu_rest must have length na - 1"
    return mu


def sample_rank_vectors(rng, mu, sigma, size) -> np.ndarray:
    """Rank vectors r[..., a] in {0, ..., na-1}, 0 = best, of shape (*size, na)."""
    y = mu + np.asarray(sigma) * rng.standard_normal((*size, mu.size))
    order = np.argsort(-y, axis=-1)
    return np.argsort(order, axis=-1).astype(np.int8)


# =============================================================================
# Space of rankings (permutations) and kernels
# =============================================================================
class PermutationSpace:
    """Enumerates S_na and maps rank vectors to indices in O(1)."""

    def __init__(self, na: int):
        self.na = na
        self.rv = np.array(list(itertools.permutations(range(na))), dtype=np.int8)
        self._pow = na ** np.arange(na)
        self._lookup = np.full(na ** na, -1, dtype=np.int32)
        self._lookup[self.rv.astype(np.int64) @ self._pow] = np.arange(len(self.rv))

    def __len__(self):
        return len(self.rv)

    def index(self, rv: np.ndarray) -> np.ndarray:
        return self._lookup[rv.astype(np.int64) @ self._pow]


def kernel_value(cfg: Config, r1: np.ndarray, r2: np.ndarray) -> float:
    na = cfg.na
    if cfg.kernel == "jaccard":
        a, b = set(np.flatnonzero(r1 < cfg.jaccard_t)), set(np.flatnonzero(r2 < cfg.jaccard_t))
        return len(a & b) / len(a | b)
    if cfg.kernel == "borda":            # nu = 1 / na
        b1 = np.sum(r1 >= r1[cfg.borda_astar])
        b2 = np.sum(r2 >= r2[cfg.borda_astar])
        return math.exp(-abs(b1 - b2) / na)
    if cfg.kernel == "mallows":          # nu = 1 / binom(na, 2); nd = discordant unordered pairs
        s1 = np.sign(r1[:, None] - r1[None, :])
        s2 = np.sign(r2[:, None] - r2[None, :])
        nd = np.abs(s1 - s2).sum() / 4     # each discordant pair appears twice with |.| = 2
        return math.exp(-nd / math.comb(na, 2))
    raise ValueError(cfg.kernel)


def epsilon_star(cfg: Config) -> float:
    """eps* = sqrt(2 (k_sup - f_k(delta*))), Section 4.2 (recommended kernel parameters)."""
    if cfg.kernel == "jaccard":
        return math.sqrt(2 * cfg.delta_star)
    return math.sqrt(2 * (1 - math.exp(-cfg.delta_star)))


def gram_matrix(cfg: Config, space: PermutationSpace) -> np.ndarray:
    m = len(space)
    G = np.empty((m, m))
    for i in range(m):
        for j in range(i, m):
            G[i, j] = G[j, i] = kernel_value(cfg, space.rv[i], space.rv[j])
    return G


def histograms(idx: np.ndarray, m: int) -> np.ndarray:
    """Row-wise normalised histograms of integer indices, idx of shape (B, n) -> (B, m)."""
    B, n = idx.shape
    flat = (idx + m * np.arange(B)[:, None]).ravel()
    return np.bincount(flat, minlength=B * m).reshape(B, m) / n


def mmd(hx: np.ndarray, hy: np.ndarray, G: np.ndarray) -> np.ndarray:
    """MMD between empirical measures given as histograms on the support (V-statistic, eq. (2))."""
    D = hx - hy
    return np.sqrt(np.clip(np.einsum("bi,ij,bj->b", D, G, D), 0.0, None))


# =============================================================================
# Significance
# =============================================================================
def friedman_pvalues(R: np.ndarray) -> np.ndarray:
    """Vectorised Friedman test; R has shape (M, N, k) with untied ranks 0..k-1."""
    _, N, k = R.shape
    Rj = (R.astype(float) + 1).sum(axis=1)
    chi2 = 12.0 / (N * k * (k + 1)) * (Rj ** 2).sum(axis=1) - 3.0 * N * (k + 1)
    return stats.chi2.sf(chi2, k - 1)


def holm(p: np.ndarray) -> np.ndarray:
    m = p.shape[-1]
    order = np.argsort(p, axis=-1)
    ps = np.take_along_axis(p, order, axis=-1)
    adj = np.minimum(np.maximum.accumulate(ps * (m - np.arange(m)), axis=-1), 1.0)
    out = np.empty_like(adj)
    np.put_along_axis(out, order, adj, axis=-1)
    return out


def posthoc_pvalues(cfg: Config, R: np.ndarray, winner: np.ndarray) -> np.ndarray:
    """p-values of winner vs. each other alternative, shape (M, k-1)."""
    M, N, k = R.shape
    others = np.array([np.delete(np.arange(k), w) for w in range(k)])[winner]      # (M, k-1)

    if cfg.posthoc == "wilcoxon-holm":
        rw = np.take_along_axis(R, winner[:, None, None].repeat(N, 1), axis=2).astype(int)
        ro = np.take_along_axis(R, np.broadcast_to(others[:, None, :], (M, N, k - 1)), axis=2).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = stats.wilcoxon(rw - ro, axis=1, method=cfg.wilcoxon_method).pvalue
        return holm(np.atleast_2d(p))

    import scikit_posthocs as sp  # optional dependency
    fun = {"conover-holm": sp.posthoc_conover_friedman, "conover": sp.posthoc_conover_friedman,
           "nemenyi": sp.posthoc_nemenyi_friedman}[cfg.posthoc]
    p = np.empty((M, k - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in range(M):
            P = np.asarray(fun(R[s]) if cfg.posthoc == "nemenyi" else fun(R[s], p_adjust=None))
            p[s] = P[winner[s], others[s]]
    return holm(p) if cfg.posthoc == "conover-holm" else p


def significance(cfg: Config, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (significant, winner). The winner is the alternative with the best mean rank."""
    winner = R.mean(axis=1).argmin(axis=1)
    sig = (posthoc_pvalues(cfg, R, winner) < cfg.alpha_test).all(axis=1)
    if cfg.omnibus:
        sig &= friedman_pvalues(R) < cfg.alpha_test
    return sig, winner


# =============================================================================
# Reliability
# =============================================================================
def true_reliability(cfg, rng, space, G, mu, n) -> float:
    """R^k(P, n, eps*) = Pr_{X, Y ~ P^n}(MMD < eps*), Monte Carlo in chunks."""
    eps, hits, done, chunk = epsilon_star(cfg), 0, 0, max(1, 2_000_000 // (2 * n * cfg.na))
    while done < cfg.n_mc_true:
        b = min(chunk, cfg.n_mc_true - done)
        idx = space.index(sample_rank_vectors(rng, mu, cfg.sigma, (b, 2 * n)))
        hits += np.sum(mmd(histograms(idx[:, :n], len(space)), histograms(idx[:, n:], len(space)), G) < eps)
        done += b
    return hits / cfg.n_mc_true


def estimated_reliability(cfg, rng, sample_idx, n, m, G) -> float:
    """Plug-in estimate from a single study: resample two samples of size n from its empirical distribution."""
    N, B = sample_idx.size, cfg.n_resamples
    if cfg.est_scheme == "with_replacement":
        pos = rng.integers(0, N, size=(B, 2 * n))
    elif cfg.est_scheme == "without_replacement":
        if 2 * n > N:
            raise ValueError("without_replacement requires 2n <= N")
        pos = rng.permuted(np.tile(np.arange(N), (B, 1)), axis=1)[:, : 2 * n]
    else:
        raise ValueError(cfg.est_scheme)
    idx = sample_idx[pos]
    return np.mean(mmd(histograms(idx[:, :n], m), histograms(idx[:, n:], m), G) < epsilon_star(cfg))


# =============================================================================
# Experiments
# =============================================================================
def run_grid(cfg: Config, space, G) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for delta in cfg.deltas:
        mu = mu_vector(cfg, delta)
        top = sample_rank_vectors(rng, mu, cfg.sigma, (200_000,))
        p_top0 = np.mean(top[:, 0] == 0)
        for N in cfg.Ns:
            R = sample_rank_vectors(rng, mu, cfg.sigma, (cfg.n_studies, N))
            sig, winner = significance(cfg, R)
            q = np.array([np.mean(sig & (winner == w)) for w in range(cfg.na)])   # Pr(sig, winner = w)
            rel = true_reliability(cfg, rng, space, G, mu, N)
            rows.append(dict(
                delta=delta, N=N, p_top0=p_top0,
                p_sig=sig.mean(),
                p_sig_winner0=q[0],
                p_sig_other=q[1:].sum(),
                p_replicate=(q ** 2).sum() / q.sum() if q.sum() > 0 else np.nan,
                reliability=rel,
                reliability_se=math.sqrt(rel * (1 - rel) / cfg.n_mc_true),
                reliable=rel >= cfg.alpha_star,
            ))
            print(f"Delta={delta:<5g} N={N:<4d} Pr(sig)={sig.mean():.3f}  "
                  f"Pr(rep|sig)={rows[-1]['p_replicate']:.3f}  R={rel:.3f}")
    return pd.DataFrame(rows)


def run_per_study(cfg: Config, space, G) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    rows = []
    for delta, N in cfg.per_study_cells:
        mu = mu_vector(cfg, delta)
        n = max(1, round(cfg.est_n_fraction * N))
        R = sample_rank_vectors(rng, mu, cfg.sigma, (cfg.n_studies, N))
        sig, winner = significance(cfg, R)
        idx = space.index(R)
        rel_true = true_reliability(cfg, rng, space, G, mu, n)
        for s in range(cfg.n_studies):
            group = 0 if not sig[s] else (1 if winner[s] == 0 else 2)
            rows.append(dict(delta=delta, N=N, n=n, study=s, significant=bool(sig[s]),
                             winner=int(winner[s]), group=group,
                             rhat=estimated_reliability(cfg, rng, idx[s], n, len(space), G),
                             reliability_true=rel_true))
    return pd.DataFrame(rows)


# =============================================================================
# Outputs
# =============================================================================
def set_style():
    plt.rcParams.update({
        "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE,
        "legend.title_fontsize": FONT_SIZE, "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "mathtext.fontset": "cm", "pdf.fonttype": 42,
    })


def plot_grid(cfg: Config, df: pd.DataFrame, path: Path):
    fig, (ax_p, ax_r) = plt.subplots(1, 2, figsize=(16, 6.5), sharex=True)
    colors = dict(zip(cfg.deltas, DELTA_COLORS * (len(cfg.deltas) // len(DELTA_COLORS) + 1)))
    for delta, d in df.groupby("delta"):
        kw = dict(color=colors[delta], lw=2.5, marker="o", ms=9, mec="white", mew=1.5, label=f"{delta:g}")
        ax_p.plot(d["N"], d["p_sig"], **kw)
        ax_r.plot(d["N"], d["reliability"], **kw)

    ax_p.axhline(cfg.alpha_test, color=INK_2, lw=1.5, ls=":")
    ax_p.text(df["N"].max(), cfg.alpha_test + 0.02, LABELS["alpha_test"], ha="right", va="bottom", color=INK_2)
    ax_r.axhline(cfg.alpha_star, color=INK_2, lw=1.5, ls="--")
    ax_r.text(df["N"].max(), cfg.alpha_star - 0.02, LABELS["alpha_star"], ha="right", va="top", color=INK_2)

    for ax, key in ((ax_p, "power"), (ax_r, "reliability")):
        ax.set_xlabel(LABELS["N"])
        ax.set_ylabel(LABELS[key])
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, df["N"].max() * 1.03)
    handles, labels = ax_r.get_legend_handles_labels()
    fig.legend(handles, labels, title=LABELS["delta_legend"], loc="upper center",
               ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_per_study(cfg: Config, df: pd.DataFrame, path: Path):
    cells = list(dict.fromkeys(zip(df["delta"], df["N"])))
    fig, axes = plt.subplots(1, len(cells), figsize=(8 * len(cells), 7), squeeze=False, sharey=True)
    rng = np.random.default_rng(0)
    for ax, (delta, N) in zip(axes[0], cells):
        d = df[(df["delta"] == delta) & (df["N"] == N)]
        for g, (label, color) in enumerate(zip(LABELS["groups"], GROUP_COLORS)):
            y = d.loc[d["group"] == g, "rhat"].to_numpy()
            if y.size == 0:
                continue
            x = g + rng.uniform(-0.25, 0.25, y.size)
            ax.scatter(x, y, s=18, color=color, alpha=0.35, lw=0)
            q1, med, q3 = np.percentile(y, [25, 50, 75])
            ax.plot([g - 0.32, g + 0.32], [med, med], color=INK, lw=2.5)
            ax.plot([g, g], [q1, q3], color=INK, lw=2.5)
            ax.text(g, -0.07, f"{y.size}", ha="center", va="top", color=INK_2)
        ax.axhline(d["reliability_true"].iloc[0], color=INK_2, lw=1.5, ls="--")
        ax.text(2.45, d["reliability_true"].iloc[0] + 0.015, LABELS["true_R"], ha="right", va="bottom", color=INK_2)
        ax.set_xticks(range(3), LABELS["groups"], rotation=20, ha="right")
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(-0.12, 1.02)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="x", visible=False)
        ax.set_title(LABELS["panel_title"].format(delta=delta, N=N))
    axes[0, 0].set_ylabel(LABELS["rhat"])
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def latex_table(cfg: Config, df: pd.DataFrame, path: Path, Ns=None):
    Ns = Ns or [N for N in cfg.Ns if N in (10, 20, 40, 80)] or list(cfg.Ns)
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{$N={N}$}}" for N in Ns)
    rules = " ".join(rf"\cmidrule(lr){{{2 * i + 3}-{2 * i + 4}}}" for i in range(len(Ns)))
    sub = " & ".join(r"Sig. & $R^{k}$" for _ in Ns)
    lines = [
        r"\begin{tabular}{r r " + "r r " * len(Ns) + "}",
        r"\toprule",
        rf"$\Delta$ & $\Pr(\text{{top}} = a_0)$ & {head} \\",
        rules,
        rf" & & {sub} \\",
        r"\midrule",
    ]
    for delta, d in df.groupby("delta"):
        d = d.set_index("N")
        cells = " & ".join(f"{d.loc[N, 'p_sig']:.2f} & {d.loc[N, 'reliability']:.2f}" for N in Ns)
        lines.append(rf"{delta:g} & {d['p_top0'].iloc[0]:.2f} & {cells} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("appendix_a_output"))
    parser.add_argument("--quick", action="store_true", help="small run for testing")
    parser.add_argument("--posthoc", default=None, help="override Config.posthoc")
    args = parser.parse_args()

    cfg = Config()
    if args.quick:
        cfg = replace(cfg, n_studies=200, n_mc_true=4000, n_resamples=200, Ns=(5, 10, 20, 40))
    if args.posthoc:
        cfg = replace(cfg, posthoc=args.posthoc)
    args.out.mkdir(parents=True, exist_ok=True)

    space = PermutationSpace(cfg.na)
    G = gram_matrix(cfg, space)
    print(f"kernel={cfg.kernel}  eps*={epsilon_star(cfg):.4f}  posthoc={cfg.posthoc}  |S_na|={len(space)}")

    grid = run_grid(cfg, space, G)
    grid.to_csv(args.out / "grid.csv", index=False)
    per_study = run_per_study(cfg, space, G)
    per_study.to_csv(args.out / "per_study.csv", index=False)

    set_style()
    plot_grid(cfg, grid, args.out / f"fig_power_reliability.{FIG_FORMAT}")
    plot_per_study(cfg, per_study, args.out / f"fig_estimated_reliability.{FIG_FORMAT}")
    latex_table(cfg, grid, args.out / "table_grid.tex")

    summary = (per_study.groupby(["delta", "N", "group"])["rhat"]
               .agg(["count", "mean", "std"]).round(3))
    print(summary)
    print(f"Outputs written to {args.out.resolve()}")


if __name__ == "__main__":
    main()