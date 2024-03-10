import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typing import Iterable, List


def remove_diagonal(mat: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the diagonal from a wide dataframe, useful for plotting heatmaps without diagonal.
    """
    tmp = mat.to_numpy()
    np.fill_diagonal(tmp, np.nan)
    return pd.DataFrame(tmp, index=mat.index, columns=mat.columns)


def separate_zeros(mat: pd.DataFrame, significant_digits: int) -> (pd.DataFrame, np.array):
    """
    Separate a wide dataframe into its zero (after rounding) and non-zero components, useful for plotting to have zeros
        in a different format.
    """
    zeros = np.zeros_like(mat)
    zeros[(mat.round(significant_digits) != 0) | (np.isnan(mat))] = np.nan
    mat[mat.round(significant_digits) == 0] = np.nan
    return mat, zeros


def separate_ones(mat: pd.DataFrame, significant_digits: int) -> (pd.DataFrame, np.array):
    """
    Separate a wide dataframe into its zero (after rounding) and non-zero components, useful for plotting to have zeros
        in a different format.
    """
    ones = np.ones_like(mat)
    ones[(mat.round(significant_digits) != 1) | (np.isnan(mat))] = np.nan
    mat[mat.round(significant_digits) == 1] = np.nan
    return mat, ones


def heatmap_int(zeros: np.array, ax: plt.axis, **heatmap_kws) -> plt.axis:
    return sns.heatmap(zeros, ax=ax, annot=True, fmt=".0f", square=True, cbar=False, **heatmap_kws)


def heatmap_(mat: pd.DataFrame, ax: plt.axis, significant_digits: int, **heatmap_kws):
    return sns.heatmap(mat, ax=ax, annot=True, fmt=f".{significant_digits}f", square=True, cbar=False, **heatmap_kws)


def heatmap_long(df_sim: pd.DataFrame,
                 ax: plt.axis,
                 similarities: List[str],
                 comparison_level: str,
                 summary_statistic: str,
                 significant_digits: int,
                 cmaps: List[mpl.colors.Colormap],
                 **heatmap_kws) -> plt.axis:

    assert len(similarities) == len(cmaps) == 2
    assert set(similarities).issubset(df_sim.columns)
    assert {f"{comparison_level}_1", f"{comparison_level}_2"}.issubset(df_sim.columns)

    for i, (similarity, cmap) in enumerate(zip(similarities, cmaps)):

        # compute statistic (mean, average) from df_sim
        cl = [f"{comparison_level}_1", f"{comparison_level}_2"]
        aggsim = df_sim[cl + [similarity]].groupby(cl).agg(summary_statistic).reset_index() \
            .pivot(index=cl[0], columns=cl[1]) \
            .droplevel([0], axis=1)

        if i == 1:
            aggsim = aggsim.T

        aggsim = remove_diagonal(aggsim)
        aggsim, zeros = separate_zeros(aggsim, significant_digits=significant_digits)
        aggsim, ones = separate_ones(aggsim, significant_digits=significant_digits)

        ax = heatmap_int(zeros, ax=ax, cmap=cmap, vmin=0, vmax=1, **heatmap_kws)
        ax = heatmap_int(ones, ax=ax, cmap=cmap, vmin=0, vmax=1, **heatmap_kws)
        ax = heatmap_(aggsim, ax=ax, significant_digits=significant_digits, cmap=cmap, vmin=0, vmax=1, **heatmap_kws)

    ax.set(xlabel=None, ylabel=None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    sns.despine()
    plt.tight_layout(pad=0.5)
    plt.show()

    return ax
