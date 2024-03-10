"""
Utility mddule to deal with relations.
Including conversion functions from rankings (arrays) to adjacency matrices and scores.
"""

from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd


def get_relation_properties(mat: np.array) -> list:
    """
    model is a domination (adjancecy) matrix
    """

    if len(mat.shape) != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Input must be a square 2-matrix, but has shape {mat.shape}.")

    na = mat.shape[0]

    def check(l):
        return len(l) == sum(l)

    properties = {
        "totality": check([mat[i, j] + mat[j, i] >= 1 for i, j in product(range(na), repeat=2) if i < j]),
        "reflexivity": check([mat[i, i] == 1 for i in range(na)]),
        "antisymmetry": check([mat[i, j] + mat[j, i] == 1 for i, j in product(range(na), repeat=2) if i < j]),
        "transitivity": check([mat[i, j] + mat[j, k] - mat[i, k] <= 1
                               for i, j, k in product(range(na), repeat=3) if i != j != k != i]),
        "acyclicity": check([mat[i, j] - mat[k, j] - mat[i, k] >= -1
                            for i, j, k in product(range(na), repeat=3) if i != j != k != i]),
    }
    return [p for p, satisfied in properties.items() if satisfied]


def score2rf(score: pd.Series, lower_is_better: bool = True, impute_missing: bool = True) -> pd.Series:
    """
    Rank the elements of 'score.index' according to 'score'.
    lower_is_better =
        True: lower score = better rank (for instance, if score is the result of a loss function or a ranking itself)
        False: greater score = better rank (for instance, if score is the result of a score such as roc_auc_score)
    """
    if impute_missing:
        score = score.fillna(score.max())
    c = 1 if lower_is_better else -1
    return score.map({s: sorted(score.unique(), key=lambda x: c * x).index(s) for s in score.unique()})


def vec2rf(arr: np.ndarray[int | float], lower_is_better: bool = True) -> np.ndarray[int | float]:
    """
    Rank the elements of 'arr' according to their value.
    lower_is_better =
        True: lower score = better rank (for instance, if arr is a loss function or a ranking)
        False: greater score = better rank (for instance, if arr is a score, such as roc_auc_score)
    """
    c = 1 if lower_is_better else -1
    # Unique sorted values and their inverse to rebuild the original array
    _, inverse = np.unique(c*arr, return_inverse=True)
    # Use the inverse indices which map each original value to its rank
    return inverse


def rf2mat(r: np.array, kind: str = "preference") -> np.array:
    """
    map a ranking to a matrix of kind 'kind'
    kind =
        preference: computes the antisymmetric preference matrix Mij = int(Ri < Rj), Mji = -Mij
        {domination, outranking}: computes the domination (outranking) matrix Mij = 1 iff (Ri <= Rj) else 0
        incidence: computes the incidence matrix of a strict linear order Mij = 1 iff R1 < Rj
        yoo: as preference, but Mij = 1 if Ri <= Rj, -1 if Ri > Rj, 0 if i=j
            Adapted from [1]

    [1] Yoo, Y., & Escobedo, A. R. (2021). A new binary programming formulation and social choice property for
        Kemeny rank aggregation. Decision Analysis, 18(4), 296-320.

    """
    na = len(r)     # num alternatives
    mat = - np.zeros((na, na))
    for i, j in product(range(na), repeat=2):
        if j > i:
            continue
        if kind == 'preference':
            mat[i, j] = np.sign((r[j] - r[i]))
            mat[j, i] = - mat[i, j]
        elif kind in {"domination", "outranking"}:
            mat[i, j] = int(r[i] <= r[j])
            mat[j, i] = int(r[j] <= r[i])
        elif kind == "incidence":
            mat[i, j] = int(r[i] < r[j])
            mat[j, i] = int(r[j] < r[i])
        elif kind == "yoo":
            if i == j:
                mat[i, j] = 0
            elif r[i] == r[j]:
                mat[i, j] = mat[j, i] = 1
            else:
                mat[i, j] = np.sign((r[j] - r[i]))
                mat[j, i] = - mat[i, j]
        else:
            raise ValueError(f"kind={kind} is not accepted.")

    return mat


def mat2rf(mat: np.array, alternatives: Iterable) -> pd.Series:
    """
    Get the ranking from a transitive adjacency matrix.
    """
    if "transitivity" not in get_relation_properties(mat):
        raise ValueError("The input matrix must be transitive.")
    return score2rf(pd.Series(np.sum(mat, axis=0), index=alternatives))


def dr2mat(dr: pd.DataFrame, kind: str = "preference") -> np.array:
    """
    Map each column of a dataframe of rankings ('dr.index' are the alternatives, 'dr.columns' are the voters) into an
        adjacency matrix of kind 'kind'.
    Gather the matrices into a 3-tensor 'ms' so that ms[ic] = ms[ic, :, :] is the adjacency matrix of the
        ic-th column of dr.
    """
    ms = np.zeros((dr.shape[1], dr.shape[0], dr.shape[0]))
    for ic, col in enumerate(dr.columns):
        ms[ic] = rf2mat(dr[col].to_numpy(), kind=kind)
    return ms


def get_constraints(mat: np.array, ranking_type: str = "total_order") -> list[bool]:
    """
    Get constraints on the adjacency matrix of a ranking 'mat', in order to satisfy properties defined by 'ranking_type'.

    consensus_kind =
        weak_order: totality and transitivity (reflexivity?)
        total_order: antisymmetry and transitivity
        strict_order: antisymmetry and transitivity
        yoo_weak_order: Adapted from Yoo (2021): acyclicity, totality,
        all: return all constraints
    """
    na = mat.shape[0]

    # ---  constraints
    totality = [
        mat[i, j] + mat[j, i] >= 1
        for i, j in product(range(na), repeat=2) if i < j
    ]
    reflexivity = [
        mat[i, i] == 1
        for i in range(na)
    ]
    antisymmetry = [
        mat[i, j] + mat[j, i] == 1
        for i, j in product(range(na), repeat=2) if i < j
    ]
    transitivity = [
        mat[i, j] + mat[j, k] - mat[i, k] <= 1
        for i, j, k in product(range(na), repeat=3) if i != j != k != i
    ]
    acyclicity = [
        mat[i, j] - mat[k, j] - mat[i, k] >= -1
        for i, j, k in product(range(na), repeat=3) if i != j != k != i
    ]

    if ranking_type == "total_order":
        return reflexivity + antisymmetry + transitivity
    elif ranking_type == "weak_order":
        return totality + transitivity
    elif ranking_type == "strict_order":
        return antisymmetry + transitivity
    elif ranking_type == "yoo_weak_order":
        return acyclicity + totality
    else:
        raise ValueError(f"consensus_kind = {ranking_type} is not a valid value.")
