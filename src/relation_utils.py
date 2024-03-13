"""
Utility mddule to deal with relations.
Including conversion functions from rankings (arrays) to adjacency matrices and scores.
"""

from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

# Put score2rf and vec2rf into same function, cast pdSeries to array
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
