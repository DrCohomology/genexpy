"""
Utility module to deal with relations.

Including conversion functions from rankings (arrays) to adjacency matrices and scores.
"""

import numpy as np
import pandas as pd

def score2rv(score: pd.Series, lower_is_better: bool = True, impute_missing: bool = True) -> pd.Series:
    """
    Rank the elements of 'score.index' according to 'score'.

    Parameters
    ----------
    score : pd.Series
        The scores to be ranked.
    lower_is_better : bool, optional
        Whether lower scores are better. If True, lower scores will be assigned higher ranks.
        If False, higher scores will be assigned higher ranks. The default is True.
    impute_missing : bool, optional
        Whether to impute missing values in the score. If True, missing values will be
        imputed with the maximum score. The default is True.

    Returns
    -------
    pd.Series
        A Series containing the ranks of the elements in 'score.index'.

    Examples
    --------
    >>> import pandas as pd
    >>> score = pd.Series([1, 2, 3], index=['A', 'B', 'C'])
    >>> score2rv(score)
    A    0
    B    1
    C    2
    dtype: int64
    >>> score2rv(score, lower_is_better=False)
    A    2
    B    1
    C    0
    dtype: int64
    """
    if impute_missing:
        score = score.fillna(score.max())
    c = 1 if lower_is_better else -1
    return score.map({s: sorted(score.unique(), key=lambda x: c * x).index(s) for s in score.unique()})


def vec2rv(vec: np.ndarray[int | float], lower_is_better: bool = True) -> np.ndarray:
    """
    Rank the elements of 'vec' according to their value.

    Parameters
    ----------
    vec : np.ndarray
        The array to be ranked.
    lower_is_better : bool, optional
        Whether lower values are better. If True, lower values will be assigned higher ranks.
        If False, higher values will be assigned higher ranks. The default is True.

    Returns
    -------
    np.ndarray
        An array containing the ranks of the elements in 'vec'.

    Examples
    --------
    >>> vec = np.array([1, 2, 3])
    >>> vec2rv(vec)
    array([0, 1, 2])
    >>> vec2rv(vec, lower_is_better=False)
    array([2, 1, 0])
    """
    c = 1 if lower_is_better else -1
    # Unique sorted values and their inverse to rebuild the original array
    _, inverse = np.unique(c * vec, return_inverse=True)
    # Use the inverse indices which map each original value to its rank
    return inverse
