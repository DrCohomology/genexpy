"""
Utility module to deal with relations.

Conversion of scores (or errors) into rank vectors: rank 0 is the best, tied values get the same rank,
and ranks are dense (0, 1, 1, 2 rather than 0, 1, 1, 3).
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
        Whether lower scores are better, i.e., get lower (better) ranks. The default is True.
    impute_missing : bool, optional
        Whether to impute missing values. If True, missing values are imputed with the worst score
        (the maximum if lower_is_better, the minimum otherwise), i.e., they tie with the worst alternative.
        If False, missing values get a missing rank. The default is True.

    Returns
    -------
    pd.Series
        The (dense) rank of the elements in 'score.index', 0 being the best.

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
        score = score.fillna(score.max() if lower_is_better else score.min())
    out = score.rank(method="dense", ascending=lower_is_better) - 1
    return out if out.isna().any() else out.astype(np.int64)


def vec2rv(vec: np.ndarray[int | float], lower_is_better: bool = True) -> np.ndarray:
    """
    Rank the elements of 'vec' according to their value.

    Parameters
    ----------
    vec : np.ndarray
        The array to be ranked.
    lower_is_better : bool, optional
        Whether lower values are better, i.e., get lower (better) ranks. The default is True.

    Returns
    -------
    np.ndarray
        The (dense) rank of the elements of 'vec', 0 being the best.

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
    _, inverse = np.unique(c * np.asarray(vec), return_inverse=True)
    # Use the inverse indices which map each original value to its rank
    return inverse
