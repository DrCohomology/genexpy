import numpy as np

from typing import Any, Callable, Literal, TypeAlias, Union

from . import rankings_utils as ru

Kernel: TypeAlias = Callable[[np.array, np.array, bool, Any], float]

RankVector: TypeAlias = np.ndarray[int]
RankByte: TypeAlias = bytes
Ranking: TypeAlias = Union[RankVector, RankByte]


# ---- Vectorized Gram matrices
@np.vectorize(signature="(n, na, na), (n, na, na), () -> (n, n)", otypes=[float])
def mallows_gram(ams1: ru.AdjacencyMatrix, ams2: ru.AdjacencyMatrix, nu: Union[float, Literal["auto"]] = "auto"):
    """
    Computes the Gram matrix of the Mallows kernel between two sets of rankings,
    represented as adjacency matrices.

    The Mallows kernel is a similarity measure between two adjacency matrices,
    based on the number of discordant pairs between the two corresponding rankings.

    Parameters
    ----------
    ams1 : ru.AdjacencyMatrix
        The first set of adjacency matrices, represented as a tensor of shape (n, na, na),
        where n is the number of matrices, and na is the number of alternatives.
    ams2 : ru.AdjacencyMatrix
        The second set of adjacency matrices, with the same shape as ams1.
    nu : float or "auto", optional
        The scaling parameter for the kernel. If "auto", it is set to 2 / (na*(na-1)),
        where na is the number of alternatives. The default is "auto".

    Returns
    -------
    ndarray
        A tensor of shape (n, n) representing the Gram matrix of the  Mallows kernel between
        the two sets of adjacency matrices.

    Raises
    ------
    ValueError
        If the two tensors do not have the same shape.

    See Also
    --------
    ru.AdjacencyMatrix : Class representing a ranking.

    Notes
    -----
    The Mallows kernel is defined as:

    .. math::
        K(A_1, A_2) = exp(-nu/2 * \sum_{i < j} |A_{1, i, j} - A_{2, i, j}|)

    where :math:`A_1` and :math:`A_2` are adjacency matrices, and :math:`nu` is a scaling parameter.

    Examples
    --------
    >>> from . import rankings_utils as ru
    >>> ams1 = ru.AdjacencyMatrix(np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]))
    >>> ams2 = ru.AdjacencyMatrix(np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]))
    >>> mallows_gram(ams1, ams2)
    array([[0.60653066, 0.36787944],
           [0.36787944, 0.60653066]])
    """
    if ams1.shape != ams2.shape:
        raise ValueError("The two tensors should have the same shape.")

    if nu == "auto":
        na = ams1.shape[1]
        nu = 2 / (na*(na-1))

    ndisc = np.logical_xor(np.expand_dims(ams1, axis=1), np.expand_dims(ams2, axis=0)).sum(axis=(-1, -2))
    return np.exp(-nu / 2 * ndisc)


@np.vectorize(signature="(na, n), (na, n), (), () -> (n, n)", otypes=[float])
def borda_gram(rv1: RankVector, rv2: RankVector, idx: int, nu: Union[float, Literal["auto"]] = "auto"):
    """
    Computes the Gram matrix of the Borda kernel between two sets of rankings, represented as vectors.

    The Borda kernel is a similarity measure between two rankings,
    based on the number of elements that are ranked higher than a given element in each ranking.

    Parameters
    ----------
    rv1 : RankVector
        The first set of rankings, represented as a tensor of shape (na, n),
        where na is the number of alternatives, and n is the number of rankings.
    rv2 : RankVector
        The second set of rank vectors, with the same shape as rv1.
    idx : int
        The index of the element to compare the rankings for.
    nu : float or "auto", optional
        The scaling parameter for the kernel. If "auto", it is set to 2 / (na*(na-1)),
        where na is the number of elements. The default is "auto".

    Returns
    -------
    ndarray
        A tensor of shape (n, n) representing the Gram matrix of the Borda kernel between the two sets of
        rankings.

    Raises
    ------
    ValueError
        If the two tensors do not have the same shape.

    See Also
    --------
    RankVector : Class representing a ranking.

    Notes
    -----
    The Borda kernel is defined as:

    .. math::
        K(R_1, R_2) = exp(-nu * |d_1 - d_2|)

    where :math:`R_1` and :math:`R_2` are rank vectors, :math:`d_1` is the number of elements
    ranked higher than the element at index `idx` in :math:`R_1`, and :math:`d_2` is the
    number of elements ranked higher than the element at index `idx` in :math:`R_2`.

    Examples
    --------
    >>> from . import ranking_utils as ru
    >>> rv1 = ru.RankVector(np.array([[1, 2, 3], [3, 1, 2]]))
    >>> rv2 = ru.RankVector(np.array([[2, 1, 3], [1, 3, 2]]))
    >>> borda_gram(rv1, rv2, idx=0)
    array([[0.60653066, 0.36787944],
           [0.36787944, 0.60653066]])
    """
    if rv1.shape != rv2.shape:
        raise ValueError("The two matrices should have the same shape.")

    if nu == "auto":
        na = rv1.shape[0]
        nu = 2 / (na*(na-1))

    d1 = np.sum(rv1 >= rv1[idx], axis=0)  # dominated
    d2 = np.sum(rv2 >= rv2[idx], axis=0)
    return np.exp(- nu * np.abs(np.expand_dims(d1, axis=1) - np.expand_dims(d2, axis=0)))


@np.vectorize(signature="(na, n), (na, n), () -> (n, n)", otypes=[float])
def jaccard_gram(rv1: RankVector, rv2: RankVector, k: int):
    """
    Computes the Gram matrix of the Jaccard kernel between two sets of rankings.

    The Jaccard kernel is a similarity measure between two sets of rankings,
    based on the number of elements ranked within a given cutoff in each ranking.

    Parameters
    ----------
    rv1 : RankVector
        The first set of rankings, represented as a tensor of shape (na, n),
        where na is the number of alternatives, and n is the number of vectors.
    rv2 : RankVector
        The second set of rank vectors, with the same shape as rv1.
    k : int
        The cutoff value for the ranking.

    Returns
    -------
    ndarray
        A tensor of shape (n, n) representing the Gram matrix of the Jaccard kernel between the two sets of
        rankings.

    See Also
    --------
    RankVector : Class representing a ranking.

    Notes
    -----
    The Jaccard kernel is defined as:

    .. math::
        K(R_1, R_2) = \frac{|R_1 \cap R_2|}{|R_1 \cup R_2|}

    where :math:`R_1` and :math:`R_2` are rank vectors, and :math:`R_1 \cap R_2`
    represents the set of elements ranked within the cutoff `k` in both vectors, and
    :math:`R_1 \cup R_2` represents the set of elements ranked within the cutoff `k`
    in either vector.

    Examples
    --------
    >>> from . import ranking_utils as ru
    >>> rv1 = ru.RankVector(np.array([[0, 1, 2], [2, 0, 1]]))
    >>> rv2 = ru.RankVector(np.array([[1, 0, 2], [0, 2, 1]]))
    >>> jaccard_gram(rv1, rv2, k=2)
    array([[0.66666667, 0.5       ],
           [0.5       , 0.66666667]])
    """
    k1 = rv1 < k
    k2 = rv2 < k
    intersection = np.logical_and(np.expand_dims(k1, 2), np.expand_dims(k2, 1)).astype(int).sum(axis=0)
    union = np.logical_or(np.expand_dims(k1, 2), np.expand_dims(k2, 1)).astype(int).sum(axis=0)
    return intersection / union


AVAILABLE_VECTORIZED_KERNELS = {  # available kernels and format of inputs. adjmat = adjacency matrix
    "borda": [borda_gram, "vector"],
    "jaccard": [jaccard_gram, "vector"],
    "mallows": [mallows_gram, "adjmat"],
}
