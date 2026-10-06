"""
Utility module for handling rankings.

A ranking of na alternatives is represented by its adjacency matrix A, with A[i, j] = int(r[i] <= r[j]),
where r is the rank vector of the ranking (r[i] is the rank of alternative i, 0 being the best; ties allowed).
For speed and hashability, adjacency matrices are stored as the bytes of their int8 entries.

- ``AdjacencyMatrix``: a single ranking.
- ``UniverseAM`` / ``SampleAM``: an array of rankings (bytes), e.g., the results of an experimental study.
- ``MultiSampleAM``: a 2-D array of rankings, i.e., many samples of the same size.
- ``get_matrix_from_df``: turn a dataframe of experimental results into a matrix of rankings.
"""

import numpy as np
import pandas as pd

from collections import Counter
from collections.abc import Collection
from typing import AnyStr, Iterable, Union

from genexpy.utils import relations as rlu


def _bytes_to_adjacency_matrices(arr, na: int, writable: bool = False) -> np.ndarray:
    """
    Decode an array of encoded adjacency matrices (bytes) of any shape into an int8 array of shape (*arr.shape, na, na).
    """
    arr = np.asarray(arr)
    flat = arr.ravel()
    if flat.dtype.kind == "S" and flat.dtype.itemsize == na * na:
        buffer = flat.tobytes()  # fixed-width bytes: the raw buffer already is the concatenation
    else:
        buffer = b"".join(flat.tolist())
    if len(buffer) != flat.size * na * na:
        raise ValueError(f"The rankings are not encodings of {na}x{na} adjacency matrices.")
    out = np.frombuffer(bytearray(buffer) if writable else buffer, dtype=np.int8)
    return out.reshape(*arr.shape, na, na)


def _dense_ranks_from_adjacency(A: np.ndarray) -> np.ndarray:
    """
    Rank vectors of adjacency matrices of shape (..., na, na): output of shape (..., na).

    The column sum s[j] = #{i: r[i] <= r[j]} is larger for worse alternatives; the dense rank of j is the number of
    distinct column sums smaller than s[j].
    """
    na = A.shape[-1]
    s = A.sum(axis=-2, dtype=np.int64)  # (..., na), values in [1, na]
    present = np.zeros(s.shape[:-1] + (na + 1,), dtype=np.int64)
    np.put_along_axis(present, s, 1, axis=-1)
    distinct_below = np.cumsum(present, axis=-1) - 1  # number of distinct values < v, for v present
    return np.take_along_axis(distinct_below, s, axis=-1)


def _adjacency_from_rank_vectors(rv: np.ndarray) -> np.ndarray:
    """Adjacency matrices of the rank vectors in the COLUMNS of rv, of shape (na, n): output of shape (n, na, na)."""
    r = np.asarray(rv).T  # (n, na)
    return (r[:, :, None] <= r[:, None, :]).astype(np.int8)


class AdjacencyMatrix(np.ndarray):
    """
    Class to represent a ranking as an adjacency matrix.

    The adjacency matrix M is constructed such that M[i, j] = int(R[i] <= R[j]),
    where R is the ranking.
    AdjacencyMatrix objects are hashable and can therefore be used as keys for
    dictionaries.

    Parameters
    ----------
    input_array : np.ndarray
        A 2D square array representing the adjacency matrix.

    Methods
    -------
    zero(na: int) -> AdjacencyMatrix
        Creates an adjacency matrix for the zero-ranking (everything tied), of size na x na.
    from_rank_vector(rv: Iterable) -> AdjacencyMatrix
        Constructs an adjacency matrix from a rank vector.
    from_bytes(bytestring: bytes, shape: Iterable[int]) -> AdjacencyMatrix
        Creates an adjacency matrix from a bytestring.
    tohashable() -> bytes
        Converts the adjacency matrix to a hashable byte representation.
    to_rank_vector() -> np.ndarray
        Returns the (dense) rank vector of the ranking.
    get_ntiers() -> int
        Returns the number of unique ranks (tiers) in the adjacency matrix.
    """

    __slots__ = ()

    def __new__(cls, input_array):
        assert len(input_array.shape) == 2, "Wrong number of dimensions."
        assert input_array.shape[0] == input_array.shape[1], "An adjacency matrix is always square."
        assert np.all(input_array == input_array.astype(bool).astype(int)), "Matrix is not boolean."
        return np.asarray(input_array).view(cls)

    @classmethod
    def zero(cls, na):
        """Creates the adjacency matrix of the zero-ranking (all alternatives tied), of size na x na: all ones."""
        return np.ones((na, na), dtype=int).view(cls)

    @classmethod
    def from_rank_vector(cls, rv: Iterable) -> "AdjacencyMatrix":
        """
        Constructs an adjacency matrix from a rank vector.

        Parameters
        ----------
        rv : Iterable
            A rank vector: rv[i] is the rank of alternative i (lower is better).

        Returns
        -------
        AdjacencyMatrix
            An adjacency matrix representation of the rank vector.
        """
        r = np.asarray(rv)
        return (r[:, None] <= r[None, :]).astype(int).view(cls)

    @classmethod
    def from_bytes(cls, bytestring: bytes, shape: Iterable[int]) -> "AdjacencyMatrix":
        """
        Creates an adjacency matrix from a bytestring.

        Parameters
        ----------
        bytestring : bytes
            A bytestring representation of the adjacency matrix.
        shape : Iterable[int]
            The shape of the matrix.

        Returns
        -------
        AdjacencyMatrix
            An adjacency matrix constructed from the bytestring.
        """
        return np.frombuffer(bytestring, dtype=np.int8).reshape(shape).view(cls)

    def tohashable(self) -> bytes:
        """Converts the adjacency matrix to a hashable byte representation."""
        return self.astype(np.int8).tobytes()

    def to_rank_vector(self) -> np.ndarray:
        """Returns the dense rank vector of the ranking (0 = best)."""
        return _dense_ranks_from_adjacency(np.asarray(self))

    def get_ntiers(self) -> int:
        """
        Returns the number of unique ranks (tiers) in the adjacency matrix.

        Returns
        -------
        int
            The number of unique ranks in the adjacency matrix.
        """
        return len(set(np.sum(self, axis=1)))

    def __hash__(self) -> int:
        """Computes the hash of the adjacency matrix based on its byte representation."""
        return hash(self.tohashable())


class UniverseAM(np.ndarray):
    """
    Class representing a set of AdjacencyMatrix objects.

    This class stores binary encodings of adjacency matrices in an array-like
    structure. The dtype is set to object to preserve the integrity of the
    adjacency matrices' representations.

    Parameters
    ----------
    input_iter : Iterable
        An iterable of AdjacencyMatrix objects or their hashable representations.

    Methods
    -------
    to_adjmat_array(shape: Iterable[int]) -> np.ndarray
        Converts the support of hashes back to an array of AdjacencyMatrix objects.
    merge(other: UniverseAM) -> UniverseAM
        Merges with another UniverseAM instance, retaining unique entries.
    """

    def __new__(cls, input_iter: Iterable):
        try:
            return np.asarray([x.tohashable() for x in input_iter], dtype=object).view(cls)
        except AttributeError:
            if isinstance(input_iter, np.ndarray) or isinstance(input_iter, UniverseAM):
                return input_iter.view(cls)
            raise ValueError("Invalid input to UniverseAM.")

    def to_adjmat_array(self, shape: Iterable[int]) -> np.ndarray:
        """
        Converts the binary encodings back to adjacency matrices.

        Parameters
        ----------
        shape : Iterable[int]
            The shape (na, na) of the adjacency matrices to be reconstructed.

        Returns
        -------
        np.ndarray
            An int8 array of shape (len(self), na, na).
        """
        na = tuple(shape)[0]
        return _bytes_to_adjacency_matrices(self, na, writable=True)

    def __contains__(self, bstring: bytes) -> bool:
        """
        Checks if a bytestring representation is contained in the support.

        Parameters
        ----------
        bstring : bytes
            The bytestring to check for presence in the support.

        Returns
        -------
        bool
            True if the bytestring is present, False otherwise.
        """
        return np.any(np.isin(self, bstring))

    def _get_na_nv(self) -> None:
        """Determines the number of alternatives (methods) and sets attributes."""
        na = np.sqrt(len(self[0]))
        assert na == int(na), "Wrong length"
        self.na = int(na)
        self.nv = len(self)

    def get_na(self) -> int:
        """
        Returns the number of alternatives (methods).

        Returns
        -------
        int
            The number of alternatives in the support.
        """
        self._get_na_nv()
        return self.na

    def merge(self, other: 'UniverseAM') -> 'UniverseAM':
        """
        Merges with another UniverseAM instance, retaining unique entries.

        Parameters
        ----------
        other : UniverseAM
            The other UniverseAM instance to merge with.

        Returns
        -------
        UniverseAM
            A new UniverseAM instance containing unique entries from both.
        """
        return np.unique(np.append(self, other)).view(UniverseAM)


class SampleAM(UniverseAM):
    """
    Class representing a sample of rankings (e.g., the results of an experimental study), stored as the bytes of
    their adjacency matrices.

    Attributes
    ----------
    rv : np.ndarray, optional
        The rank vector matrix representation of the sample.
    ntiers : int, optional
        The number of tiers per ranking in the sample.

    Methods
    -------
    from_rank_vector_dataframe(rv: pd.DataFrame) -> SampleAM
        Constructs a SampleAM from a DataFrame of rank vectors.
    from_rank_vector_matrix(rv_matrix: np.ndarray) -> SampleAM
        Converts a rank function matrix into a SampleAM object.
    to_rank_vector_matrix() -> np.ndarray
        Returns a matrix of ranks arranged by method and voter.
    get_rank_vector_matrix() -> np.ndarray
        Retrieves or computes the rank vector matrix representation.
    set_key(key: Collection) -> None
        Sets a key for entries in the sample.
    get_subsamples_pair(subsample_size: int, seed: int, use_key: bool = False, replace: bool = False,
                            disjoint: bool = True) -> tuple[SampleAM, SampleAM]:
        Draws two subsamples from the sample.
    get_subsample(subsample_size: int, seed: int, use_key: bool = False, replace: bool = False) -> SampleAM:
        Draws a single subsample from the sample.
    get_multisample_pair(subsample_size: int, rep: int, seed: int, disjoint: bool, replace: bool)
        Draws `rep` pairs of subsamples.
    get_support_pmf() -> tuple[SampleAM, np.ndarray]:
        Returns the support of unique rankings and their probability mass function (PMF).

    Examples
    --------
    >>> rv = np.array([[0, 1], [1, 0], [2, 2]])     # 3 alternatives (rows), 2 rankings (columns)
    >>> sample = SampleAM.from_rank_vector_matrix(rv)
    >>> sample.to_rank_vector_matrix()
    """

    rv = None  # rank vector matrix representation of the sample
    ntiers = None  # number of tiers per ranking in the sample

    def __new__(cls, *args, **kwargs):
        """Creates a new instance of SampleAM."""
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def from_rank_vector_dataframe(cls, rv: pd.DataFrame) -> 'SampleAM':
        """
        Constructs a SampleAM from a DataFrame of rank vectors.

        Parameters
        ----------
        rv : pd.DataFrame
            DataFrame where each row represents an alternative and each column
            represents a voter. For instance, in benchmarking, a voter is an
            experimental condition. every experimental condition produces a
            ranking of the benchmarked alternatives.

        Returns
        -------
        SampleAM
            A SampleAM instance.
        """
        return cls.from_rank_vector_matrix(rv.to_numpy())

    @classmethod
    def from_rank_vector_matrix(cls, rv_matrix: np.ndarray) -> 'SampleAM':
        """
        Converts a rank function matrix into a SampleAM object.

        Parameters
        ----------
        rv_matrix : np.ndarray
            A matrix where each row represents an alternative and each column
            represents an experimental condition or voter.

        Returns
        -------
        SampleAM
            A SampleAM instance constructed from the rank function matrix.
        """
        A = _adjacency_from_rank_vectors(rv_matrix)  # (n, na, na)
        out = np.empty(A.shape[0], dtype=object)
        out[:] = [a.tobytes() for a in A]
        return out.view(cls)

    def to_rank_vector_matrix(self) -> np.ndarray:
        """
        Returns a matrix of ranks arranged by method and voter.

        The output matrix contains ranks such that out[i, j] is the (dense) rank of
        alternative (method) i according to voter (experimental condition) j.

        Returns
        -------
        np.ndarray
            A matrix of ranks of shape (na, nv).
        """
        self._get_na_nv()
        A = _bytes_to_adjacency_matrices(self, self.na)  # (nv, na, na)
        return _dense_ranks_from_adjacency(A).T

    def get_rank_vector_matrix(self) -> np.ndarray:
        """
        Retrieves or computes the rank vector matrix representation.

        If the rank vector matrix has not been computed, it computes it and
        sets the rv attribute.

        Returns
        -------
        np.ndarray
            The rank vector matrix representation of the sample.
        """
        if self.rv is None:
            self.rv = self.to_rank_vector_matrix()
        return self.rv

    def set_key(self, key: Collection) -> 'SampleAM':
        """
        Set the key of entries. key must have the same length as self.
        Useful for advanced sampling, e.g., sampling datasets.
        Entries of key may not be unique, the idea is that to every key are associated multiple elements of self.

        Parameters
        ----------
        key : Collection
            A collection representing the key for the sample entries.

        Returns
        -------
        SampleAM
            The updated SampleAM instance with the set key.
        """
        assert len(key) == len(self), f"Entered key has length {len(key)}, while it should have length {len(self)}"
        self.key = np.array(key)
        return self

    def get_subsamples_pair(self, subsample_size: int, seed: int, use_key: bool = False, replace: bool = False,
                            disjoint: bool = True) -> tuple['SampleAM', 'SampleAM']:
        """
        Draws two subsamples from the sample.

        Parameters
        ----------
        subsample_size : int
            The size of each subsample.
        seed : int
            The random seed to use for subsampling.
        use_key : bool, optional
            Deprecated, must be False.
        replace : bool, optional
            If True, sample with replacement. Allow repetitions within a subsample.
            The default is False.
        disjoint : bool, optional
            If True, the returned subsamples have disjoint indices.
            The default is True.

        Returns
        -------
        tuple[SampleAM, SampleAM]
            A tuple containing the two subsamples.

        Raises
        ------
        ValueError
            If use_key is True, or if the subsample size is too large.
        """

        if use_key:
            raise ValueError("use_key = True is not accepted anymore.")

        max_size = len(self)
        max_size //= 2 if disjoint else 1

        if not replace and subsample_size > max_size:
            raise ValueError(f"Size of subsamples is too large, must be at most {max_size}.")

        rng = np.random.default_rng(seed)

        if disjoint and replace:  # get two disjoint subsamples, then samples from them
            shuffled = rng.choice(self, len(self), replace=False)
            out1 = rng.choice(shuffled[:len(self) // 2], subsample_size, replace=True)
            out2 = rng.choice(shuffled[len(self) // 2:], subsample_size, replace=True)
        elif disjoint or replace:  # implies replace = not disjoint
            out1, out2 = rng.choice(self, 2*subsample_size, replace=replace).reshape(2, subsample_size)
        else:  # if not disjoint and no replacement, we just sample twice
            out1 = rng.choice(self, subsample_size, replace=False)
            out2 = rng.choice(self, subsample_size, replace=False)

        return SampleAM(out1), SampleAM(out2)

    def get_subsample(self, subsample_size: int, seed: Union[int, np.random.Generator], use_key: bool = False,
                      replace: bool = False) -> 'SampleAM':
        """
        Get a subsample of self.

        Parameters
        ----------
        subsample_size : int
            The size of the subsample.
        seed : int or np.random.Generator
            The random seed (or generator) to use for subsampling.
        use_key : bool, optional
            Deprecated, must be False.
        replace : bool, optional
            If True, sample with replacement. The default is False.

        Returns
        -------
        SampleAM
            A subsample of the original sample.

        Raises
        ------
        ValueError
            If use_key is True, or if the subsample size is too large.
        """

        if use_key:
            raise ValueError("use_key = True is not accepted anymore.")

        max_size = len(self)

        if not replace and subsample_size > max_size:
            raise ValueError(f"Size of subsamples is too large, must be at most {max_size}.")

        return SampleAM(np.random.default_rng(seed).choice(self, subsample_size, replace=replace))

    def get_support_pmf(self) -> tuple['SampleAM', np.ndarray]:
        """
        Returns the support of unique rankings and their probability mass function (PMF).

        Returns
        -------
        tuple[SampleAM, np.ndarray]
            The unique rankings (in order of first appearance) and their relative frequencies.
        """
        counter = Counter(self)
        support = SampleAM(np.array(list(counter.keys())))
        pmf = np.array(list(counter.values()), dtype=float)
        return support, pmf / np.sum(pmf)

    def get_ntiers(self):
        """
        Number of tiers of every ranking in the sample, as an array of shape (len(self), ).
        """
        if self.ntiers is None:
            self.get_rank_vector_matrix()
            self.ntiers = np.max(self.rv, axis=0) - np.min(self.rv, axis=0) + 1
        return self.ntiers

    def partition_with_ntiers(self) -> dict:
        """
        Split self according to the number of tiers of its rankings.
        Return a dictionary {ntiers: SampleAM of the rankings with ntiers tiers}.
        """
        ntiers = self.get_ntiers()
        return {nt: self[ntiers == nt] for nt in np.unique(ntiers)}

    def append(self, other):
        """Return a new sample with the rankings of `other` appended."""
        return np.append(self, other).view(SampleAM)

    # ---- Draw many pairs of subsamples at once.
    # Every method returns two MultiSampleAM of shape (rep, n), self has shape (N, ).

    def _multisample_disjoint_replace(self, rep: int, n: int, rng: np.random.Generator):
        """
        Get 'rep' pairs of subsamples of size 'n', sampled with replacement from disjoint subsamples of 'self'.

        Algorithm:
        1. Get rep copies of sample (rep, N).
        2. Shuffle each row independently.
        3. Split every row (roughly) in half and sample with replacement from each half independently.
        """
        N = len(self)
        samples = np.broadcast_to(np.expand_dims(self, axis=0), (rep, N))  # (rep, N)
        shuffled = rng.permuted(samples, axis=1)
        subs1 = np.array([rng.choice(sub, n, replace=True) for sub in shuffled[:, :N // 2]])  # (rep, n)
        subs2 = np.array([rng.choice(sub, n, replace=True) for sub in shuffled[:, N // 2:]])  # (rep, n)

        return MultiSampleAM(subs1), MultiSampleAM(subs2)

    def _multisample_disjoint_not_replace(self, rep: int, n: int, rng: np.random.Generator):
        """
        Get 'rep' pairs of subsamples of size 'n', sampled without replacement from disjoint subsamples of 'self'.

        Algorithm:
        1. Get rep copies of self (rep, N).
        2. Shuffle each row independently.
        3. Split every row (roughly) in half and sample without replacement from each half independently.
        """
        N = len(self)
        samples = np.broadcast_to(np.expand_dims(self, axis=0), (rep, N))  # (rep, N)
        shuffled = rng.permuted(samples, axis=1)
        subs1 = np.array([rng.choice(sub, n, replace=False) for sub in shuffled[:, :N // 2]])  # (rep, n)
        subs2 = np.array([rng.choice(sub, n, replace=False) for sub in shuffled[:, N // 2:]])  # (rep, n)

        return MultiSampleAM(subs1), MultiSampleAM(subs2)

    def _multisample_not_disjoint_replace(self, rep: int, n: int, rng: np.random.Generator):
        """
        Get 'rep' pairs of samples of size 'n', sampled with replacement from 'self'.

        Algorithm:
        1. Get rep copies of sample (rep, N).
        2. Get a sample of size 2n with replacement from each row independently.
        3. Split the rows in half.
        """
        N = len(self)
        samples = np.broadcast_to(np.expand_dims(self, axis=0), (rep, N))  # (rep, N)
        tmp = np.array([rng.choice(sub, 2 * n, replace=True) for sub in samples])  # (rep, 2*n)
        subs1 = tmp[:, :n]
        subs2 = tmp[:, n:]

        return MultiSampleAM(subs1), MultiSampleAM(subs2)

    def _multisample_not_disjoint_not_replace(self, rep: int, n: int, rng: np.random.Generator):
        """
        Get 'rep' pairs of samples of size 'n', each sampled without replacement from 'self' (the two samples of a
        pair can overlap).

        Algorithm:
        1. Get rep copies of self (rep, N).
        2. Get two independent samples without replacement of size n from each row.
        """
        N = len(self)
        samples = np.broadcast_to(np.expand_dims(self, axis=0), (rep, N))  # (rep, N)
        subs1 = np.array([rng.choice(sub, n, replace=False) for sub in samples])  # (rep, n)
        subs2 = np.array([rng.choice(sub, n, replace=False) for sub in samples])  # (rep, n)

        return MultiSampleAM(subs1), MultiSampleAM(subs2)

    def get_multisample_pair(self, subsample_size: int, rep: int, seed: int, disjoint: bool = True,
                             replace: bool = False) -> tuple['MultiSampleAM', 'MultiSampleAM']:
        """
        Get 'rep' pairs of subsamples of size 'subsample_size', sampled from 'self' (which has shape (N, )).

        Parameters
        ----------
        subsample_size : int
            Size n of every subsample.
        rep : int
            Number of pairs.
        seed : int
            Random seed.
        disjoint : bool
            If True, the two subsamples of a pair are drawn from two disjoint halves of 'self'.
        replace : bool
            If True, the sampling is with replacement.

        Returns
        -------
        tuple[MultiSampleAM, MultiSampleAM]
            Two arrays of shape (rep, n); row r of the first is paired with row r of the second.
        """

        rng = np.random.default_rng(seed)

        match (disjoint, replace):
            case (True, True):
                return self._multisample_disjoint_replace(rep=rep, n=subsample_size, rng=rng)
            case (True, False):
                return self._multisample_disjoint_not_replace(rep=rep, n=subsample_size, rng=rng)
            case (False, True):
                return self._multisample_not_disjoint_replace(rep=rep, n=subsample_size, rng=rng)
            case (False, False):
                return self._multisample_not_disjoint_not_replace(rep=rep, n=subsample_size, rng=rng)


class MultiSampleAM(np.ndarray):
    """
    A sample of samples (a 2d sample).

    This class represents a collection of samples, where each sample is itself a
    collection of adjacency matrices. It provides methods for converting between
    different representations of the multi-sample, such as rank vectors and
    adjacency matrices.

    Methods
    -------
    to_rank_vectors() -> np.ndarray
        Converts the multi-sample to a representation of rank vectors.
    to_adjacency_matrices(na: int) -> np.ndarray
        Converts the multi-sample to a representation of adjacency matrices.
    get_pmfs(support) -> np.ndarray
        Empirical pmf of every sample over a common support.
    get_alpha(other, support) -> tuple[np.ndarray, UniverseAM]
        Difference of the empirical pmfs of two multi-samples.
    """

    def __new__(cls, input_iter: Iterable):
        """Creates a new instance of MultiSampleAM."""
        return np.asarray(input_iter).view(cls)

    def to_rank_vectors(self) -> np.ndarray:
        """
        Converts the multi-sample to a representation of rank vectors.

        Returns
        -------
        np.ndarray
            A 3D array of shape (rep, na, n) representing the rank vectors,
            where rep is the number of samples, na is the number of alternatives,
            and n is the size of each sample.
        """
        a = np.asarray(self)
        na = int(np.sqrt(len(a.flat[0])))
        A = _bytes_to_adjacency_matrices(a, na)  # (rep, n, na, na)
        return np.swapaxes(_dense_ranks_from_adjacency(A), -1, -2)  # (rep, na, n)

    def to_adjacency_matrices(self, na: int) -> np.ndarray:
        """
        Converts the multi-sample to a representation of adjacency matrices.

        Parameters
        ----------
        na : int
            The number of alternatives in each adjacency matrix.

        Returns
        -------
        np.ndarray
            A 4D int8 array of shape (rep, n, na, na) representing the adjacency matrices,
            where rep is the number of samples, n is the size of each sample,
            and na is the number of alternatives.
        """
        return _bytes_to_adjacency_matrices(self, na, writable=True)  # (rep, n, na, na)

    def get_pmfs(self, support: UniverseAM) -> np.ndarray:
        """
        Empirical pmf of every sample in the multi-sample, over a common support.

        Parameters
        ----------
        support : UniverseAM
            The rankings indexing the output, in the order they are to appear.
            Must contain every ranking occurring in the multi-sample, and must
            not contain duplicates.

        Returns
        -------
        np.ndarray
            A 2D array of shape (rep, m), row r holding the pmf of sample r over
            `support`, where rep is the number of samples and m is the size of
            the support.

        Raises
        ------
        ValueError
            If a ranking in the multi-sample is absent from `support`.
        """
        a = np.asarray(self)
        rep, n = a.shape
        index = pd.Index(np.asarray(support))  # raises if the support has duplicates
        m = len(index)

        codes = index.get_indexer(a.ravel())  # -1 where absent
        if codes.min() < 0:
            raise ValueError("There are rankings in the sample that are not contained in the support.")

        # offsetting sample r by r * m gives every sample a disjoint block of the
        # count vector, so one bincount fills the whole (rep, m) array
        offset = m * np.arange(rep, dtype=np.int64)[:, None]
        flat = (codes.reshape(rep, n) + offset).ravel()

        return np.bincount(flat, minlength=rep * m).reshape(rep, m) / n

    def get_alpha(self, other: "MultiSampleAM", support: UniverseAM = None) -> tuple[np.ndarray, UniverseAM]:
        """
        Signed difference of the empirical pmfs of two multi-samples.

        Column r of the output is the coefficient vector of the difference of
        kernel mean embeddings of self[r] and other[r], so that the squared MMD
        between them is alpha[:, r] @ K @ alpha[:, r], with K the Gram matrix of
        the returned support.

        Parameters
        ----------
        other : MultiSampleAM
            A multi-sample with the same number of samples as self. self[r] is
            compared with other[r].
        support : UniverseAM, optional
            The rankings indexing the output. If None, the union of the two
            multi-samples is used, ordered by first appearance.

        Returns
        -------
        alpha : np.ndarray
            A 2D array of shape (m, rep).
        support : UniverseAM
            The support indexing the rows of alpha. The Gram matrix must be
            built on this, not on an independently computed support.

        Raises
        ------
        ValueError
            If the two multi-samples have different numbers of samples, or if a
            ranking is absent from a `support` that was passed explicitly.
        """
        if self.shape[0] != other.shape[0]:
            raise ValueError(f"The multi-samples hold {self.shape[0]} and {other.shape[0]} samples.")

        if support is None:
            support = SampleAM(pd.unique(np.concatenate([np.asarray(self).ravel(),
                                                         np.asarray(other).ravel()])))

        return (self.get_pmfs(support) - other.get_pmfs(support)).T, support

    def get_pmfs_df(self, support: UniverseAM = None) -> pd.DataFrame:
        """
        Legacy (slow) version of get_pmfs: a dataframe with the rankings as index and the samples as columns.
        If support is not None, the index of the output contains `support`.
        """
        tmps = [pd.Series(index=support, name="support_tmp")]
        for i, s in enumerate(self):
            support_lcl, pmf = SampleAM(s).get_support_pmf()
            if support is not None and not set(support_lcl).issubset(support):
                raise ValueError("There are rankings in the sample that are not contained in self.support (which is "
                                 "not None)")

            tmps.append(pd.Series(pmf, index=support_lcl))

        return pd.concat(tmps, axis=1, ignore_index=False).drop(columns="support_tmp").fillna(0)


def get_matrix_from_df(df: pd.DataFrame, factors: Iterable, alternatives: AnyStr, target: AnyStr,
                       impute_missing=True, tol_missing_indices: float = 0.2,
                       tol_missing_columns: float = 0.2,
                       get_rankings: bool = True, lower_is_better: bool = True,
                       as_numpy: bool = False) -> Union[pd.DataFrame, np.ndarray]:
    """
    Pivot a dataframe of experimental results into a matrix alternatives x conditions, of scores or of rankings.

    Every combination of levels of `factors` is an experimental condition (a column of the output); every
    alternative is a row (sorted by name). If `get_rankings`, every column is converted to the (dense) ranks of the
    alternatives, 0 being the best.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    factors : Iterable
        The columns whose combinations of levels define the experimental conditions.
    alternatives : AnyStr
        The name of the column containing the alternatives to be ranked.
    target : AnyStr
        The name of the column containing the values to rank by.
    impute_missing : bool, optional
        Whether to impute missing evaluations. A missing evaluation is never better than an observed one: it is
        imputed with the worse of 0 and the worst observed value of its condition (for non-negative scores, this
        is 0). The default is True.
    tol_missing_indices : float, optional
        Maximum allowed fraction of missing alternatives for a condition (column) to be kept.
    tol_missing_columns : float, optional
        Maximum allowed fraction of missing conditions for an alternative (index) to be kept.
    get_rankings : bool, optional
        If True, return ranks instead of the target values. The default is True.
    lower_is_better : bool, optional
        Whether lower values of `target` are better (True for errors, False for scores). The default is True.
    as_numpy : bool, optional
        If True, return a numpy array instead of a DataFrame.

    Returns
    -------
    pd.DataFrame or np.ndarray
        The matrix of shape (alternatives, conditions); the columns are indexed by the levels of `factors`.

    Raises
    ------
    ValueError
        If any of the factors, alternatives, or target columns are not present in the DataFrame.
    """

    if not set(factors).issubset(df.columns):
        raise ValueError("factors must be an iterable of columns of df.")
    if alternatives not in df.columns:
        raise ValueError("alternatives must be a column of df.")
    if target not in df.columns:
        raise ValueError("target must be a column of df.")

    out = df.reset_index(drop=True).pivot(index=alternatives, columns=factors, values=target)

    # filter out columns
    out = out.loc[:, out.isna().mean(axis=0) <= tol_missing_indices]
    out = out.loc[out.isna().mean(axis=1) <= tol_missing_columns, :]

    if impute_missing:
        if lower_is_better:  # errors: the worst value is the largest
            fill = np.maximum(out.max(axis=0), 0)
        else:  # scores: the worst value is the smallest
            fill = np.minimum(out.min(axis=0), 0)
        out = out.fillna(fill)

    if get_rankings:
        if out.isna().to_numpy().any():  # score2rv imputes the missing values left
            out = out.apply(lambda x: rlu.score2rv(x, lower_is_better=lower_is_better))
        else:  # same as score2rv on every column, in one call
            out = (out.rank(axis=0, method="dense", ascending=lower_is_better) - 1).astype(np.int64)

    if as_numpy:
        return out.to_numpy()

    return out
